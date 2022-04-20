"""
Main routine for the evaluation of the NNCLR model. Only an additional classification block on top
of the frozen NNCLR model is trained.

@author Vadym Gryshchuk
"""

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoaderSSL, Mode
from data_processing.data_reader import DataReader
from data_processing.utils import set_logging, set_seeds
from models.nnclr.linear_eval import ClassificationModel

from models.nnclr.nnclr import get_convnext

configuration = Configuration(Mode.evaluation)  # Load a configuration file

for seed in configuration.seeds:
    # Set-up:
    set_logging(seed, "evaluation_nnclr")  # logging
    set_seeds(seed)  # set seed for the reproducibility of the results

    # Data:
    data_paths = DataReader(configuration.caps_directories, configuration.info_data_files, configuration.diagnoses_info,
                            configuration.quality_check, configuration.valid_dataset_names, configuration.col_names)
    data_loader = DataLoaderSSL(configuration, data_paths, Mode.evaluation)
    data_loader.batch_size = configuration.cls_conf.batch_size
    data_loader.create_data_loader()

    # Evaluation:
    backbone = get_convnext()
    linear_eval = ClassificationModel(backbone, data_loader.classes)
    linear_eval.load(configuration.cls_conf.checkpoint_folder_save, configuration.device)  # load a saved model
    linear_eval.to(configuration.device)
    linear_eval.test_(configuration, data_loader.eval_loader)  # one run for evaluation
    linear_eval.test_ext(configuration, data_loader.eval_loader)  # multiple runs for evaluation
    linear_eval.extract_features(configuration, data_loader.train_loader, "train")
    linear_eval.extract_features(configuration, data_loader.eval_loader, "test")
