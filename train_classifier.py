"""
Main routine for training a top linear layer on top of the NNCLR model.

@author Vadym Gryshchuk
"""

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoaderSSL, Mode
from data_processing.data_reader import DataReader
from data_processing.utils import set_seeds, set_logging
from models.nnclr.linear_eval import ClassificationModel
from models.nnclr.nnclr import NNCLR, get_convnext


def execute(conf, freeze_backbone):
    # Set-up:
    set_logging(seed, "classification_model")  # logging
    set_seeds(seed)  # set seed for the reproducibility of the results

    # Data:
    data_paths = DataReader(conf.caps_directories, conf.info_data_files, conf.diagnoses_info,
                            conf.quality_check, conf.valid_dataset_names, conf.col_names)
    data_loader = DataLoaderSSL(conf, data_paths, Mode.evaluation)
    data_loader.batch_size = conf.cls_conf.batch_size
    data_loader.create_data_loader()

    # Training procedure:
    backbone = get_convnext()
    backbone = NNCLR.load_state_dict_(backbone, conf.cls_conf.checkpoint_load)  # load a saved backbone
    linear_eval = ClassificationModel(backbone, data_loader.classes, data_loader.class_weights, freeze_backbone)
    linear_eval.to(conf.device)
    linear_eval.train_(conf, data_loader.train_loader)


configuration = Configuration(Mode.evaluation)  # Load a configuration file
for seed in configuration.seeds:
    if configuration.cls_conf.comparison:  # if True, then two models will be trained
        execute(conf=configuration, freeze_backbone=False)
        execute(conf=configuration, freeze_backbone=True)
    else:
        execute(conf=configuration, freeze_backbone=False)


