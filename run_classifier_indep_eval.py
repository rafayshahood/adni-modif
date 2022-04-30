"""
The main routine for the evaluation of the model.

@author Vadym Gryshchuk
"""

import torch

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoader, Mode
from data_processing.data_reader import DataReader
from data_processing.utils import set_logging, set_seed
from models.classifier import ClassificationModel, LOG_IDENTIFIER_INDEPENDENT_EVALUATION
from models.nnclr import get_convnext

configuration = Configuration(mode=Mode.independent_evaluation)  # Load a configuration file
set_logging(log_dir=configuration.logs_folder, suffix=LOG_IDENTIFIER_INDEPENDENT_EVALUATION)  # logging
set_seed(seed=configuration.ind_eval_conf.seed)  # set seed for the reproducibility of the results

# Data:
data_paths = DataReader(caps_directories=configuration.caps_directories,
                        info_data=configuration.info_data_files,
                        diagnoses_info=configuration.diagnoses_info,
                        quality_check=configuration.quality_check,
                        valid_dataset_names=configuration.valid_dataset_names,
                        info_data_cols=configuration.col_names)
data_loader = DataLoader(configuration=configuration,
                         data=data_paths,
                         mode=Mode.independent_evaluation)
data_loader.batch_size = configuration.ind_eval_conf.batch_size
data_loader.create_data_loader(shuffle=False)

# Evaluation procedure:
backbone = get_convnext()
linear_eval = ClassificationModel(backbone,
                                  num_classes=torch.load(configuration.ind_eval_conf.checkpoint_load,
                                                         map_location=configuration.device)["classifier"][
                                      "top_layer.bias"].shape[0])
linear_eval.load(file_path=configuration.ind_eval_conf.checkpoint_load,
                 device=configuration.device)  # load a saved model
linear_eval.to(configuration.device)
data_loader.eval_loader.dataset.dataset.middle_slice = True
linear_eval.test_(configuration=configuration, test_loader=data_loader.eval_loader)  # one run for evaluation
data_loader.eval_loader.dataset.dataset.middle_slice = False
linear_eval.test_ext(configuration=configuration, test_loader=data_loader.eval_loader)  # multiple runs for evaluation
linear_eval.extract_features(configuration=configuration, data_loader=data_loader.eval_loader,
                             file_name="independent_eval")
