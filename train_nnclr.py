"""
Main routine for the training routine of the NNCLR model.

@author Vadym Gryshchuk
"""

import logging

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoaderSSL
from data_processing.data_reader import DataReader
from data_processing.utils import Mode, set_seeds, set_logging
from models.nnclr.nnclr import NNCLR, get_convnext

# Load a configuration file
configuration = Configuration(Mode.training)

for seed in configuration.seeds:

    # Set-up:
    set_logging(seed, "nnclr")  # logging
    set_seeds(seed)  # set seed for the reproducibility of the results

    # Data:
    data_paths = DataReader(configuration.caps_directories, configuration.info_data_files, configuration.diagnoses_info,
                            configuration.quality_check, configuration.valid_dataset_names, configuration.col_names)
    data_loader = DataLoaderSSL(configuration, data_paths, Mode.training)  # batch data loader
    data_loader.batch_size = configuration.nnclr_conf.batch_size  # set correct batch size
    data_loader.create_data_loader()  # create a data loader

    #  Training procedure:
    backbone = get_convnext()
    backbone = NNCLR.load_state_dict_(backbone, configuration.nnclr_conf.checkpoint_resume)  # load a saved backbone
    model = NNCLR(backbone, freeze_layers=configuration.nnclr_conf.trainable_layers)  # initialize the NNCLR model
    model.to(configuration.device)  # move the model to a device: either cpu or gpu
    model.train_(configuration, data_loader.train_loader)  # train the model
