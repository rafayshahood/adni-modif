"""
Main routine for the training procedure of the NNCLR model.

@author Vadym Gryshchuk
"""
import logging

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoader
from data_processing.data_reader import DataReader
from data_processing.utils import Mode, set_seed, set_logging
from models.nnclr.nnclr import NNCLR, get_convnext

# Load a configuration file and set logging
configuration = Configuration(mode=Mode.training)
set_logging(configuration.logs_folder, suffix="nnclr")

for seed in configuration.seeds:
    # Set-up seeds:
    logging.info("Seed: {}".format(seed))
    set_seed(seed=seed)  # set seed for the reproducibility of the results

    # Data:
    data_paths = DataReader(caps_directories=configuration.caps_directories,
                            info_data=configuration.info_data_files,
                            diagnoses_info=configuration.diagnoses_info,
                            quality_check=configuration.quality_check,
                            valid_dataset_names=configuration.valid_dataset_names,
                            info_data_cols=configuration.col_names)
    data_loader = DataLoader(configuration=configuration, data=data_paths, mode=Mode.training)  # batch data loader
    data_loader.batch_size = configuration.nnclr_conf.batch_size  # set correct batch size
    data_loader.create_data_loader()  # create a data loader

    #  Training procedure:
    backbone = get_convnext()
    backbone = NNCLR.load_state_dict_(feature_extractor=backbone,
                                      checkpoint=configuration.nnclr_conf.checkpoint_resume)  # load a saved backbone
    model = NNCLR(backbone, freeze_layers=configuration.nnclr_conf.trainable_layers)  # initialize the NNCLR model
    model.to(configuration.device)  # move the model to a device: either cpu or gpu
    model.set_name(seed, configuration.id)
    model.train_(configuration=configuration, data_loader=data_loader.train_loader)  # train the model
