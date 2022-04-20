"""
Main routine for the training procedure of a classification block on top of the NNCLR model.

@author Vadym Gryshchuk
"""
import os

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoaderSSL, Mode
from data_processing.data_reader import DataReader
from data_processing.utils import set_seeds, set_logging
from models.nnclr.linear_eval import ClassificationModel
from models.nnclr.nnclr import NNCLR, get_convnext


def execute(conf: Configuration, freeze_backbone: bool):
    """
    Executes the training procedure
    :param conf: Configuration
    :param freeze_backbone: if True, then a backbone will be frozen, otherwise not.
    """
    set_seeds(seed)  # set seed for the reproducibility of the results

    # Data:
    data_paths = DataReader(conf.caps_directories, conf.info_data_files, conf.diagnoses_info,
                            conf.quality_check, conf.valid_dataset_names, conf.col_names)
    data_loader = DataLoaderSSL(conf, data_paths, Mode.evaluation)
    data_loader.batch_size = conf.cls_conf.batch_size
    data_loader.create_data_loader()

    # Training procedure:
    backbone = get_convnext()
    backbone = NNCLR.load_state_dict_(backbone, conf.cls_conf.backbone_checkpoint)  # load a saved backbone
    linear_eval = ClassificationModel(backbone, data_loader.classes, data_loader.class_weights, freeze_backbone)
    linear_eval.to(conf.device)
    linear_eval.set_name(seed, configuration.id)
    linear_eval.train_(conf, data_loader.train_loader)


configuration = Configuration(Mode.evaluation)  # Load a configuration file
set_logging(configuration.logs_folder, suffix="classification_model")  # logging

for seed in configuration.seeds:

    if len(configuration.cls_conf.backbone_checkpoint) == 0:
        configuration.cls_conf.backbone_checkpoint = "{}{}".format(configuration.checkpoints_folder,
                                                                   NNCLR.get_name_as_string(seed, configuration.id))

    if configuration.cls_conf.comparison:  # if True, then two models will be trained
        execute(conf=configuration, freeze_backbone=False)
        execute(conf=configuration, freeze_backbone=True)
    else:
        execute(conf=configuration, freeze_backbone=False)


