"""
Main routine for the training procedure of a classification block on top of the NNCLR model.

@author Vadym Gryshchuk
"""
import logging

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoader, Mode
from data_processing.data_reader import DataReader
from data_processing.utils import set_seed, set_logging
from models.classifier import ClassificationModel, LOG_IDENTIFIER_CLASSIFIER
from models.nnclr import NNCLR, get_convnext


def execute(conf: Configuration, freeze_backbone: bool, backbone_ckpt: str):
    """
    Executes the training procedure
    :param conf: Configuration
    :param freeze_backbone: if True, then a backbone will be frozen, otherwise not
    :param backbone_ckpt: Backbone checkpoint
    """
    set_seed(seed)  # set seed for the reproducibility of the results
    logging.info("SEED: {}".format(seed))
    logging.info("Freeze backbone: {}".format(freeze_backbone))

    # Data:
    data_paths = DataReader(caps_directories=conf.caps_directories,
                            info_data=conf.info_data_files,
                            diagnoses_info=conf.diagnoses_info,
                            quality_check=conf.quality_check,
                            valid_dataset_names=conf.valid_dataset_names,
                            info_data_cols=conf.col_names)
    data_loader = DataLoader(configuration=conf,
                             data=data_paths,
                             mode=Mode.classifier)
    data_loader.batch_size = conf.cls_conf.batch_size
    data_loader.create_data_loader()

    # Training procedure:
    backbone = get_convnext()
    if freeze_backbone:
        backbone = NNCLR.load_state_dict_(feature_extractor=backbone,
                                          checkpoint=backbone_ckpt)  # load a saved backbone
    cls = ClassificationModel(feature_extractor=backbone,
                              num_classes=data_loader.classes,
                              class_weights=data_loader.class_weights,
                              freeze_backbone=freeze_backbone)
    cls.to(conf.device)
    cls.set_name(seed=seed, freeze_backbone=freeze_backbone, conf_id=configuration.id)
    logging.info(">>> Training")
    cls.train_(configuration=conf, train_loader=data_loader.train_loader)


configuration = Configuration(mode=Mode.classifier)  # Load a configuration file
set_logging(log_dir=configuration.logs_folder, suffix=LOG_IDENTIFIER_CLASSIFIER)  # logging

search_backbone = False
for seed in configuration.seeds:

    backbone_checkpoint = "{}{}".format(configuration.checkpoints_folder,
                                        NNCLR.get_name_as_string(seed, configuration.id))

    if configuration.cls_conf.comparison:  # if True, then two models will be trained
        execute(conf=configuration, freeze_backbone=True, backbone_ckpt=backbone_checkpoint)
        execute(conf=configuration, freeze_backbone=False, backbone_ckpt=backbone_checkpoint)
    else:
        execute(conf=configuration, freeze_backbone=False, backbone_ckpt=backbone_checkpoint)
