"""
Main routine for the training procedure of a classification block on top of the NNCLR model.

@author Vadym Gryshchuk
"""
import logging

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoader, Mode
from data_processing.data_reader import DataReader
from data_processing.utils import set_seed, set_logging
from models.nnclr.classifier import ClassificationModel, LOG_IDENTIFIER_CLASSIFIER
from models.nnclr.nnclr import NNCLR, get_convnext


def execute(conf: Configuration, freeze_backbone: bool):
    """
    Executes the training procedure
    :param conf: Configuration
    :param freeze_backbone: if True, then a backbone will be frozen, otherwise not.
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
    backbone = NNCLR.load_state_dict_(feature_extractor=backbone,
                                      checkpoint=conf.cls_conf.backbone_checkpoint)  # load a saved backbone
    linear_eval = ClassificationModel(feature_extractor=backbone,
                                      num_classes=data_loader.classes,
                                      class_weights=data_loader.class_weights,
                                      freeze_backbone=freeze_backbone)
    linear_eval.to(conf.device)
    linear_eval.set_name(seed=seed, freeze_backbone=freeze_backbone, conf_id=configuration.id)
    logging.info(">>> Training")
    linear_eval.train_(configuration=conf, train_loader=data_loader.train_loader)

    # Evaluation procedure:
    logging.info(">>> Evaluation")
    linear_eval.test_(configuration=configuration, test_loader=data_loader.eval_loader)  # one run for evaluation
    linear_eval.test_ext(configuration=configuration,
                         test_loader=data_loader.eval_loader)  # multiple runs for evaluation
    linear_eval.extract_features(configuration=configuration, data_loader=data_loader.train_loader, file_name="train")
    linear_eval.extract_features(configuration=configuration, data_loader=data_loader.eval_loader, file_name="test")


configuration = Configuration(mode=Mode.classifier)  # Load a configuration file
set_logging(log_dir=configuration.logs_folder, suffix=LOG_IDENTIFIER_CLASSIFIER)  # logging

search_backbone = False
for seed in configuration.seeds:

    if len(configuration.cls_conf.backbone_checkpoint) == 0:
        search_backbone = True
        configuration.cls_conf.backbone_checkpoint = "{}{}".format(configuration.checkpoints_folder,
                                                                   NNCLR.get_name_as_string(seed, configuration.id))

    if configuration.cls_conf.comparison:  # if True, then two models will be trained
        execute(conf=configuration, freeze_backbone=False)
        execute(conf=configuration, freeze_backbone=True)
    else:
        execute(conf=configuration, freeze_backbone=False)

    if search_backbone:
        configuration.cls_conf.backbone_checkpoint = ""
