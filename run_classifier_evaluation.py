"""
Main routine for the training procedure of a classification block on top of the NNCLR model.

@author Vadym Gryshchuk
"""
import logging

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoader, Mode
from data_processing.data_reader import DataReader
from data_processing.utils import set_seed, set_logging, SEED
from models.classifier import ClassificationModel, LOG_IDENTIFIER_CLASSIFIER, LOG_IDENTIFIER_CLASSIFIER_EVALUATION
from models.nnclr import get_convnext


def execute(conf: Configuration, ckpt_folder: str):
    """
    Executes the training procedure
    :param conf: Configuration
    :param freeze_backbone: if True, then a backbone will be frozen, otherwise not.
    """
    set_seed(seed)  # set seed for the reproducibility of the results
    logging.info("{} {}".format(SEED, seed))

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
    data_loader.create_data_loader(shuffle=False)

    # Classifier:
    backbone = get_convnext()
    cls_eval = ClassificationModel(feature_extractor=backbone,
                                   num_classes=data_loader.classes)
    cls_eval.load(ckpt_folder, configuration.device)  # load a saved model
    cls_eval.to(conf.device)

    # Evaluation procedure:
    logging.info(">>> Evaluation")
    data_loader.eval_loader.dataset.dataset.middle_slice = True
    cls_eval.test_(configuration=configuration, test_loader=data_loader.eval_loader)  # one run for evaluation
    data_loader.eval_loader.dataset.dataset.middle_slice = False
    cls_eval.test_ext(configuration=configuration,
                      test_loader=data_loader.eval_loader)  # multiple runs for evaluation
    cls_eval.extract_features(configuration=configuration, data_loader=data_loader.train_loader, file_name="train")
    cls_eval.extract_features(configuration=configuration, data_loader=data_loader.eval_loader, file_name="test")


configuration = Configuration(mode=Mode.classifier)  # Load a configuration file
set_logging(log_dir=configuration.logs_folder, suffix=LOG_IDENTIFIER_CLASSIFIER_EVALUATION)  # logging

for seed in configuration.seeds:
    logging.info("Evaluating classifier with a frozen backbone")
    checkpoint_load = "{}{}".format(configuration.checkpoints_folder,
                                    ClassificationModel.get_name_as_string(seed, True, configuration.id))
    execute(conf=configuration, ckpt_folder=checkpoint_load)

    logging.info("Evaluating classifier with a re-trained backbone")
    checkpoint_load = "{}{}".format(configuration.checkpoints_folder,
                                    ClassificationModel.get_name_as_string(seed, False, configuration.id))
    execute(conf=configuration, ckpt_folder=checkpoint_load)
