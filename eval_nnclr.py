"""
Main routine for evaluating the NNCLR model using a top layer trained previously on top of NNCLR.

@author Vadym Gryshchuk
"""

import logging
import random

import numpy as np
import torch
import torchvision
from torch import nn
from torch.backends import cudnn

from configuration.configuration import Configuration
from data_processing.data_loader import DataLoaderSSL, Mode
from data_processing.data_reader import DataReader
from models.nnclr.linear_eval import LinearEval

# Load a configuration file
configuration = Configuration(Mode.evaluation)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/mnt/ssd2/ClinicNET/log/debug_eval_{}.log".format(configuration.seed)),
        logging.StreamHandler()
    ]
)

# Assure that seed is set
random.seed(configuration.seed)
np.random.seed(configuration.seed)
torch.manual_seed(configuration.seed)
cudnn.deterministic = True

# An object referencing the paths to files
data_paths = DataReader(configuration.caps_directories, configuration.info_data_files, configuration.diagnoses_info,
                        configuration.quality_check, configuration.valid_dataset_names, configuration.col_names)

# A data loader
data_loader = DataLoaderSSL(configuration, data_paths, Mode.evaluation)

# Use an convnext_tiny backbone
backbone = torchvision.models.convnext_tiny()
backbone.features[0][0] = nn.Conv2d(1, 96, (4, 4), (4, 4))
backbone = nn.Sequential(*list(backbone.children())[:-1])

logging.info("Evaluation of the NNCLR model on the test set...")
data_loader.mode = Mode.evaluation
data_loader.batch_size = configuration.le_conf.batch_size
data_loader.create_data_loader()

linear_eval = LinearEval(backbone, data_loader.classes)
linear_eval.load(configuration.le_conf.checkpoint_save, configuration.device)  # load a saved model
linear_eval.to(configuration.device)

logging.info("Test ...")
linear_eval.test_(configuration, data_loader.eval_loader)  # one run for evaluation
logging.info("Extended test ...")
linear_eval.test_ext(configuration, data_loader.eval_loader)  # multiple runs for evaluation

logging.info("Feature extraction ...")
linear_eval.extract_features(configuration, data_loader.train_loader, "train")
linear_eval.extract_features(configuration, data_loader.eval_loader, "test")
