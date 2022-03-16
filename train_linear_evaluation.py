"""
Main routine for training a top linear layer on top of the NNCLR model.

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
from models.nnclr.nnclr import NNCLR

# Load a configuration file
configuration = Configuration(Mode.evaluation)

# Logging information will be saved in a file 'debug_le_{seed}.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug_le_{}.log".format(configuration.seed)),
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
data_loader = DataLoaderSSL(configuration, data_paths)

# Use an efficientB0 backbone
backbone = torchvision.models.efficientnet_b0()
backbone = nn.Sequential(*list(backbone.children())[:-1])

# Training procedure:
logging.info("Linear evaluation of the NNCLR model ...")
data_loader.mode = Mode.evaluation
data_loader.batch_size = configuration.le_conf.batch_size
data_loader.create_data_loader()
backbone = NNCLR.load_state_dict_(backbone, configuration.nnclr_conf.checkpoint)  # load a saved backbone
linear_eval = LinearEval(backbone, data_loader.classes, data_loader.class_weights)  # initialize a classifier
linear_eval.to(configuration.device)
linear_eval.train_(configuration, data_loader.train_loader)
