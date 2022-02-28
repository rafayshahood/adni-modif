"""
Main routine for training of the NNCLR model.

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
from data_processing.data_loader import DataLoaderSSL
from data_processing.data_reader import DataReader
from models.nnclr.nnclr import NNCLR
from data_processing.utils import Mode

# Logging information will be saved in a file 'debug.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Load a configuration file
configuration = Configuration(Mode.training)

# Assure that seed is set
random.seed(configuration.seed)
np.random.seed(configuration.seed)
torch.manual_seed(configuration.seed)
cudnn.deterministic = True

# An object referencing the paths to files
data_paths = DataReader(configuration.caps_directories, configuration.info_data_files, configuration.diagnoses_info)

# A data loader
data_loader = DataLoaderSSL(configuration, data_paths)

# Use an efficientB4 backbone
backbone = torchvision.models.efficientnet_b4(pretrained=True)  # pretrained model is loaded
backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove classification layer

#  Training procedure:
logging.info("NNCLR Training ...")
data_loader.mode = Mode.training  # set correct mode
data_loader.batch_size = configuration.nnclr_conf.batch_size  # set correct batch size
data_loader.create_data_loader()  # create a data loader
model = NNCLR(backbone)  # initialize the NNCLR model
model.to(configuration.device)  # move the model to a device: either cpu or gpu
model.train_(configuration, data_loader.train_loader)  # train the model