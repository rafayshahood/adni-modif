import os
import random
from enum import Enum

import numpy as np
import torch
from torch.backends import cudnn


class Mode(str, Enum):
    """"
    A helper class to differentiate between different stages: training, training/evaluation of a classifier, and independent evaluation
    """
    training = 'training'
    classifier = 'classifier'
    independent_evaluation = 'independent_evaluation'


def create_folder(root, folder_name):
    """
    Creates a folder.
    :param root: Path to the location
    :param folder_name: a folder name
    :return: full path to the folder
    """
    if not os.path.exists(root):
        os.makedirs(root)
    output_dir = os.path.join(os.path.abspath(root), folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def set_logging(log_dir, suffix):
    """
    Creates a logging file.
    :param log_dir: a logging directory
    :param suffix: an additional identifier that will be appended to the folder name
    """
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("{}debug_{}.log".format(log_dir, suffix)),
            logging.StreamHandler()
        ]
    )


def set_seed(seed):
    """
    Sets seed for reproducible results.
    :param seed: int value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
