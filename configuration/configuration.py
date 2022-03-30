import random

import numpy as np
import torch
import yaml
from torch.backends import cudnn

from data_processing.utils import Mode


class NNCLRConfiguration:
    """
    Configuration for the NNCLR model
    """
    def __init__(self, settings: dict) -> None:
        self.epochs = settings['epochs']
        self.batch_size = settings['batch_size']
        self.checkpoint = settings['checkpoint']
        self.checkpoint_resume = settings['checkpoint_resume']
        self.save_nepoch = settings['save_nepoch']
        self.trainable_layers = settings['trainable_layers']


class LinearEvaluationConfiguration:
    """
    Configuration for the linear evaluation of the NNCLR model
    """
    def __init__(self, settings: dict):
        self.epochs = settings['epochs']
        self.batch_size = settings['batch_size']
        self.checkpoint_load = settings['checkpoint_load']
        self.checkpoint_save = settings['checkpoint_save']
        self.replicas = settings['replicas']
        self.replicas_extraction = settings['replicas_extraction']
        self.eval_labels = settings['eval_labels']


class IndependentLinearEvaluationConfiguration:
    """
    Configuration for the independent linear evaluation of the NNCLR model
    """
    def __init__(self, settings: dict):
        self.batch_size = settings['batch_size']
        self.checkpoint_load = settings['checkpoint_load']
        self.replicas = settings['replicas']
        self.replicas_extraction = settings['replicas_extraction']
        self.eval_labels = settings['eval_labels']


class LRPConfiguration:
    """
    Configuration for LRP
    """
    def __init__(self, settings: dict):
        self.batch_size = settings['batch_size']
        self.checkpoint = settings['checkpoint']
        self.is_train_set = settings['is_train_set']
        self.classes = settings['classes']


class Configuration:
    """
    Configuration for all components
    """
    def __init__(self, mode):
        with open('./configuration/configuration.yaml', 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- general ---
            self.seed = settings['seed']
            self.dry_run = settings['dry_run']
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # --- data ---
            data = self.get_data(settings, mode)
            self.caps_directories = data[0]['caps_directories']
            self.info_data_files = data[1]['info_data_files']

            data = settings['data']
            self.slices_range = data['slices_range']
            self.features_out = data['features_out']
            self.diagnoses_info = data['diagnoses_info']
            self.quality_check = data['quality_check']
            self.valid_dataset_names = data['valid_dataset_names']
            self.col_names = data['col_names']

            # --- NNCLR ---
            self.nnclr_conf = NNCLRConfiguration(settings['nnclr'])

            # --- Linear evaluation ---
            self.le_conf = LinearEvaluationConfiguration(settings['linear_eval'])

            # --- Independent linear evaluation ---
            self.ind_le_conf = IndependentLinearEvaluationConfiguration(settings['independent_linear_eval'])

            # --- LRP ---
            self.lrp_conf = LRPConfiguration(settings['lrp'])

    def set_seeds(self, seed: int = None) -> None:
        """
        Set seed to assure the reproducibility of results
        :param seed: a seed as int
        """
        new_seed = self.seed if seed is None else seed
        random.seed(new_seed)
        np.random.seed(new_seed)
        torch.manual_seed(new_seed)
        cudnn.deterministic = True

    @staticmethod
    def get_data(settings, mode):
        if mode == Mode.training:
            data = settings['nnclr']['data']
        elif mode == Mode.evaluation:
            data = settings['nnclr']['data']
        elif mode == Mode.independent_evaluation:
            data = settings['independent_linear_eval']['data']
        else:
            raise ValueError("Mode {} is not recognized".format(mode))

        return data



