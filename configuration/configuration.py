import torch
import yaml

from data_processing.utils import Mode, create_folder


class NNCLRConfiguration:
    """
    Configuration for the NNCLR model
    """

    def __init__(self, settings: dict, ckpt_dir: str) -> None:
        """
        Initialises with all required parameters
        :param settings: settings as dict object
        :param ckpt_dir: checkpoint directory
        """
        self.epochs = settings['epochs']
        self.batch_size = settings['batch_size']
        self.checkpoint_folder = ckpt_dir
        self.checkpoint_resume = settings['checkpoint_resume']
        self.save_nepoch = settings['save_nepoch']
        self.trainable_layers = settings['trainable_layers']


class ClassifierConfiguration:
    """
    Configuration for the classifier
    """

    def __init__(self, settings: dict, ckpt_dir: str):
        """
        Initialises with all required parameters
        :param settings: settings as dict object
        :param ckpt_dir: checkpoint directory
        """
        self.epochs = settings['epochs']
        self.batch_size = settings['batch_size']
        self.backbone_checkpoint = settings['backbone_checkpoint']
        self.checkpoint_folder_save = ckpt_dir
        self.replicas = settings['replicas']
        self.replicas_extraction = settings['replicas_extraction']
        self.eval_labels = settings['eval_labels']
        self.comparison = settings['comparison']


class IndependentEvaluationConfiguration:
    """
    Configuration for the independent evaluation of the model
    """

    def __init__(self, settings: dict):
        """
        Initialises with all required parameters
        :param settings: settings as dict object
        """
        self.seed = settings['seed']
        self.batch_size = settings['batch_size']
        self.checkpoint_load = settings['checkpoint_load']
        self.replicas = settings['replicas']
        self.replicas_extraction = settings['replicas_extraction']
        self.eval_labels = settings['eval_labels']


class Configuration:
    """
    Configuration for all components
    """

    def __init__(self, mode: Mode):
        """
        Initialises with all required parameters
        :param mode: Mode, e.g. for the training of the NNCLR/classifier or independent evaluation
        """
        with open('./configuration/configuration.yaml', 'r') as stream:
            self.settings = yaml.load(stream, yaml.Loader)

            # --- general ---
            self.work_dir = self.settings['working_dir']
            self.id = self.settings['id']
            self.seeds = self.settings['seeds']
            self.dry_run = self.settings['dry_run']
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.checkpoints_folder = create_folder(self.work_dir, "checkpoints/")
            self.logs_folder = create_folder(self.work_dir, "logs/")
            self.features_folder = create_folder(self.work_dir, "features/")

            # --- data ---
            data = self.get_data(mode)
            self.caps_directories = data[0]['caps_directories']
            self.info_data_files = data[1]['info_data_files']

            data = self.settings['data']
            self.slices_range = data['slices_range']
            self.diagnoses_info = data['diagnoses_info']
            self.quality_check = data['quality_check']
            self.valid_dataset_names = data['valid_dataset_names']
            self.col_names = data['col_names']

            # --- NNCLR ---
            self.nnclr_conf = NNCLRConfiguration(self.settings['nnclr'], ckpt_dir=self.checkpoints_folder)

            # --- Linear evaluation ---
            self.cls_conf = ClassifierConfiguration(self.settings['classifier'], ckpt_dir=self.checkpoints_folder)

            # --- Independent linear evaluation ---
            self.ind_eval_conf = IndependentEvaluationConfiguration(self.settings['independent_evaluation'])

    def get_data(self, mode):
        """
        Returns the appropriate data configuration based on the selected settings
        :param mode: Mode object, e.g. for the training of the NNCLR/classifier or independent evaluation
        :return: data configuration
        """
        if mode == Mode.training:
            data = self.settings['nnclr']['data']
        elif mode == Mode.classifier:
            data = self.settings['nnclr']['data']
        elif mode == Mode.independent_evaluation:
            data = self.settings['independent_evaluation']['data']
        else:
            raise ValueError("Mode {} is not recognized".format(mode))

        return data
