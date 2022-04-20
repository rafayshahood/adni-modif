import torch
import yaml

from data_processing.utils import Mode, create_folder


class NNCLRConfiguration:
    """
    Configuration for the NNCLR model
    """

    def __init__(self, settings: dict, ckpt_dir: str) -> None:
        self.epochs = settings['epochs']
        self.batch_size = settings['batch_size']
        self.checkpoint_folder = ckpt_dir
        self.checkpoint_resume = settings['checkpoint_resume']
        self.save_nepoch = settings['save_nepoch']
        self.trainable_layers = settings['trainable_layers']


class ClassificationModelConfiguration:
    """
    Configuration for the linear evaluation of the NNCLR model
    """

    def __init__(self, settings: dict, ckpt_dir: str):
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
    Configuration for the independent linear evaluation of the NNCLR model
    """

    def __init__(self, settings: dict):
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
        with open('./configuration/configuration.yaml', 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- general ---
            self.work_dir = settings['working_dir']
            self.id = settings['id']
            self.seeds = settings['seeds']
            self.dry_run = settings['dry_run']
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.checkpoints_folder = create_folder(self.work_dir, "checkpoints/")
            self.logs_folder = create_folder(self.work_dir, "logs/")
            self.features_folder = create_folder(self.work_dir, "features/")
            self.figures_folder = create_folder(self.work_dir, "figures/")

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
            self.nnclr_conf = NNCLRConfiguration(settings['nnclr'], ckpt_dir=self.checkpoints_folder)

            # --- Linear evaluation ---
            self.cls_conf = ClassificationModelConfiguration(settings['classifier'], ckpt_dir=self.checkpoints_folder)

            # --- Independent linear evaluation ---
            self.ind_eval_conf = IndependentEvaluationConfiguration(settings['independent_eval'])

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
