"""
Evaluation script for running an independent evaluation of a trained classification model.
"""

import torch
from configuration.configuration import Configuration
from data_processing.data_loader import DataLoader, Mode
from data_processing.data_reader import DataReader
from data_processing.utils import set_logging, set_seed
from models.classifier import ClassificationModel, LOG_IDENTIFIER_INDEPENDENT_EVALUATION
from models.nnclr import get_convnext

def load_model(checkpoint_path, device, num_classes):
    """ 
    Loads the model from a given checkpoint path.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print("Top-level keys in checkpoint:", checkpoint.keys())

        backbone = get_convnext()  # Initialize backbone without pretrained weights
        model = ClassificationModel(feature_extractor=backbone, num_classes=num_classes)

        # Load state dicts separately for backbone and classifier
        model.feature_extractor.load_state_dict(checkpoint['feature_extractor'], strict=False)
        model.classifier.load_state_dict(checkpoint['classifier'], strict=False)

        model.to(device)
        return model
    except FileNotFoundError:
        print(f"Checkpoint file not found at {checkpoint_path}. Please check the file path.")
        exit()
    except KeyError as e:
        print(f"Error loading the checkpoint: {e}. Check if the correct model checkpoint is being loaded.")
        exit()

def main():
    configuration = Configuration(mode=Mode.independent_evaluation)
    set_logging(log_dir=configuration.logs_folder, suffix=LOG_IDENTIFIER_INDEPENDENT_EVALUATION)
    set_seed(seed=configuration.ind_eval_conf.seed)

    # Directly use caps_directories and info_data_files from configuration
    caps_directories = configuration.caps_directories
    info_data_files = configuration.info_data_files

    # Data reader setup
    data_paths = DataReader(
        caps_directories=caps_directories,
        info_data=info_data_files,
        diagnoses_info=configuration.diagnoses_info,
        quality_check=configuration.quality_check,
        valid_dataset_names=configuration.valid_dataset_names,
        info_data_cols=configuration.col_names
    )

    # Data loader setup
    data_loader = DataLoader(configuration=configuration, data=data_paths, mode=Mode.independent_evaluation)
    data_loader.batch_size = configuration.ind_eval_conf.batch_size
    data_loader.create_data_loader(shuffle=False)

    # Load the model
    num_classes = len(configuration.ind_eval_conf.eval_labels)
    model = load_model(configuration.ind_eval_conf.checkpoint_load, configuration.device, num_classes)

    # Evaluate with middle slice
    data_loader.eval_loader.dataset.middle_slice = True
    print("Running evaluation on middle slice...")
    model.test_(configuration=configuration, test_loader=data_loader.eval_loader)

    # Extended evaluation (multiple slices)
    data_loader.eval_loader.dataset.middle_slice = False
    print("Running extended evaluation (multiple slices)...")
    model.test_ext(configuration=configuration, test_loader=data_loader.eval_loader)

if __name__ == "__main__":
    main()
