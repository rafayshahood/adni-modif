import logging
from typing import Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch.utils import data as torch_data
from torch.utils.data import Subset
from configuration.configuration import Configuration
from data_processing.data_provider import DataProvider
from data_processing.data_reader import DataReader
from data_processing.utils import Mode


class DataLoader:
    """
    Creates train and test/evaluation loaders
    """

    def __init__(self, configuration: Configuration, data: DataReader, mode: Mode) -> None:
        """
        Initializes with all required attributes.
        :param configuration: Configuration object
        :param data: DataReader object
        :param mode: Mode object
        """
        self.configuration = configuration
        self.data = data
        self.mode = mode

        # Values will be filled during execution
        self.batch_size = None
        self.classes = None
        self.train_loader, self.eval_loader = None, None
        self.class_weights = None

    def filter_data(self, dataset: Subset, diagnoses: list) -> Subset:
        """
        Selects only data that are relevant for evaluation/test
        :param dataset: A dataset (Subset) with indices
        :param diagnoses: A list of diagnoses
        :return: A dataset (Subset) with indices that correspond to relevant diagnoses
        """
        if self.mode == Mode.independent_evaluation:
            eval_labels = self.configuration.ind_eval_conf.eval_labels
        else:
            eval_labels = self.configuration.cls_conf.eval_labels

        indices = [index for index, element in enumerate(dataset.dataset.diagnoses) if element in eval_labels]

        # Select indices that refer only to evaluation labels:
        intersect = [element for element in dataset.indices if element in indices]
        dataset = torch_data.Subset(dataset.dataset, intersect)

        # Adjust targets to start from 0 up to the number of classes
        dataset.dataset.targets[:] = [None for _ in range(len(dataset.dataset.targets))]
        for pos, label in enumerate(eval_labels):
            logging.info("{} is encoded as {}".format(label, pos))
            label_indices = [idx for idx, element in enumerate(diagnoses) if element == label]
            for index in label_indices:
                dataset.dataset.targets[index] = pos

        return dataset

    def split_data(self, targets: list, diagnoses: list, test_size: float = 0.4) -> Tuple[Subset, Subset]:
        """
        Splits data into train and evaluation sets
        :param targets: Targets/Labels/Diagnoses as int values
        :param diagnoses: Diagnoses as str values
        :param test_size: Proportion of samples that should be in the test set
        :return: training and evaluation sets
        """
        patients = self.data.data['patient'].tolist()

        # Split indices for train and eval sets
        if test_size == 0.0:
            train_pt_indices = list(range(len(patients)))
            eval_pt_indices = []
        else:
            train_pt_indices, eval_pt_indices = train_test_split(
                list(range(len(patients))), test_size=test_size, stratify=targets
            )

        # Ensure one set per patient across train and eval
        train_idx, eval_idx = [], []
        for idx in range(len(patients)):
            if idx in train_pt_indices:
                train_idx.append(idx)
            elif idx in eval_pt_indices:
                eval_idx.append(idx)
        
        eval_idx_copy = eval_idx.copy()
        for idx, pt in enumerate([patients[i] for i in eval_idx_copy]):
            if pt in [patients[i] for i in train_idx]:
                eval_idx.remove(eval_idx_copy[idx])
                train_idx.append(eval_idx_copy[idx])
        assert len(set(train_idx).intersection(set(eval_idx))) == 0

        logging.info("Number of training samples: {}".format(len(train_idx)))
        logging.info("Training diagnoses distribution: {}".format(
            dict(zip(*np.unique([diagnoses[i] for i in train_idx], return_counts=True)))
        ))

        if self.mode == Mode.classifier:
            logging.info("Number of evaluation samples: {}".format(len(eval_idx)))
            logging.info("Evaluation diagnoses distribution: {}".format(
                dict(zip(*np.unique([diagnoses[i] for i in eval_idx], return_counts=True)))
            ))

        # Create a dataset for accessing samples
        dataset = DataProvider(self.data.data['file'].tolist(), targets, diagnoses,
                               self.configuration.slices_range, self.mode)

        train_dataset = torch_data.Subset(dataset, train_idx)
        eval_dataset = torch_data.Subset(dataset, eval_idx)
        return train_dataset, eval_dataset

    def create_data_loader(self, shuffle: bool = True) -> None:
        """
        Creates data loader.
        :param shuffle: if True, eval loader will shuffle data samples before sampling, otherwise not
        """
        # Split data into training and evaluation datasets
        train_dataset, eval_dataset = self.split_data(
            self.data.data['target'].tolist(),
            self.data.data['diagnosis'].tolist(),
            test_size=0.0 if self.mode == Mode.independent_evaluation else 0.4
        )

        # Check if the datasets are empty
        if len(train_dataset) == 0:
            logging.warning("Training dataset is empty. Check data split configuration.")
        if len(eval_dataset) == 0:
            logging.warning("Evaluation dataset is empty. Check data split configuration.")

        # Handle independent evaluation mode
        if self.mode == Mode.independent_evaluation:
            dataset = self.filter_data(train_dataset, self.data.data['diagnosis'].tolist())
            if len(dataset) > 0:
                self.eval_loader = torch_data.DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0
                )
            else:
                logging.warning("Filtered evaluation dataset is empty. Adjust test_size or filter criteria.")
            self.train_loader = None
            return

        # Handle classifier mode
        if self.mode == Mode.classifier:
            train_dataset = self.filter_data(train_dataset, self.data.data['diagnosis'].tolist())
            eval_dataset = self.filter_data(eval_dataset, self.data.data['diagnosis'].tolist())

            # Compute class weights for unbalanced data
            y = np.asarray([train_dataset.dataset.targets[i] for i in train_dataset.indices])
            self.class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)

            # Number of classes
            self.classes = len(set([train_dataset.dataset.targets[i] for i in train_dataset.indices]))
            logging.info("Number of classes: {}".format(self.classes))
            logging.info("Class weights: {}".format(self.class_weights))

        # Create DataLoaders for training and evaluation if datasets are not empty
        if len(train_dataset) > 0:
            self.train_loader = torch_data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )
        else:
            logging.warning("Training dataset is empty after filtering.")

        if len(eval_dataset) > 0:
            self.eval_loader = torch_data.DataLoader(
                eval_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0
            )
        else:
            logging.warning("Evaluation dataset is empty after filtering.")
