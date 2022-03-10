import logging
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from torch.utils import data as torch_data
from torch.utils.data import Subset

from configuration.configuration import Configuration
from data_processing.data_provider import DataProviderSSL
from data_processing.data_reader import DataReader
from data_processing.utils import Mode


class DataLoaderSSL:
    """
    Creates the train and test/evaluation loaders
    """

    def __init__(self, configuration: Configuration, data: DataReader) -> None:
        """
        Initialize with all required attributes.
        :type configuration: Configuration
        :type data: DataReader
        """
        self.configuration = configuration
        self.data = data

        # Values will be filled during execution:
        self.batch_size = None
        self.mode = None
        self.classes = None
        self.train_loader, self.eval_loader = None, None
        self.class_weights = None

    def filter_data(self, dataset: Subset, diagnoses: list) -> Subset:
        """
        Selects only data that are relevant for evaluation
        :param dataset: A dataset (Subset) with indices
        :param diagnoses: A list of diagnoses
        :return: A dataset (Subset) with indices that correspond to relevant diagnoses
        """

        indices = [index for index, element in enumerate(dataset.dataset.diagnoses) if
                   element in self.configuration.le_conf.eval_labels]

        # Select indices that refer only to evaluation labels:
        intersect = [element for element in dataset.indices if element in indices]
        dataset = torch_data.Subset(dataset.dataset, intersect)

        # Targets should start with 0 and the last one should correspond to the number of classes:
        dataset.dataset.targets[:] = [None for i in range(len(dataset.dataset.targets))]
        for pos, label in enumerate(self.configuration.le_conf.eval_labels):
            logging.info("{} is encoded as {}".format(label, pos))
            label_indices = [idx for idx, element in enumerate(diagnoses) if element == label]
            label_replacements = [pos for i in range(len(label_indices))]
            for (index, replacement) in zip(label_indices, label_replacements):
                dataset.dataset.targets[index] = replacement

        return dataset

    def split_data(self, targets, diagnoses) -> Tuple[Subset, Subset]:
        """
        Split data into training and evaluation sets
        :param targets: Targets/Labels/Diagnoses as int values
        :param diagnoses: Diagnoses as str values
        :return: training and evaluation sets
        """

        # Assure that the split is always the same
        self.configuration.set_seeds(145794547)

        patients = self.data.data['patient'].tolist()  # A list of patient IDs

        # Get indices of training and evaluation sets:
        train_pt_indices, eval_pt_indices = train_test_split(list(range(len(patients))), test_size=0.5,
                                                             stratify=targets)

        # Samples from multiple sessions of one patient should appear only in one set: either training or evaluation:
        train_pt = []
        eval_pt = []
        train_idx = []
        eval_idx = []
        for idx in range(0, len(patients)):
            if idx in train_pt_indices:
                train_pt.append(patients[idx])
                train_idx.append(idx)
            elif idx in eval_pt_indices:
                eval_pt.append(patients[idx])
                eval_idx.append(idx)
            else:
                raise ValueError

        eval_idx_copy = eval_idx.copy()
        for idx, pt in enumerate(eval_pt):
            if pt in train_pt:
                eval_idx.remove(eval_idx_copy[idx])
                train_idx.append(eval_idx_copy[idx])
        assert len(set(train_idx).intersection(set(eval_idx))) == 0

        logging.info("Number of samples in training set: {}".format(len(train_idx)))
        values = [diagnoses[i] for i in train_idx]
        logging.info("Counts: {}".format(dict(zip(list(values), [list(values).count(i) for i in list(values)]))))

        if self.mode == Mode.evaluation:
            logging.info("Number of samples in evaluation set: {}".format(len(eval_idx)))
            values = [diagnoses[i] for i in eval_idx]
            logging.info("Counts: {}".format(dict(zip(list(values), [list(values).count(i) for i in list(values)]))))

        # Create a dataset for accessing samples:
        dataset = DataProviderSSL(self.data.data['file'].tolist(), targets, diagnoses,
                                  self.configuration.slices_range, self.configuration.slices_per_view, self.mode)

        train_dataset = torch_data.Subset(dataset, train_idx)
        eval_dataset = torch_data.Subset(dataset, eval_idx)

        # Restore original seeds:
        self.configuration.set_seeds()

        return train_dataset, eval_dataset

    def create_data_loader(self) -> None:
        """
        Create data loader.
        """

        train_dataset, eval_dataset = self.split_data(self.data.data['target'].tolist(),
                                                      self.data.data['diagnosis'].tolist())

        if self.mode == Mode.evaluation:
            # During evaluation another set of targets can be used:
            train_dataset = self.filter_data(train_dataset, self.data.data['diagnosis'].tolist())
            eval_dataset = self.filter_data(eval_dataset, self.data.data['diagnosis'].tolist())

            # Data may be unbalanced. Calculate weights for the loss function:
            y = np.asarray([train_dataset.dataset.targets[i] for i in train_dataset.indices])
            self.class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y),
                                                                   y=y)

            # Get the number of classes:
            self.classes = len(set([train_dataset.dataset.targets[i] for i in train_dataset.indices]))
            logging.info("# classes: {}".format(self.classes))

        # Finally create data loaders that will be used during training/evaluation/feature extraction:
        self.train_loader = torch_data.DataLoader(train_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=True, num_workers=4)
        self.eval_loader = torch_data.DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
