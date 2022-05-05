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
        Initialises with all required attributes.
        :type configuration: Configuration object
        :type data: DataReader object
        :type mode: Mode object
        """
        self.configuration = configuration
        self.data = data
        self.mode = mode

        # Values will be filled during the execution:
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

        if Mode.independent_evaluation == self.mode:
            eval_labels = self.configuration.ind_eval_conf.eval_labels
        else:
            eval_labels = self.configuration.cls_conf.eval_labels

        indices = [index for index, element in enumerate(dataset.dataset.diagnoses) if
                   element in eval_labels]

        # Select indices that refer only to evaluation labels:
        intersect = [element for element in dataset.indices if element in indices]
        dataset = torch_data.Subset(dataset.dataset, intersect)

        # Targets should start with 0 and the last one should correspond to the number of classes:
        dataset.dataset.targets[:] = [None for i in range(len(dataset.dataset.targets))]
        for pos, label in enumerate(eval_labels):
            logging.info("{} is encoded as {}".format(label, pos))
            label_indices = [idx for idx, element in enumerate(diagnoses) if element == label]
            label_replacements = [pos for i in range(len(label_indices))]
            for (index, replacement) in zip(label_indices, label_replacements):
                dataset.dataset.targets[index] = replacement

        return dataset

    def split_data(self, targets: list, diagnoses: list, test_size: float = 0.4) -> Tuple[Subset, Subset]:
        """
        Splits data into train and evaluation sets
        :param targets: Targets/Labels/Diagnoses as int values
        :param diagnoses: Diagnoses as str values
        :param test_size: the proportion of samples that should be in a test set
        :return: training and evaluation sets
        """

        patients = self.data.data['patient'].tolist()  # A list of patient IDs

        # Get indices of training and evaluation sets:
        if test_size == 0.0:
            train_pt_indices = list(range(len(patients)))
            eval_pt_indices = []
        else:
            train_pt_indices, eval_pt_indices = train_test_split(list(range(len(patients))), test_size=test_size,
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

        logging.info("Number of samples: {}".format(len(train_idx)))
        values = [diagnoses[i] for i in train_idx]
        logging.info("Counts: {}".format(dict(zip(list(values), [list(values).count(i) for i in list(values)]))))

        if self.mode == Mode.classifier:
            logging.info("Number of samples in evaluation set: {}".format(len(eval_idx)))
            values = [diagnoses[i] for i in eval_idx]
            logging.info("Counts: {}".format(dict(zip(list(values), [list(values).count(i) for i in list(values)]))))

        # Create a dataset for accessing samples:
        dataset = DataProvider(self.data.data['file'].tolist(), targets, diagnoses,
                               self.configuration.slices_range, self.mode)

        train_dataset = torch_data.Subset(dataset, train_idx)
        eval_dataset = torch_data.Subset(dataset, eval_idx)
        logging.info("IDs: {}".format(eval_idx))
        return train_dataset, eval_dataset

    def create_data_loader(self, shuffle: bool = True) -> None:
        """
        Creates data loader.
        :type shuffle: if True, then eval loader will shuffle data samples before sampling, otherwise not
        """

        train_dataset, eval_dataset = self.split_data(self.data.data['target'].tolist(),
                                                      self.data.data['diagnosis'].tolist(),
                                                      test_size=0.0 if self.mode == Mode.independent_evaluation else 0.4)

        if self.mode == Mode.independent_evaluation:
            dataset = self.filter_data(train_dataset, self.data.data['diagnosis'].tolist())
            self.eval_loader = torch_data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                                                     num_workers=8)
            self.train_loader = None
            return

        if self.mode == Mode.classifier:
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
                                                  shuffle=True, num_workers=8)
        self.eval_loader = torch_data.DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=shuffle,
                                                 num_workers=8)
