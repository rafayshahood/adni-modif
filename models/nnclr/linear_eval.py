import logging
from functools import partial
from random import random

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from torch import nn, Tensor
from torch.nn import Sequential
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils import data as torch_data
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix, MatthewsCorrcoef

from configuration.configuration import Configuration


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class LinearEval(torch.nn.Module):
    """
    Perform a linear evaluation of the NNCLR model
    """

    def __init__(self, feature_extractor: Sequential, num_classes: int, class_weights: ndarray = None,
                 num_ftrs: int = 1536):
        """
        Initialize with the provided attributes
        :param feature_extractor: a NNCLR backbone
        :param num_classes: the number of classes
        """
        super(LinearEval, self).__init__()
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        self.classifier = nn.Sequential(
            norm_layer(768), nn.Flatten(1), nn.Linear(768, num_classes)
        )

        for m in self.classifier():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if class_weights is not None:
            self.ce = torch.nn.CrossEntropyLoss(torch.tensor(class_weights, dtype=torch.float))
        else:
            self.ce = torch.nn.CrossEntropyLoss()

        self.optimizer = Adam([{"params": self.classifier.parameters(), "lr": 0.001}])

    def forward(self, x, only_features=False, lrp_run=False):
        """
        Perform a forward propagation
        :param x: input data
        :param only_features: If True then only features are returned, otherwise predictions for each class (diagnosis)
        :param lrp_run: If True then gradient is required, otherwise not
        :return: torch.Tensor (features or predictions)
        """
        if not lrp_run:
            with torch.no_grad():
                out = self.feature_extractor(x)
        else:
            out = self.feature_extractor(x)
        if only_features:
            out = out.flatten(start_dim=1)
        else:
            out = self.classifier(out.flatten(start_dim=1))
        return out

    def load(self, file_path: str, device: str = "cpu") -> None:
        """
        Load a saved model
        :param file_path: a path to a file
        :param device: device
        """
        checkpoint = torch.load(file_path, map_location=device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        logging.info("Checkpoint: {} is loaded".format(str(file_path)))

    def save(self, file_path: str) -> None:
        """
        Save a model
        :param file_path: a path to a file
        """
        feature_extractor_dict = self.feature_extractor.state_dict()
        classifier_dict = self.classifier.state_dict()
        torch.save({"feature_extractor": feature_extractor_dict,
                    "classifier": classifier_dict},
                   file_path)
        logging.info("Checkpoint: {} is saved".format(str(file_path)))

    def train_(self, configuration: Configuration, train_loader: torch_data.DataLoader) -> None:
        """
        Train linear evaluation
        :param configuration: Configuration
        :param train_loader: torch.utils.data.DataLoader
        """
        model_children = list(self.feature_extractor.children())
        for idx, child in enumerate(model_children):
            for param in child.parameters():
                param.requires_grad = False
        self.feature_extractor.eval()
        self.classifier.train()
        logging.info("# trainable parameters in backbone: {}".format(
            sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)))
        logging.info("# trainable parameters in classifier: {}".format(
            sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)))
        logging.info("# trainable parameters: {}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

        for epoch in range(1, configuration.le_conf.epochs + 1):
            total_loss = 0
            metrics_torch = MetricCollection({'acc': Accuracy(compute_on_step=False, num_classes=self.num_classes),
                                              'precision': Precision(compute_on_step=False, average='macro',
                                                                     num_classes=self.num_classes),
                                              'recall': Recall(compute_on_step=False, average='macro',
                                                               num_classes=self.num_classes),
                                              'macro-f1': F1(compute_on_step=False, average='macro',
                                                             num_classes=self.num_classes),
                                              'cm': ConfusionMatrix(num_classes=self.num_classes)})
            metrics_torch.to(configuration.device)
            if epoch == int(configuration.le_conf.epochs * 0.5) or \
                    epoch == int(configuration.le_conf.epochs * 0.75):
                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.1  # divide by 10
            for idx, (view_one, view_two, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                # forward pass
                view_one = view_one.to(configuration.device)
                view_two = view_two.to(configuration.device)

                output = self.forward(view_one if random() > 0.5 else view_two).squeeze(1)

                target = target.to(configuration.device)
                loss = self.ce(output, target)
                total_loss += loss.detach()
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(output, 1)
                metrics_torch(predicted, target.int())
                if configuration.dry_run:
                    self.save(configuration.le_conf.checkpoint_save)
                    return
            if epoch % 10 == 0:
                self.save(configuration.le_conf.checkpoint_save)
            avg_loss = total_loss / len(train_loader)
            logging.info(f"epoch: |{epoch:>02}|, loss: |{avg_loss:.5f}|")
            logging.info("Train metrics: {}".format(metrics_torch.compute()))

    def test_(self, configuration: Configuration, test_loader: torch_data.DataLoader):
        """
        Test the NNCLR model
        :param configuration: Configuration
        :param test_loader: torch.utils.data.DataLoader
        """
        self.feature_extractor.eval()
        self.classifier.eval()

        with torch.no_grad():
            metrics_torch = MetricCollection({'acc': Accuracy(compute_on_step=False, num_classes=self.num_classes),
                                              'precision': Precision(compute_on_step=False, average='macro',
                                                                     num_classes=self.num_classes),
                                              'recall': Recall(compute_on_step=False, average='macro',
                                                               num_classes=self.num_classes),
                                              'macro-f1': F1(compute_on_step=False, average='macro',
                                                             num_classes=self.num_classes),
                                              'mcc': MatthewsCorrcoef(compute_on_step=False,
                                                                      num_classes=self.num_classes),
                                              'cm': ConfusionMatrix(num_classes=self.num_classes)})
            metrics_torch.to(configuration.device)

            for idx, (view_one, view_two, target) in enumerate(test_loader):
                # forward pass
                view_one = view_one.to(configuration.device)
                view_two = view_two.to(configuration.device)

                output = self.forward(view_one if random() > 0.5 else view_two).squeeze(1)
                target = target.to(configuration.device)
                _, predicted = torch.max(output, 1)

                metrics_torch(predicted, target.int())

        logging.info("Test metrics: {}".format(metrics_torch.compute()))

    def test_ext(self, configuration, test_loader):
        """
        Test the NNCLR model by performing multiple runs with random slices for each sample
        :param configuration: Configuration
        :param test_loader: torch.utils.data.DataLoader
        """
        self.feature_extractor.eval()
        self.classifier.eval()
        metrics_torch = MetricCollection({'acc': Accuracy(compute_on_step=False, num_classes=self.num_classes),
                                          'precision': Precision(compute_on_step=False, average='macro',
                                                                 num_classes=self.num_classes),
                                          'recall': Recall(compute_on_step=False, average='macro',
                                                           num_classes=self.num_classes),
                                          'macro-f1': F1(compute_on_step=False, average='macro',
                                                         num_classes=self.num_classes),
                                          'mcc': MatthewsCorrcoef(compute_on_step=False, num_classes=self.num_classes),
                                          'cm': ConfusionMatrix(num_classes=self.num_classes)})
        metrics_torch.to('cpu')
        data_list = []
        for replica in range(configuration.le_conf.replicas):
            with torch.no_grad():
                last_idx = 0
                for idx, (view_one, view_two, target) in enumerate(test_loader):
                    # forward pass
                    view_one = view_one.to(configuration.device)
                    view_two = view_two.to(configuration.device)

                    output = self.forward(view_one if random() > 0.5 else view_two).squeeze(1)
                    target = target.to(configuration.device)

                    _, predicted = torch.max(output, 1)

                    data_list.append(
                        torch.stack(
                            (predicted,
                             target,
                             torch.as_tensor(np.array(list(range(last_idx, last_idx + view_one.shape[0]))),
                                             device=configuration.device)
                             ), 1))
                    last_idx += view_one.shape[0]

        data = torch.cat(data_list).detach().cpu().numpy()
        data_df = pd.DataFrame(data=data)
        data_replicated = data_df.groupby(data_df.columns[-1]).agg(lambda x: x.value_counts().index[0]).to_numpy()
        predictions = torch.as_tensor(data_replicated[:, 0])
        targets = torch.as_tensor(data_replicated[:, -1])
        metrics_torch(predictions, targets.int())

        logging.info("Test metrics: {}".format(metrics_torch.compute()))

    def extract_features(self, configuration: Configuration, data_loader: torch_data.DataLoader,
                         file_name: str) -> None:
        self.feature_extractor.eval()
        self.classifier.eval()
        data_list = []
        for replica in range(configuration.le_conf.replicas_extraction):
            with torch.no_grad():
                for idx, (view_one, view_two, target) in enumerate(data_loader):
                    view_one = view_one.to(configuration.device)
                    view_two = view_two.to(configuration.device)

                    output = self.forward(view_one if random() > 0.5 else view_two, only_features=True).squeeze(1)
                    target = target.to(configuration.device)

                    data_list.append(torch.cat((output, target.unsqueeze(1)), 1))

        data = torch.cat(data_list).detach().cpu().numpy()
        np.save('_'.join([configuration.features_out, file_name]), data)
