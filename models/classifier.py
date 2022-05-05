import logging
from collections import OrderedDict
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils import data as torch_data
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, ConfusionMatrix, MatthewsCorrcoef

from configuration.configuration import Configuration

LOG_IDENTIFIER_CLASSIFIER = "classification_model"
LOG_IDENTIFIER_INDEPENDENT_EVALUATION = "independent_evaluation"


class LayerNorm2d(nn.LayerNorm):
    """
    Normalisation layer
    """

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ClassificationModel(torch.nn.Module):
    """
    Classification model that is used on top of the features provided by the backbone trained in a self-supervised way.
    """

    def __init__(self, feature_extractor: Sequential, num_classes: int, class_weights: ndarray = None,
                 num_ftrs: int = 768, freeze_backbone: bool = True, scheduler_iterations=10):
        """
        Initialises with the provided parameters.
        :param feature_extractor: the backbone
        :param num_classes: number of classes
        :param class_weights: class weights
        :param num_ftrs: number of features
        :param scheduler_iterations: the maximum number of iterations for a scheduler
        :param freeze_backbone: if True then a backbone will be frozen, otherwise not
        """
        super(ClassificationModel, self).__init__()
        self.freeze_backbone = freeze_backbone
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        norm_layer = partial(LayerNorm2d, eps=1e-6)

        self.classifier = nn.Sequential(OrderedDict([
            ('norm_layer', norm_layer(num_ftrs)),
            ('flatten_layer', nn.Flatten(1)),
            ('top_layer', nn.Linear(num_ftrs, num_classes)),
        ]))

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and name == 'classifier.top_layer':
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if class_weights is not None:
            self.ce = torch.nn.CrossEntropyLoss(torch.tensor(class_weights, dtype=torch.float))
        else:
            self.ce = torch.nn.CrossEntropyLoss()

        self.optimizer = Adam([{"params": self.classifier.parameters(), "lr": 0.001, "weight_decay": 0.0001}])
        self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=scheduler_iterations, eta_min=0.0000001,
                                           verbose=True)
        self.name = ""

    def set_name(self, seed, freeze_backbone, conf_id):
        """
        Name of the component.
        :param seed: seed
        :param freeze_backbone: if True then a backbone will be frozen, otherwise not
        :param conf_id: configuration ID
        """
        self.name = self.get_name_as_string(seed, freeze_backbone, conf_id)

    @staticmethod
    def get_name_as_string(seed, freeze_backbone, conf_id):
        """
        Returns name of the component.
        :param seed: seed
        :param freeze_backbone:  if True then a backbone will be frozen, otherwise not
        :param conf_id: configuration ID
        :return: name of the component
        """
        return "cls_seed-{}_freeze-{}_conf_id-{}".format(seed, freeze_backbone, conf_id)

    def forward(self, x, only_features=False):
        """
        Performs a forward propagation.
        :param x: input data
        :param only_features: If True then only features are returned, otherwise predictions for each class (diagnosis)
        :return: torch.Tensor (features or predictions)
        """
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.feature_extractor(x)
        else:
            out = self.feature_extractor(x)
        if only_features:
            out = out.flatten(start_dim=1)
        else:
            out = self.classifier(out)
        return out

    def load(self, file_path: str, device: str = "cpu") -> None:
        """
        Loads a saved model.
        :param file_path: a path to a file
        :param device: device
        """
        checkpoint = torch.load(file_path, map_location=device)
        self.feature_extractor.load_state_dict(checkpoint["feature_extractor"])
        self.classifier.load_state_dict(checkpoint["classifier"])
        logging.info("Checkpoint: {} is loaded".format(str(file_path)))

    def save(self, file_path: str) -> None:
        """
        Saves a model.
        :param file_path: a path to a folder
        """
        feature_extractor_dict = self.feature_extractor.state_dict()
        classifier_dict = self.classifier.state_dict()
        out = "{}{}".format(file_path, self.name)
        torch.save({"feature_extractor": feature_extractor_dict,
                    "classifier": classifier_dict},
                   out)
        logging.info("Checkpoint: {} is saved".format(str(file_path)))

    def train_(self, configuration: Configuration, train_loader: torch_data.DataLoader) -> None:
        """
        Trains a classifier.
        :param configuration: Configuration
        :param train_loader: torch.utils.data.DataLoader
        """
        logging.info("Training of the classification model ...")
        model_children = list(self.feature_extractor.children())
        if self.freeze_backbone:
            for idx, child in enumerate(model_children):
                for param in child.parameters():
                    param.requires_grad = False
        if self.freeze_backbone:
            self.feature_extractor.eval()
        else:
            self.feature_extractor.train()
        self.classifier.train()
        logging.info("# trainable parameters in backbone: {}".format(
            sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)))
        logging.info("# trainable parameters in classifier: {}".format(
            sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)))
        logging.info("# trainable parameters: {}".format(sum(p.numel() for p in self.parameters() if p.requires_grad)))

        for epoch in range(1, configuration.cls_conf.epochs + 1):
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
            if epoch == int(configuration.cls_conf.epochs * 0.5) or \
                    epoch == int(configuration.cls_conf.epochs * 0.75):
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
                    self.save(configuration.cls_conf.checkpoint_folder_save)
                    return
            if epoch % 10 == 0:
                self.save(configuration.cls_conf.checkpoint_folder_save)
            avg_loss = total_loss / len(train_loader)
            logging.info(f"epoch: |{epoch:>02}|, loss: |{avg_loss:.5f}|")
            logging.info("Train metrics: {}".format(metrics_torch.compute()))
            self.scheduler.step()

    def test_(self, configuration: Configuration, test_loader: torch_data.DataLoader):
        """
        Tests the the classifier.
        :param configuration: Configuration
        :param test_loader: torch.utils.data.DataLoader
        """
        logging.info("Test ...")

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

            for idx, (view_one, _, target) in enumerate(test_loader):
                # forward pass
                view_one = view_one.to(configuration.device)

                output = self.forward(view_one).squeeze(1)
                target = target.to(configuration.device)
                _, predicted = torch.max(output, 1)

                metrics_torch(predicted, target.int())

        logging.info("Test metrics: {}".format(metrics_torch.compute()))

    def test_ext(self, configuration, test_loader):
        """
        Tests the classification model by performing multiple runs with random slices for each sample
        :param configuration: Configuration
        :param test_loader: torch.utils.data.DataLoader
        """
        logging.info("Extended test ...")
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
        for replica in range(configuration.cls_conf.replicas):
            with torch.no_grad():
                last_idx = 0
                for idx, (view_one, _, target) in enumerate(test_loader):
                    # forward pass
                    view_one = view_one.to(configuration.device)

                    output = self.forward(view_one).squeeze(1)
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
        """
        Extracts features.
        :param configuration: Configuration object
        :param data_loader: data loader
        :param file_name: output file name
        """
        out = ''.join([configuration.features_folder, file_name])
        logging.info("Feature extraction -> {}".format(out))

        self.feature_extractor.eval()
        self.classifier.eval()
        data_list = []
        for replica in range(configuration.cls_conf.replicas_extraction):
            with torch.no_grad():
                for idx, (view_one, view_two, target) in enumerate(data_loader):
                    view_one = view_one.to(configuration.device)
                    view_two = view_two.to(configuration.device)

                    output = self.forward(view_one if random() > 0.5 else view_two, only_features=True).squeeze(1)
                    target = target.to(configuration.device)

                    data_list.append(torch.cat((output, target.unsqueeze(1)), 1))

        data = torch.cat(data_list).detach().cpu().numpy()
        np.save(out, data)
