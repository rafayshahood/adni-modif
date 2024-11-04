import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, MetricCollection
import os
from configuration.configuration import Configuration
from data_processing.data_loader import DataLoader, Mode
from data_processing.data_reader import DataReader
from data_processing.utils import set_seed, set_logging, FREEZE_BACKBONE, SEED
from models.classifier import ClassificationModel
from models.nnclr import get_convnext

def execute(conf: Configuration, freeze_backbone: bool, backbone_ckpt: str):
    set_seed(conf.seeds[0])  # Set seed for reproducibility
    logging.info(f"{SEED}: {conf.seeds[0]}")
    logging.info(f"{FREEZE_BACKBONE}: {freeze_backbone}")

    # Load data:
    data_paths = DataReader(
        caps_directories=conf.caps_directories,
        info_data=conf.info_data_files,
        diagnoses_info=conf.diagnoses_info,
        quality_check=conf.quality_check,
        valid_dataset_names=conf.valid_dataset_names,
        info_data_cols=conf.col_names
    )
    data_loader = DataLoader(
        configuration=conf,
        data=data_paths,
        mode=Mode.classifier
    )
    data_loader.batch_size = conf.cls_conf.batch_size
    data_loader.create_data_loader()

    # Load and configure model:
    backbone = get_convnext()
    checkpoint = torch.load(backbone_ckpt, map_location=conf.device)
    if 'backbone' in checkpoint:
        backbone.load_state_dict(checkpoint['backbone'])

    cls = ClassificationModel(
        feature_extractor=backbone,
        num_classes=data_loader.classes,
        class_weights=data_loader.class_weights,
        freeze_backbone=freeze_backbone
    )
    cls.to(conf.device)

    optimizer = optim.Adam(cls.parameters(), lr=0.001)
    cls.train()
    metrics = MetricCollection({'accuracy': Accuracy(num_classes=cls.num_classes, average='macro', task='multiclass')})

    # Training loop:
    for epoch in range(conf.cls_conf.epochs):
        for batch in data_loader.train_loader:
            inputs_view1, _, labels = batch
            optimizer.zero_grad()
            outputs = cls(inputs_view1)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            metrics.update(outputs, labels)

    # Save the full model:
    torch.save({
        'feature_extractor': backbone.state_dict(),
        'classifier': cls.classifier.state_dict()
    }, f"{conf.checkpoints_folder}/classifier_seed-{conf.seeds[0]}_freeze-{freeze_backbone}_conf_id-{conf.id}.pt")

    accuracy = metrics['accuracy'].compute()
    logging.info(f"Training Accuracy: {accuracy}")

configuration = Configuration(mode=Mode.classifier)
set_logging(log_dir=configuration.logs_folder, suffix='classifier_training')

for seed in configuration.seeds:
    backbone_checkpoint = f"{configuration.checkpoints_folder}/nnclr_seed-{seed}_conf_id-{configuration.id}"
    if os.path.exists(backbone_checkpoint):
        execute(conf=configuration, freeze_backbone=True, backbone_ckpt=backbone_checkpoint)
        execute(conf=configuration, freeze_backbone=False, backbone_ckpt=backbone_checkpoint)
    else:
        print("File not found, check the path and filename.")
