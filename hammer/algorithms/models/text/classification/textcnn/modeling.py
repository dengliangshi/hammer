#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
from logging import Logger

# Third-party libraries
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.functional.classification import accuracy

# User define module
from hammer.utils.config import Config
from hammer.algorithms.models.backbones.text import TextCNN
from hammer.algorithms.models.model_factory import ModelFactory

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class TextCNNForCalssification(torch.nn.Module):

    def __init__(self, config: Config, logger: Logger):
        """initialize model.

        Args:
            config (Config): configuration for model.
            logger (Logger): instance of logger.
        """
        super(TextCNNForCalssification, self).__init__()

        self.backbone = TextCNN(config, logger)
        self.fc = torch.nn.Linear(sum(config.model.num_filters), config.dataset.num_classes)

    def forward(self, inputs: dict) -> dict:
        """forward function of model.

        Args:
            inputs (dict): inputs of model.

        Returns:
            dict: outputs of model.
        """
        outputs = {}

        hiddens = self.backbone(inputs)
        outputs['logits'] = self.fc(hiddens['pooled_outputs'])

        return outputs


class TextCNNFactory(ModelFactory):

    def __init__(self, model_name: str, config: Config, logger: Logger):
        """Initialize model factory.

        Args:
            config (Config): configuration for model.
            logger (Logger): instance of logger.
        """
        super(TextCNNFactory, self).__init__(model_name, config, logger)

    def build_model(self) -> torch.nn.Module:
        """Build model.

        Returns:
            torch.nn.Module: the instance of model.
        """
        return TextCNNForCalssification(self.config, self.logger)

    def create_optimizer(self, learning_rate: float) -> Optimizer:
        """Create optimizer for training model.

        Args:
            learning_rate (float): learning rate for trianing model.

        Returns:
            Optimizer: optimizer for training model.
        """
        return torch.optim.Adam(self.parameters(), lr = learning_rate)
    
    def create_lr_scheduler(self, learning_rate) -> LRScheduler:
        """Create learning rate scheduler for training model.

        Args:
            learning_rate (float): learning rate for trianing model.

        Returns:
            LRScheduler: learning rate scheduler for training model.
        """
        return learning_rate

    def cal_loss(self, outputs: dict, targets: dict) -> float:
        """loss function of model.

        Args:
            outputs (dict): outputs of model.
            targets (dict): targets of given batch of samples.

        Returns:
            float: loss of given batch of samples.
        """
        return torch.nn.functional.cross_entropy(outputs['logits'], targets['class'])

    def metrics(self, outputs: dict, targets: dict) -> dict:
        """metrics of model.

        Args:
            outputs (dict): outputs of model.
            targets (dict): targets of given batch of samples.

        Returns:
            dict: metrics of given batch of samples.
        """
        predictions = torch.argmax(outputs['logits'], dim=1)

        acc = accuracy(
            preds = predictions, 
            target = targets['class'],
            task = 'multiclass',
            num_classes = self.config.dataset.num_classes
        )

        return {'acc': acc.item()}
