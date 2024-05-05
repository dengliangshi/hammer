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

# User define module
from hammer.utils.config import Config

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class ModelFactory(object):

    def __init__(self, config: Config, logger: Logger):
        """initialize model.

        Args:
            config (Config): configuration for model.
            logger (Logger): instance of logger.
        """
        self.model_name = config.model_name
        self.config = config
        self.logger = logger
    
    def build_model(self):
        """build model.

        Returns:
            model (torch.nn.Module): model.
            optimizer (Optimizer): optimizer.
        """
        raise NotImplementedError
    
    def loss(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """calculate loss.

        Args:
            y_true (torch.Tensor): true labels.
            y_pred (torch.Tensor): predicted labels.

        Returns: 
            loss (torch.Tensor): loss.
        """
        raise NotImplementedError

    def create_optimizer(self, model: torch.nn.Module):
        """create optimizer.

        Args:
            model (torch.nn.Module): model.

        Returns:
            optimizer (Optimizer): optimizer.
        """
        raise NotImplementedError
    
    def create_lr_scheduler(self, optimizer: Optimizer):
        """create learning rate scheduler.

        Args:
            optimizer (Optimizer): optimizer.

        Returns:
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler.
        """
        raise NotImplementedError
    
    def metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        """calculate metrics.

        Args:
            y_true (torch.Tensor): true labels.
            y_pred (torch.Tensor): predicted labels.

        Returns:
            metrics (dict): metrics.
        """
        raise NotImplementedError
