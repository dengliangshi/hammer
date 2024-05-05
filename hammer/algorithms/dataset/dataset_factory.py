#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
from logging import Logger

# Third-party libraries
from torch.utils.data import Dataset

# User define module
from hammer.tools.flow import Flow
from hammer.utils.config import Config

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class DatasetFactory(object):
    
    def __init__(self, dataset_cls: Dataset, flow: Flow, config: Config, logger: Logger):
        """Create dataset .

        Args:
            dataset_cls (Dataset): _description_
            config (Config): _description_
            logger (Logger): _description_
        """
        self.config = config
        self.logger = logger
        # create train dataset
        if config.train.enabled:
            self.train_dataset = dataset_cls(
                file_path=os.path.join(config.dataset.data_dir, 'train.json'),
                flow=flow,
                config=config
            )
        else:
            self.train_dataset = None
        # create valid dataset
        if config.valid.enabled:
            self.valid_dataset = dataset_cls(
                file_path=os.path.join(config.dataset.data_dir, 'valid.json'),
                flow=flow,
                config=config
            )
        else:
            self.valid_dataset = None
        # create test dataset
        if config.test.enabled:
            self.test_dataset = dataset_cls(
                file_path=os.path.join(config.dataset.data_dir, 'test.json'),
                flow=flow,
                config=config
            )
        else:
            self.test_dataset = None

    def train(self):
        return self.train_dataset

    def valid(self):
        return self.valid_dataset

    def test(self):
        return self.test_dataset
