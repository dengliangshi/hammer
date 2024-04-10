#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
from logging import Logger

# Third-party libraries
from torch.utils.data import Dataset

# User define module
from hammer.utils.config import Config

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Dataset(object):
    
    def __init__(self, dataset_cls: Dataset, config: Config, logger: Logger):
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
                data_dir=config.dataset.data_dir,
                split='train',
                config=config
            )
        else:
            self.train_dataset = None
        # create valid dataset
        if config.valid.enabled:
            self.valid_dataset = dataset_cls(
                data_dir=config.dataset.data_dir,
                split='valid',
                config=config
            )
        else:
            self.valid_dataset = None
        # create test dataset
        if config.test.enabled:
            self.test_dataset = dataset_cls(
                data_dir=config.dataset.data_dir,
                split='test',
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
