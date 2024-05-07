#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2023
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library


# Third-party libraries


# User define module
from hammer.utils.attr_dict import AttrDict
from hammer.utils.config import Config as BaseConfig

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Config(BaseConfig):

    def __init__(self, config_file: str):
        """initialize the configuration using user-defined parameters.

        Args:
            config_file (str): path of configuration file.
        """
        self.model_name = 'TextCNN'
        
        self.dataset = AttrDict()
        self.dataset.data_path = '/home/dengliang/benchmarks/tnews'
        self.dataset.vocab_size = 2
        self.dataset.num_classes = 2

        self.model = AttrDict()
        # if word embeddings are trainable
        self.model.fixed_embeddings = False
        # embedding dimension
        self.model.embedding_dim = 128
        # number of filters
        self.model.num_filters = [32, 64, 128, 256]
        # size of filters
        self.model.filter_sizes = [2, 3, 4, 5]
        # keep probabilty of dropout
        self.model.dropout = 0.5

        self.train = AttrDict()
        # learning rate for training
        self.train.learning_rate = 1e-3


        self.valid = AttrDict()
        self.valid.enabled = True

        self.test = AttrDict()
        self.test.enabled = True

        super(Config, self).__init__(config_file)
