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
import torch
import deepspeed
from torch import distributed

# User define module
from hammer.algorithms.uitls import mpu


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Trainer(object):

    def __init__(self, model: Modeling, train_dataset: Dataset, valid_dataset: Dataset, config: Config, logger: Logger):
        pass


    def _init_distributed(self, config):
        # 
        device = config.train.rank % torch.cuda.device_count()
        if config.local_rank is not None:
            device = config.local_rank
        torch.cuda.device(device)
        # 
        if config.train.use_deepspeed:
            deepspeed.init_distributed()
        else:
            distributed.init_process_group(
                
            )
        

