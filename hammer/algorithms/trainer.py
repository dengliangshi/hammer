#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
from logging import Logger

# Third-party libraries
import torch
import deepspeed

# User define module
from hammer.algorithms.uitls import mpu


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Trainer(object):

    def __init__(self, model: Modeling, train_dataset: Dataset, valid_dataset: Dataset, config: Config, logger: Logger):
        pass
