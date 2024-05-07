#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library


# Third-party libraries


# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
def CharTokenizer(object):

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def __call__(self, sample: dict) -> dict:
        """_summary_

        Args:
            sample (dict): _description_

        Returns:
            dict: _description_
        """
        sample['tokens'] = list(sample['text'])
        return sample
