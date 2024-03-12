#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import json

# Third-party libraries


# User define module
from hammer.utils.attr_dict import AttrDict

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Config(AttrDict):

    def __init__(self, config_file: str):
        """initialize the configuration using user-defined parameters.

        Args:
            config_file (str): path of configuration file.
        """
        params = self._load_config(config_file)

        for key, value in params.items():
            if isinstance(value, dict):
                value = AttrDict(value)
            setattr(self, key, value)

    def _load_config(self, config_file: str) -> dict:
        """load in user-defined parameters from given configuration file.

        Args:
            config_file (str): path of configuration file.

        Returns:
            dict: user-defined parameters from given configuration file.
        """
        if config_file is None or not os.path.exists(config_file):
            return {}

        with open(config_file, 'r', encoding='utf-8') as input_file:
            params = json.load(input_file)

        return {k: v for k, v in params.items() if not k.startswith('__')}