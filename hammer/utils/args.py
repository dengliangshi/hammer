#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import argparse

# Third-party libraries


# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """set up argument parser.

    Returns:
        argparse.Namespace: namespace for argument parser.
    """
    parser = argparse.ArgumentParser('arguments for hammer.')

    parser.add_argument('--config_name', help='name of the config file.')

    return parser.parse_args()
