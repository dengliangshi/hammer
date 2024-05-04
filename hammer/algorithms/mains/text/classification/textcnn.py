#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import argparse

# Third-party libraries


# User define module
from hammer.utils.logger import get_logger
from hammer.algorithms.trainer.trainer import Trainer
from hammer.algorithms.models.text.classification.textcnn.config import Config
from hammer.algorithms.models.text.classification.textcnn.modeling import TextCNNFactory

# ------------------------------------------------------Global Variables----------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')

# -----------------------------------------------------------Main-----------------------------------------------------------
def main(args):

    config = Config(args.config_file)
    logger = get_logger(config.model_name)

    dataset_factory = None
    model_factory = TextCNNFactory.create(config, logger)

    if config.train.enable:
        trainer = Trainer(model_factory, dataset_factory, config, logger)
        trainer.train()

    if config.evaluate.enable:
        pass


if __name__ == '__main__':
    args =  parser.parse_args()
    main(args)
