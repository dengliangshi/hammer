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
from hammer.tools.flow import Flow
from hammer.utils.logger import get_logger
from hammer.algorithms.trainer.trainer import Trainer
from hammer.algorithms.dataset.dataset_factory import DatasetFactory
from hammer.algorithms.models.text.classification.textcnn.config import Config
from hammer.algorithms.models.text.classification.textcnn.modeling import TextCNNFactory
from hammer.algorithms.dataset.text.text_cls_dataset import TextClsDataset, TextClsIterableDataset

# ------------------------------------------------------Global Variables----------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--config-file', default=None, help='path to dataset (root dir)')

# -----------------------------------------------------------Main-----------------------------------------------------------
def main(args):

    config = Config(args.config_file)
    logger = get_logger(config.model_name)

    # 
    flow = Flow([
    ])

    # 
    if config.dataset.iterable:
        dataset_cls = TextClsIterableDataset
    else:
        dataset_cls = TextClsDataset
    # build dataset
    dataset_factory = DatasetFactory(dataset_cls, flow, config, logger)
    # build model
    model_factory = TextCNNFactory(config, logger)

    if config.train.enable:
        trainer = Trainer(model_factory, dataset_factory, config, logger)
        trainer.train()

    if config.evaluate.enable:
        pass


if __name__ == '__main__':
    args =  parser.parse_args()
    main(args)
