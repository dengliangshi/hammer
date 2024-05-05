#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import json

# Third-party libraries
import torch

# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class TextClsDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, flow, config):
        self.samples, self.num_samples = self._load_dataset(file_path)
        self.flow = flow
        self.config = config

    def _load_dataset(self, file_path: str) -> list:
        """Load dataset from file.

        Args:
            file_path (str): the path of dataset file.

        Returns:
            list: the loaded dataset.
        """
        samples = []
        with open(file_path, 'r', encoding='utf-8') as input_file:
            for line in input_file:
                sample = json.loads(line.strip())
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.flow(self.samples[index])


class TextClsIterableDataset(torch.utils.data.IterableDataset):

    def __init__(self, file_path, flow, config, logger):
        self.samples, self.num_samples = self._load_dataset(file_path)
        self.flow = flow
        self.config = config