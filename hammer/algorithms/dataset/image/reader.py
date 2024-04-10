#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import json

# Third-party libraries
from PIL import Image

# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Reader(object):

    def __init__(self, file_path: str, split: str='train', input_img_mode: str='RGB'):
        """_summary_

        Args:
            file_path (str): _description_
            split (str, optional): _description_. Defaults to 'train'.
            input_img_mode (str, optional): _description_. Defaults to 'RGB'.
        """
        self.samples = self._load_data(file_path, split)
        self.input_img_mode = input_img_mode

    def _load_data(self, file_path: str) -> list:
        """This function reads a file line by line and loads the data into a list.

        Args:
            file_path (str): the path of the file to be read.

        Returns:
            list: a list containing the data loaded from the file.
        """
        with open(file_path, 'r', encoding='utf-8') as input_file:
            return [json.loads(line.strip()) for line in input_file]

    def __getitem__(self, index: int):

        sample = self.samples[index]

        image = Image.open(sample['file_path']).convert(self.input_img_mode)

        return 
