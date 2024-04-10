#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import json
from logging import Logger

# Third-party libraries
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset

# User define module
from hammer.utils.config import Config

# ------------------------------------------------------Global Variables----------------------------------------------------
_ERROR_RETRY = 50

# -----------------------------------------------------------Main-----------------------------------------------------------
class ImageClsDataset(Dataset):

    def __init__(self,
        data_dir: str,
        split: str='train',
        input_img_mode: str='RGB',
        config: Config=None,
        logger: Logger=None
    ):
        """_summary_

        Args:
            data_dir (str): the dir of dataset.
            split (str, optional): _description_. Defaults to 'train'.
            input_img_mode (str, optional): _description_. Defaults to 'RGB'.
            config (Config, optional): _description_. Defaults to None.
            logger (Logger, optional): _description_. Defaults to None.
        """
        self.config = config
        self.logger = logger
        self._consecutive_errors = 0
        self.input_img_mode = input_img_mode

        assert os.path.isdir(data_dir), f'{data_dir} is not a directory'
        # load in samples from given file
        file_path = os.path.join(data_dir, f'{split}.json')
        self.samples = self._load_data(file_path)
        self.num_samples = len(self.samples)
        # mappings from category to index
        file_path = os.path.join(data_dir, 'mappings.json')
        self.category2idx = json.load(open(file_path, 'r'))

    def __getitem__(self, index: int):
        """_summary_

        Args:
            index (int): get sample 

        Raises:
            error: _description_

        Returns:
            (, int): _description_
        """
        sample = self.samples[index]
        # try to open the image file and convert it to the specified mode
        try:
            image = Image.open(sample['file_path'])
            if self.input_img_mode:
                image = image.convert(self.input_img_mode)
            # check if a transformation is specified and apply it if necessary
            if self.transform is not None:
                image = self.transform(image)
            # get the target label from the mappings from category to index
            target = self.category2idx[sample['category']]
        except Exception as error:
            # log a warning if an error occurs
            self.logger.warning(f'skipped sample (index {index}, file {sample["file_path"]}). {str(error)}')
            # increment the consecutive errors counter
            self._consecutive_errors += 1
            # if the error counter is below the specified threshold, try to get the next sample
            if self._consecutive_errors < _ERROR_RETRY:
                return self.__getitem__((index + 1) % self.num_samples)
            # if the error counter is equal or above the threshold, raise the error
            else:
                raise error
        # reset the consecutive errors counter
        self._consecutive_errors = 0

        return image, target

    def __len__(self):
        return len(self.samples)


class IterableImageClsDataset(IterableDataset):

    def __init__(self,
        data_dir: str,
        split: str='train',
        input_img_mode: str='RGB',
        config: Config=None,
        logger: Logger=None
    ):
        """Create iterable image classification dataset.

        Args:
            data_dir (str): the directory of dataset.
            split (str, optional): _description_. Defaults to 'train'.
            input_img_mode (str, optional): _description_. Defaults to 'RGB'.
            config (Config, optional): _description_. Defaults to None.
            logger (Logger, optional): _description_. Defaults to None.
        """
        self.config = config
        self.logger = logger
        self._consecutive_errors = 0
        self.input_img_mode = input_img_mode

        assert os.path.isdir(data_dir), f'{data_dir} is not a directory'
        # load in samples from given file
        file_path = os.path.join(data_dir, f'{split}.json')
        self.samples = self._load_data(file_path)
        self.num_samples = len(self.samples)
        # mappings from category to index
        file_path = os.path.join(data_dir, 'mappings.json')
        self.category2idx = json.load(open(file_path, 'r'))

    def _load_data(self, file_path: str) -> list:
        """This function reads a file line by line and loads the data into a list.

        Args:
            file_path (str): the path of the file to be read.

        Returns:
            list: a list containing the data loaded from the file.
        """
        with open(file_path, 'r', encoding='utf-8') as input_file:
            return [json.loads(line.strip()) for line in input_file]

    def __iter__(self):
        for sample in self.samples:
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            yield img, target

    def __len__(self) -> int:
        """Get the number of samples.

        Returns:
            int: the number of samples.
        """
        return self.num_samples
