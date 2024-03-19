#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import random
from glob import glob
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

# Third-party libraries


# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
def PreparePretraining(object):

    def __init__(self, config, logger):

        self.config = config
        self.logger = logger

    def _iterator_from_path(self, input_path: str):
        """_summary_

        Args:
            input_path (str): _description_

        Yields:
            _type_: _description_
        """
        for file_path in glob(input_path):
            with open(file_path, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    yield line

    def prepare(self, input, tokenizer):

        if os.path.isfile(input):
            iterator = open(input, 'r', encoding='utf-8')
        else:
            iterator = self._iterator_from_path(input)

        pool = ThreadPool(self.config.num_thread)
        imap_iterator = pool.imap(tokenizer.tokenize, iterator, self.config.chunk_size)

        for token_ids in tqdm(imap_iterator):
            index = random.randint(1, self.config.num_parts)

            
            
            output_file = os.path.join(self.config.output_path, f'dataset_{index}.txt')