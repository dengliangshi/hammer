#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import math

# Third-party libraries
import random

# User define module
from hammer.utils.config import Config
from hammer.tools.document.block import Block

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Blocks(object):

    def __init__(self, blocks: list, height: int, width: int, config: Config):

        self.blocks = blocks
        self.config = config
        # resize blocks
        if config.resize_factor is None:
            config.resize_factor = 1.0
        self._resize_block(blocks, config.resize_factor)
        #
        height, width = self._clip_blocks(blocks, config.margin)
        #
        self.dist_norm = math.sqrt(height**2 + width**2)

        self.blocks = [Block(block, config) for block in blocks if random.random() <= config.keep_prob]
        


    def _resize_block(self, blocks: list, resize_factor: float = 1.0) -> None:
        """Rescales the bounding boxes of each block in the given list.

        Args:
            blocks (list): a list containing block information, where each block is a dictionary with a 'bbox' key
                that holds a list of four points, each a tuple representing (x, y) coordinates.
            resize_factor (float, optional): the scaling factor, defaulting to 1.0 which implies no scaling. Values less than 1
                cause shrinking, while values greater than 1 enlarge the blocks.
        """
        for block in blocks:
            bbox = block['bbox']
            # adjust the width of the block based on the resize factor while keeping the left boundary unchanged 
            x2 = (1-resize_factor)*bbox[0][0] + resize_factor*bbox[1][0]
            x3 = (1-resize_factor)*bbox[3][0] + resize_factor*bbox[2][0]
            # adjust the width of the block based on the resize factor while keeping the center line unchanged
            y1 = (1+resize_factor)*bbox[0][0] + (1-resize_factor)*bbox[3][0]
            y2 = (1+resize_factor)*bbox[1][0] + (1-resize_factor)*bbox[2][0]
            y3 = (1-resize_factor)*bbox[1][0] + (1+resize_factor)*bbox[2][0]
            y4 = (1-resize_factor)*bbox[0][0] + (1+resize_factor)*bbox[3][0]
            # save resized boubding box
            block['resized_bbox'] = [[bbox[0][0], y1], [x2, y2], [x3, y3], [bbox[3][0], y4]]

    def _search_boundary(self, blocks: list) -> list:
        """Calculates the boundaries of the given block list.

        Args:
            blocks (list): a list of blocks, each represented as a dictionary with 'resized_bbox',
                a list of (x, y) coordinate tuples.

        Returns:
            list: a list of four elements containing the minimum x, minimum y, maximum x,
                and maximum y coordinates of all blocks.
        """
        x_list = []
        y_list = []
        # gather all x and y coordinates from all blocks
        for block in blocks:
            x_list.extend([point[0] for point in block['resized_bbox']])
            y_list.extend([point[1] for point in block['resized_bbox']])
        return min(x_list), min(y_list), max(x_list), max(y_list)
    
    def _clip_blocks(self, blocks: list, margin: float = 0.02) -> tuple:
        """Clips the given list of blocks, leaving additional space outside their boundaries based on the margin.

        Args:
            blocks (list): A list containing block information, where each block has a 'resized_bbox' key.
                This value is a list of coordinates (x, y) representing the block's four boundary points.
            margin (float, optional): The proportion of extra space to leave around
                the blocks' total dimensions. Defaults to 0.02.

        Returns:
            tuple: Returns the clipped height and width.
        """
        # find the boundary of all blocks
        x_min, y_min, x_max, y_max = self._search_boundary(blocks)
        # calculate the additional clipping space
        x_space = (x_max - x_min) * margin
        y_space = (y_max - y_min) * margin
        # calculate the clipping offset, ensuring no negative values
        x_offset = max(0, x_min - x_space)
        y_offset = max(0, y_min - y_space)
        # clip each block and store the clipped bounding box
        for block in blocks:
            block['clipped_bbox'] = [[point[0] - x_offset, point[1] - y_offset]
                for point in block['resized_bbox']]
        # calculate the total height and width after clipping
        height = y_max - y_min + 2 * y_offset
        width = x_max - x_min + 2 * x_offset
        return height, width

    def to_text(self, sep: str = ' ') -> str:

        items = []
        
        for block in self.blocks:

            if :
                items.append(sep)
            items.append(block.text)
            
        return ''.join(items)

    def get_blocks(self)
