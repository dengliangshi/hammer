#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library


# Third-party libraries


# User define module
from hammer.utils.config import Config

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Block(object):

    def __init__(self, block: dict, config: Config):
        """_summary_

        Args:
            block (dict): _description_
            config (Config): _description_
        """
        # block id
        self.id = block.get('id')
        # text of this block
        self.text = block.get('text')
        # labels of this block
        self.labels = block.get('labels')
        # category of this block
        self.category = block.get('category')
        
        # real coordinates of this block
        self.x1, self.y1, self.x2, self.y2 = block.get('bbox')
        # center coordinates of this block
        self.center_x = (self.x1 + self.x2) / 2
        self.center_y = (self.y1 + self.y2) / 2
        # width and height of this block
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1

        # coordinates of this block after resized
        self.resized_x1, self.resized_y1, self.resized_x2, self.resized_y2 = block.get('resized_bbox')
        # coordinates of this block after clipped
        self.clipped_x1, self.clipped_y1, self.clipped_x2, self.clipped_y2 = block.get('clipped_bbox')

    def _layout_features(self, ):
        pass

    def _region_of_interest(self, ):
        pass

    def _text_features(self, ):
        pass

    def _label_features(self, ):
        pass

    def _category_features(self, ):
        pass

    def _bbox_features(self, ):
        pass
