#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import io
import os
import base64
import requests
from logging import Logger

# Third-party libraries
from PIL import Image as PILImage

# User define module
from hammer.utils.config import Config

# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Image(object):

    def __init__(self, image_file: str, config: Config, logger: Logger):
        self.image_file = image_file
        self.config = config
        self.logger = logger
        if os.path.exists(image_file):
            self.image = PILImage.open(image_file)
        elif image_file.startswith('http'):
            self.image = Image.download(image_file)
        else:
            raise ValueError('image_file must be a local file or a URL')
        
        width, height = self.image.size

        self.image = Image.resize(self.image, width, height, self.config.image_resize_factor)
        self.resized_width, self.resized_height = self.image.size

    def download(self, url: str, timeout: int = 10) -> PILImage:
        """
        Download an image from the specified URL.

        Args:
            url (str): The URL of the image.
            timeout (int): The request timeout in seconds, default is 10.

        Returns:
            PIL.Image.Image: The downloaded image, or None if the download fails.
        """
        image = None
        try:
            # attempt to download the image from the given URL
            response = requests.get(url, timeout=timeout)  
            # successful request
            if response.status_code == 200:
                image = PILImage.open(io.BytesIO(response.content))
            else:
                # request failed, log the warning
                self.logger.warn(f'request image failed: {url}')
        # handle network-related exceptions
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            self.logger.warn(f'Download image failed: {url}. Error message: {str(e)}')
        # handle unexpected exceptions
        except Exception as e:
            self.logger.warn(f'An unexpected error occurred: {url}. Error message: {str(e)}.')
            
        return image
    
    def resize(self, image, image_size):

        image.resize(image_size)
    
    def to_base64_str(self):

        bytes = io.BytesIO()
        self.image.save(bytes, format='PNG')

        base64_str = base64.encode(bytes.getvalue()).decode('utf-8')

        return base64_str
