#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library


# Third-party libraries


# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class AveragedMeter(object):

    def __init__(self, name: str, str_format: str=':.2f'):
        """Create a new averaged meter

        Args:
            name (str): name of the averaged meter.
            str_format (str, optional): format of the string output. Defaults to ':.2f'.
        """
        self.name = name
        self.str_format = str_format
        # reset this averaged meter
        self.reset()

    def update(self, value: float, n: int=1):
        """Update the state of this averaged meter.

        Args:
            value (float): the current value of this averaged meter.
            n (int, optional): the count of values. Defaults to 1.
        """
        # update the current value
        self.value = value
        # update the sum of values
        self.sum += value * n
        # update the count of values
        self.count += n
        # calculate the average of values
        self.average = self.sum / self.count
    
    def reset(self):
        """Reset the state of this averaged meter.
        """
        self.value = 0
        self.sum = 0
        self.count = 0
        self.average = 0

    def __str__(self) -> str:
        """String format method.

        Returns:
            str: string format of this averaged meter.
        """
        format_str = '{name} {value' + self.str_format + '} ({average' + self.str_format + '})'
        return format_str.format(name=self.name, value=self.value, average=self.average)
