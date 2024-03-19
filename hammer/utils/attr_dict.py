#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
from typing import Any

# Third-party libraries


# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class AttrDict(dict):

    def __getattr__(self, name: str) -> Any:
        """get attribute method of this class

        Args:
            name (str): name of target atrribute.

        Returns:
            Any: the value of specified attribute.
        """
        if name.startswith('__'):
            return object.__getattribute__(self, name)

        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            try:
                return self[name]
            except KeyError:
                return None

    def __setattr__(self, name: str, value: Any) -> None:
        """set up attribute method of this class.

        Args:
            name (str): the name of target attribute.
            value (Any): the value of this attribute.

        Raises:
            AttributeError: raised when failed to set up attribute.
        """        
        if not name.startswith('__'):
            try:
                object.__getattribute__(self, name)
            except AttributeError:
                try:
                    self[name] = value
                except:
                    raise AttributeError(name)
            else:
                object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)
