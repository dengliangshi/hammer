#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2024
# Author: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os

# Third-party libraries


# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
def Trad2Simple(object):

    def __init__(self, config, logger):
        self.trad2simple = {}

    def process(self, text, offset=0):
        result = ''
        trace = {}
        for index, character in enumerate(text):
            if character in self.trad2simple:
                target = self.trad2simple.get(character)
                if self.config.save_trace:
                    trace.append({
                        'start': index + offset,
                        'end': index+ offset + 1,
                        'text': character,
                        'target': target
                    })
            else:
                target = character
            result += character
        return result, trace
    
    def 
