import torch
import numpy as np
import exputils as eu
import exputils.data.logger as log
import warnings
from collections import namedtuple

class USFAgent():

    @staticmethod
    def default_config():
        return eu.AttrDict()


    def __init__(self) -> None:
        pass