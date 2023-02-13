import torch
import numpy as np
import exputils as eu
import exputils.data.logger as log
import warnings
from collections import namedtuple
import transferRL.memory.experienceReplayMemory


class UniversalSuccessorFeatures():

    @staticmethod
    def default_config():
        return eu.AttrDict()


    def __init__(self) -> None:
        pass