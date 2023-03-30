import exputils as eu
from collections import deque
import random
import numpy as np

class ExperienceReplayMemory():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            capacity = 1000000,
        )
    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, ExperienceReplayMemory.default_config())

        self.memory = deque([],maxlen=self.config.capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size), np.ones(batch_size)

    # Useless methods for coherent interace
    def update_samples(self, batch_of_new_td_errors):
        return
    
    def anneal_beta(self, schedule_length):
        return

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self,key):
        return self.memory[key]


if __name__ == '__main__':
    memory = ExperienceReplayMemory()
