import exputils as eu
from collections import deque
import random

class PrioritizedExperienceReplayMemory():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            capacity = 1000000,
            alpha = 0,
            beta0 = 0,
        )
    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        self.memory = deque([],maxlen=self.config.capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self,key):
        return self.memory[key]


if __name__ == '__main__':
    memory = PrioritizedExperienceReplayMemory()
