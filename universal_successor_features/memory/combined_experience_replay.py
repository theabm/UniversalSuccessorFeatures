import exputils as eu
from collections import deque
import random

class CombinedExperienceReplayMemory():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            capacity = 1000000,
        )
    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        self.memory = deque([],maxlen=self.config.capacity)

        self.last_transition = [None]

    def push(self, transition):
        self.memory.append(transition)

        self.last_transition[0] = transition

    def sample(self, batch_size):
        initial_batch = random.sample(self.memory, batch_size-1)
        return initial_batch + self.last_transition

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self,key):
        return self.memory[key]


if __name__ == '__main__':
    memory = CombinedExperienceReplayMemory()
