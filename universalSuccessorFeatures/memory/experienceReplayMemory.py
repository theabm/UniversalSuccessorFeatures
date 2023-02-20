import exputils as eu
from collections import deque, namedtuple
import random

#Transition = namedtuple("Transition", ("state", "goal", "action", "reward", "next_state", "terminated", "truncated"))

#This is a flexible class which automatically builds a namedtuple the first time that push is called and uses the keys of the arguments as keys for the tuple.
#It stores the transitions as named tuples for better memory handling.
class ExperienceReplayMemory():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            capacity = 1000000,
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
    memory = ExperienceReplayMemory()
