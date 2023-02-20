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
        
        self._pusher_util = self._push_build
    
    def _push_build(self,**kwargs):
        global Transition
        Transition = namedtuple("Transition", kwargs.keys())

        self._push(**kwargs)

        self._pusher_util = self._push

    def _push(self, **kwargs):
        """Save a transition"""
        self.memory.append(Transition(**kwargs))

    def push(self, **kwargs):
        self._pusher_util(**kwargs)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    memory = ExperienceReplayMemory()
