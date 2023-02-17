import exputils as eu
from collections import deque, namedtuple
import random

#This is okay for any type of agent, whether it be multigoalDQN or USF. No need to save weights or features since, if these are an input,
#then is assumed that we are able to compute them with the information we are storing (s,a,s+) and (g).

#Note that state represents any type of information that acts as input for the agent.
Transition = namedtuple("Transition", ("goal", "state", "action", "reward", "next_state", "terminated", "truncated"))

class ExperienceReplayMemory():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            capacity = 1000000,
        )
    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        self.memory = deque([],maxlen=self.config.capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__ == '__main__':
    memory = ExperienceReplayMemory()
