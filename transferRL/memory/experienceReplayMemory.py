from collections import deque, namedtuple
import random

Transition = namedtuple("Transition", ("goal", "state", "action", "reward", "next_state", "terminated", "truncated"))

class ExperienceReplayMemory():

    def __init__(self, capacity = 1000000):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size = 32):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)