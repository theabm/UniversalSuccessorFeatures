import exputils as eu
from collections import deque
import random
import torch

class PrioritizedExperienceReplayMemory():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            capacity = 1000000,
            alpha = 0, # how much to prioritize
            beta0 = 0, # how much to correct bias 0<= beta <= 1. This is annealed linearly throughout episodes
            eps = 0, # makes probability of sampling a transition with zero td error non null
            max_priority = 0, # the max priority for newly obtained transitions, ensures that they will be sampled at least once
        )
    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        self.tree = SumTree(self.config.capacity) # tree structure that will allow us to sample with specific weights

        self.memory = deque([],maxlen=self.config.capacity) # data structure that holds actual data

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self,key):
        return self.memory[key]


class SumTree():
    """Implementation of Sum Tree specific for prioritized experience replay"""
    @staticmethod
    def default_config():
        return eu.AttrDict(
            memory_size = 0,
        )

    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, SumTree.default_config())
        self.memory_size = self.config.memory_size

        self.tree_size = 2*self.memory_size-1
        
        self.tree = torch.zeros(self.tree_size)

    def add(self, index_in_memory_array, priority):
        """Adds an element inside the tree with a certain value"""
        # The idea is as follows: given a memory array of size n, since these nodes are assumed to be leaf nodes, the corresponding tree will have size 2*n - 1.
        # In fact, nodes from 0 to n-2 (the first n-1 nodes) are intermediate nodes that contain the sums of their children nodes. Then, the nodes from n-1 to 2n-2 (the last
        # n nodes of real interest) contain the priority of our nodes in the memory in the same order as the one in the array. 
        # Therefore, since we are putting the n nodes of the memory at the end of the tree array, we must add (n-1) to the index of the memory element to obtain the corresponding
        # position in the tree array. Below is the illustration:
        # [-5, 8, 2, 4] (these elements are the priorities of four transitions in our memory)
        #   0, 1, 2, 3  -> the positions of the elements
        # [-,-,-,-5, 8, 2, 4] -> the elements as we want them positioned in the sum tree array
        #  0,1,2, 3, 4, 5, 6  -> the indexes in the sum tree (if you take any index in the memory array and sum n-1 you obtain the index in the tree)
        #  
        corresponding_index_in_tree = index_in_memory_array + (self.memory_size - 1)
        # having found the index in the tree, we simply update that value to the new value. 
        self.update(corresponding_index_in_tree, priority)
    
    def update(self, index, priority):
        """Modity the priority of the element at the given index in the tree"""

        # As the sum tree algorithm requires, to update a value, we need to propagate the changes upwards to all the parents so that these may satisfy the sum tree property.
        # Therefore, we store the change in value wrt to before.
        change_in_value = priority - self.tree[index]

        # We change the priority of the tree element at the index
        self.tree[index] = priority

        # We find the parent of the index
        parent_index = (index-1)//2

        # While the parent exists (this will stop at the root)
        # We propagate change backwards adding the value and find the next parent
        while parent_index >=0 :
            self.tree[parent_index] += change_in_value
            parent_index = (parent_index-1)//2

    def get(self, priority):
        """Get a value in the tree using the sum tree algorithm. In other words, we retrieve the the element of the sum tree for which the desired
           priority is in it's range.
        """
        index = 0
        while index < (self.size - 1):
            left = 2*index + 1
            right = 2*index + 2
            if priority < self.tree[index]:
                index = left  
            else:
                priority = self.tree[index] - self.tree[left]
                index = right
        return self.tree[index]