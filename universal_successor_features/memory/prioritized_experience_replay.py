import exputils as eu
import numpy as np
import torch

class PrioritizedExperienceReplayMemory():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            capacity = 1000000,
            alpha = 0, # how much to prioritize
            beta0 = 1, # how much to correct bias 0<= beta <= 1. This is annealed linearly throughout episodes
            eps = 0.01, # makes probability of sampling a transition with zero td error non null
            max_priority = 0.01, # the max priority for newly obtained transitions, ensures that they will be sampled at least once
        )
    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())
        
        self.capacity = self.config.capacity
        self.beta_current = self.config.beta0
        self.max_priority = self.config.max_priority

        self.tree = SumTree(self.capacity) # the sum tree that will enable sampling from the distribution efficiently.

        self.memory = [0 for i in self.capacity] # data structure that holds actual data
        self.weights = np.zeros(self.capacity) # array of weights

        self.size_so_far = 0 # The size of the data so far. This keeps track of where to sample in the beginning.  
        self.index_to_store = 0 # The index where we must store new data. This keeps track of the fixed size of the memory.

        self.max_weight = None
        

    def push(self, transition):
        """Push new transitions in the memory and in the tree with max priority."""
        # the index to store must always be between 0, capacity-1
        self.index_to_store %= self.config.capacity

        # Add the transition inside the memory and add the corresponding element with max priority inside the tree.
        self.memory[self.index_to_store] = transition

        # Calculate the weight, store it in memory in the appropriate place
        weight = (1/(self.config.capacity*self.max_priority)**self.beta_current)
        self.weights[self.index_to_store] = weight

        self.tree.add(self.index_to_store, self.max_priority)

        # The index to store is increased by one (when == capacity, line 6 will make it go back to zero)
        self.index_to_store += 1
        self.size_so_far = min(self.config.capacity, self.size_so_far+1) 


    def sample(self, batch_size):
        """Sample a batch of elements according to the rules of prioritized experience replay.
           As per the description: 'To sample a minibatch of size k, the range [0, ptotal] is divided equally into k ranges.
           Next, a value is uniformly sampled from each range. Finally the transitions that correspond to each of these sampled
           values are retrieved from the tree.'
           Returns: batch_of_elements, their_weights
        """ 
        ptotal = self.tree.ptotal()
        range_length = ptotal/batch_size

        values = [np.random.uniform(k*range_length, (k+1)*range_length) for k in range(batch_size)]

        # Array that will store indexes of where to look in the memory to retrieve the relevant transition
        self.indexes = np.zeros(batch_size)

        # Find first value manually purely to have first value of max_w
        _, index = self.tree.get(values[0])
        max_w = self.weights[index]
        self.indexes[0] = index
        
        for i, value in enumerate(values[1:]):
            _, index = self.tree.get(value)
            max_w = max(max_w, self.weights[index])
            self.indexes[i] = index

        # Get normalized weights 
        weights = self.weights[self.indexes]/max_w

        return [self.memory[i] for i in self.indexes], weights

    def update_samples(self, batch_of_new_td_errors):
        # Expected type is a batch of torch tensors (batch_size,)
        """Updates the priorities at the previous indexes after having sampled"""
        priority = torch.abs(batch_of_new_td_errors) + self.config.eps
        new_max = torch.max(priority)
        self.max_priority = max(self.max_priority, new_max)

        for i in range(len(batch_of_new_td_errors)):
            self.tree.update(self.indexes[i], priority[i].item())
    
    def anneal_beta(self, schedule_length):
        """Linearly anneal beta to 1 when learning ends."""
        m = (1-self.config.beta0)/schedule_length
        self.value = min(self.beta_current + m, 1)




class SumTree():
    """Implementation of Sum Tree specific for prioritized experience replay. Note that this implementation
       goes hand in hand with the memory class before which implicitly handles insertions/removals inside the
       sum tree.
    """
    @staticmethod
    def default_config():
        return eu.AttrDict(
            memory_size = 0,
        )

    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, SumTree.default_config())
        self.memory_size = self.config.memory_size

        self.tree_size = 2*self.memory_size-1
        
        self.tree = np.zeros(self.tree_size)
    
    def ptotal(self):
        return self.tree[0]

    def add(self, index_in_memory_array, priority):
        """Adds an element inside the tree with a certain value"""
        # The idea is as follows: given a memory array of size n, since these nodes are assumed to be leaf nodes, the corresponding tree will have size 2*n - 1.
        # In fact, nodes from 0 to n-2 (the first n-1 nodes) are intermediate nodes that contain the sums of their children nodes. Then, the nodes from 
        # n-1 to 2n-2 (the last n nodes of real interest) contain the priority of our nodes in the memory in the same order as the one in the array. 
        # Therefore, since we are putting the n nodes of the memory at the end of the tree array, we must add (n-1) to the index of the memory element 
        # to obtain the corresponding position in the tree array. Below is the illustration:
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
           priority is in its range.
           Returns priority value and corresponding index in the memory.
        """
        # start with root
        index = 0
        # Since we have n leaf nodes (the size of the memory), and the whole tree is of size (2*n - 1), then we have (n-1) elements in the
        # beginning of the array, followed by n leaf nodes. 
        # The index of the first (n-1) elements goes from 0 to n-2, so the first leaf node will have index (n-1).
        # Therefore, we iterate the procedure until we find a leaf node which happens when the index >= n-1
        while index < (self.memory_size - 1):
            # For a node at index i, the left and right children nodes will be at 2*i+1 and 2*i+2 respectively.
            left = 2*index + 1
            right = 2*index + 2
            if priority <= self.tree[left]:
                index = left  
            else:
                # if the priority is greater than the right element 
                priority -= self.tree[left]
                index = right
        return self.tree[index], index - (self.memory_size - 1)
    
    def display(self):
        import math

        levels = math.ceil(math.log2(self.tree_size + 1))
        i = 0
        num_elems = 0
        level = 0
        print("\t"*int(math.floor(2**(levels-1)-1)), end = "")
        while level < levels and i<self.tree_size:
            num_seps = "\t"*int(math.floor(2**(levels-level)-1))
            print("%8d"%self.tree[i], end = "")
            print(num_seps, end = "", sep = "")
            num_elems += 1
            if num_elems == 2**level:
                print("\n")
                level += 1
                num_tabs = "\t"*int(math.floor(2**(levels-1-level)-1))
                print(num_tabs)
                print(num_tabs, end = "")
                num_elems = 0
            i+=1
        print("\n")

if __name__ == '__main__':

    my_memory = [8,2,1,4,0,0,0,0]
    my_tree = SumTree(memory_size = len(my_memory))
    
    for i,val in enumerate(my_memory):
        my_tree.add(i, val)
    my_tree.display()

    print(my_tree.get(1))
    
