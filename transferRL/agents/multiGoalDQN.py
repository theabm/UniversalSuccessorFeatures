import numpy as np
import torch
import config
import exputils as eu
import warnings
import copy
import experienceReplayMemory as mem
import multigoalDQNNetwork as dnn
from collections import namedtuple

Experiences = namedtuple("Experiences", ("goal_batch", "state_batch", "action_batch", "reward_batch", "next_state_batch", "terminated_batch", "truncated_batch"))


class MultigoalDQN():

    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            device = "gpu",
            env = eu.AttrDict(
                obs_space_size = 2,
                num_actions = 4,
            ),
            discount_rate = 0.9,
            learning_starts_after = 128,
            batch_size = 64,
            learning_rate = 1e-3,
            epsilon = eu.AttrDict(
                max = 0.1,
                min = 0.01,
            ),
            schedule = eu.AttrDict(
                type = "linear",
                frequency = 5, #in episodes
            ),
            update_rule = eu.AttrDict(
                type = "hard",
                frequency = 10, #in episodes
                alpha = 0.01,
            ),
            memory = eu.AttrDict(
                type = "normal",
                size = 1e6,
            ),
            network = eu.AttrDict(
                hidden_layer_structure = [64,128,64],
                optimizer = torch.optim.Adam,
                loss = torch.nn.MSELoss,
            ),
            save = eu.AttrDict(
                directory = None,
                frequency = 10, # in episodes
            ),
        )
        return cnf


    def __init__(self, config = None, **kwargs):
        
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if self.config.device == "gpu":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                warnings.warn('Cuda not available. Using CPU as device ...')
        else:
            self.device = torch.device("cpu")

        self.additional_info = ["goal"]
        self.state_size = self.config.env.obs_space_size + len(self.additional_info)
        
        self.policy_net = dnn.MultigoalDQNNetwork(in_features = self.state_size, out_features = self.config.env.num_actions)
        self.policy_net.to(self.device)

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.to(self.device)

        self.loss = torch.nn.MSELoss()
        self.optimizer = self.config.network.optimizer(self.policy_net.parameters(), lr = self.config.learning_rate)
        
        self.batch_size = self.config.batch_size      
        self.memory = mem.ExperienceReplayMemory(capacity = int(self.config.memory.size))

        self.discount_rate = self.config.discount_rate
        self.eps_max = self.config.epsilon.max
        self.eps_min = self.config.epsilon.min

        self.current_epsilon = self.eps_max

        self.update_frequency = self.config.update_rule.frequency
        self.learning_starts_after = self.config.learning_starts_after 

        self.episode_counter = 0
        self.steps_since_last_training = 0

    def remember(self, **kwargs):
        self.memory.push(kwargs)
        return

    
    def epsilon_greedy_action_selection(self, obs):
        """Epsilon greedy action selection"""
        if torch.rand(1).item() > self.current_epsilon:
            with torch.no_grad():
                return torch.argmax(self.policy_net(obs)).item()
        else:
            return torch.randint(0,self.config.env.num_actions,(1,)).item() 
            
    def _sample_experiences(self):
        experiences = self.memory.sample(self.batch_size)
        return Experiences(*zip(*experiences))

    def _get_target_batch(self, next_state_batch, reward_batch, terminated_batch):
        with torch.no_grad():
            max_action = torch.argmax(self.policy_net(next_state_batch), axis = 1).unsqueeze(1).to(self.device)
            target = reward_batch + self.discount_rate * torch.mul(self.target_net(next_state_batch).gather(1,max_action).squeeze(), 1-torch.tensor(terminated_batch))
        return target

    def train(self):
        if len(self.memory) < self.learning_starts_after:
            # Need to acquire more experience
           return
        experiences = self._sample_experiences()

        next_state_batch = torch.tensor(experiences.next_state_batch).to(self.device)
        reward_batch = torch.tensor(experiences.reward_batch).to(self.device)
        terminated_batch = torch.tensor(experiences.terminated_batch).to(self.device)

        target_batch = self._get_target_batch(next_state_batch, reward_batch, terminated_batch)
    
        del next_state_batch
        del reward_batch
        del terminated_batch
        

        state_batch = torch.tensor(experiences.state_batch).to(self.device)
        action_batch = torch.tensor(experiences.action_batch).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(target_batch, self.policy_net(state_batch).gather(1, action_batch).squeeze())
        
        loss.backward()
        self.optimizer.step()
        
        del state_batch
        del action_batch

    def update_target_network(self):
        pass



if __name__ == '__main__':

    dqn = MultigoalDQN()
    print(dqn.policy_net)