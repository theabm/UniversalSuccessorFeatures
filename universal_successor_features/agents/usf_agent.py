import numpy as np
import torch
import exputils as eu
import exputils.data.logging as log
import warnings
import copy
from collections import namedtuple, deque
import universal_successor_features.networks as nn
import universal_successor_features.envs as envs
import random


Experiences = namedtuple("Experiences", ("agent_position_batch", "goal_batch", "action_batch", "reward_batch", "next_agent_position_batch", "terminated_batch", "truncated_batch"))

class UsfAgent():

    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            device = "cuda", # "cuda" or "cpu"
            discount_factor = 0.99,
            batch_size = 32,
            learning_rate = 5e-4,
            loss_weight = 0.01,
            eps  = 0.25,
            update_freq = 10,
            buffer_size = 1000000
        )
        return cnf
    
    def __init__(self, env, config = None, **kwargs):
        
        self.config = eu.combine_dicts(kwargs, config, UsfAgent.default_config())
        self.action_space = env.action_space.n
        self.position_size = env.observation_space["agent_position"].shape[1]
        self.features_size = env.observation_space["agent_position_features"].shape[1]

        # Setting the device
        if self.config.device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                warnings.warn('Cuda not available. Using CPU as device ...')
        else:
            self.device = torch.device("cpu")
        
        # Creating object instances
        self.config.network.state_size = self.position_size
        self.config.network.goal_size = self.position_size
        self.config.network.features_size = self.features_size
        self.config.network.num_actions = self.action_space
         
        self.policy_net = nn.UsfNN(self.config.network)

        self.memory = deque([], maxlen = self.config.buffer_size)

        self.epsilon = self.config.eps

        self.target_net = copy.deepcopy(self.policy_net)

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        self.loss = torch.nn.MSELoss()
        self.loss_weight = self.config.loss_weight
        self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr = self.config.learning_rate)
        
        self.batch_size = self.config.batch_size      

        self.discount_factor = self.config.discount_factor

        self.update_target_network_every_n_steps = self.config.update_freq
        self.steps_since_last_network_update = 1

        self.learning_starts_after = self.batch_size*2
    
    def start_episode(self, episode):
        return
    def end_episode(self):
        return
    def save(self, episode, step):
        return

    def choose_action(self, agent_position, goal_position, training):
        if training:
            return self._epsilon_greedy_action_selection(agent_position, goal_position).item()
        else:
            return self._greedy_action_selection(agent_position, goal_position).item()

    def _epsilon_greedy_action_selection(self, agent_position, goal_position):
        """Epsilon greedy action selection"""
        if torch.rand(1).item() > self.epsilon:
            return self._greedy_action_selection(agent_position, goal_position)
        else:
            return torch.randint(0,self.action_space,(1,)) 

    def _greedy_action_selection(self, agent_position, goal_position):
        with torch.no_grad():
            psi, w, phi = self.policy_net(
                    agent_position = torch.tensor(agent_position).to(torch.float).to(self.device),
                    goal_position = torch.tensor(goal_position).to(torch.float).to(self.device)
                )
            # shape of torch.sum result = (1,4) 
            # shape of q = (1,) shape of a = (1,)
            # verified explicitly multiplication works as expected .
            q, a = torch.max(torch.sum(torch.mul(psi, w.unsqueeze(1)), dim=2), dim=1)
            return a

    def _sample_experiences(self):
        experiences = random.sample(self.memory, self.batch_size)
        return Experiences(*zip(*experiences))

    def _build_tensor_from_batch_of_np_arrays(self, batch_of_np_arrays):
        # expected shape: [(1,n), (1,n), ..., (1,n)] where in total we have batch_size elements
        batch_of_np_arrays = np.array(batch_of_np_arrays)
        # batch of np_arrays has form (batch_size, 1, n) so after squeeze() we have (batch_size, n)
        batch_of_np_arrays = torch.tensor(batch_of_np_arrays).squeeze().to(torch.float)

        return batch_of_np_arrays
    
    def _train_one_batch(self):
        experiences = self._sample_experiences()
        goal_batch = self._build_tensor_from_batch_of_np_arrays(experiences.goal_batch).to(self.device)

        self.optimizer.zero_grad()
        target_batch_q, target_batch_psi, r = self._build_target_batch(experiences, goal_batch)
        predicted_batch_q, predicted_batch_psi, phi_w = self._build_predicted_batch(experiences, goal_batch)
        loss = self.loss(target_batch_q, predicted_batch_q) + self.loss_weight * self.loss(target_batch_psi, predicted_batch_psi) #+ self.loss(r, phi_w)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    # explicitly checked that it works.
    def _build_target_batch(self, experiences, goal_batch):
        next_agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.next_agent_position_batch).to(self.device) # shape (batch_size, n)

        # reward and terminated batch are handled differently because they are a list of floats and bools respectively and not a list of np.arrays
        reward_batch = torch.tensor(experiences.reward_batch).to(torch.float).to(self.device) # shape (batch_size,)
        terminated_batch = torch.tensor(experiences.terminated_batch).to(self.device) # shape (batch_size,)

        with torch.no_grad():
            sf_s_g, w, reward_phi_batch = self.target_net(next_agent_position_batch, goal_batch)
            q = torch.sum(torch.mul(sf_s_g, w.unsqueeze(1)), dim=2) # shape (batch, num_actions)
            
        qm, action = torch.max(q, axis = 1) # both have shape (batch,)

        target_q = reward_batch + self.discount_factor * torch.mul(qm, ~terminated_batch) # shape (batch_size,)

        terminated_batch = terminated_batch.unsqueeze(1) # shape(batch,1)
        action = action.reshape(self.batch_size, 1, 1).tile(self.features_size).to(self.device) # shape (batch_size,1,n)

        target_psi = reward_phi_batch + self.discount_factor * torch.mul(sf_s_g.gather(1, action).squeeze(), ~terminated_batch) # shape (batch, features_size)

        del reward_phi_batch
        del next_agent_position_batch
        del terminated_batch

        return target_q, target_psi, reward_batch

    # explicitly checked it works
    def _build_predicted_batch(self, experiences, goal_batch):
        agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.agent_position_batch).to(self.device)
        action_batch = torch.tensor(experiences.action_batch).unsqueeze(1).to(self.device)

        sf_s_g, w, phi = self.policy_net(agent_position_batch, goal_batch)
        q = torch.sum(torch.mul(sf_s_g, w.unsqueeze(1)), dim=2)

        predicted_q = q.gather(1,action_batch).squeeze() # shape (batch_size,)
            
        action_batch = action_batch.reshape(self.batch_size, 1, 1).tile(self.features_size)
        predicted_psi = sf_s_g.gather(1, action_batch).squeeze() # shape (batch_size, features_size)

        del sf_s_g
        del agent_position_batch
        del action_batch

        return predicted_q, predicted_psi, torch.sum(phi * w, dim = 1)


    def train(self, transition):
        
        self.memory.append(transition)
        
        if len(self.memory) < self.learning_starts_after:
            return

        self._train_one_batch()
        
        if self.steps_since_last_network_update >= self.update_target_network_every_n_steps:
            self.steps_since_last_network_update = 1

            self._update_target_network()
        else:
            self.steps_since_last_network_update += 1

    def _update_target_network(self):
        self.target_net = copy.deepcopy(self.policy_net)
