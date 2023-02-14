import numpy as np
import torch
import exputils as eu
import exputils.data.logging as log
import warnings
import copy
from collections import namedtuple
import transferRL.memory.experienceReplayMemory as mem
import transferRL.networks.multigoalDQNNetwork as dnn


Experiences = namedtuple("Experiences", ("goal_batch", "state_batch", "action_batch", "reward_batch", "next_state_batch", "terminated_batch", "truncated_batch"))


class MultigoalDQN():

    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            device = "cuda", # "cuda" or "cpu"
            env = eu.AttrDict(
                obs_space_size = 2,
                num_actions = 4,
            ),
            goal_size = 2,
            discount_rate = 0.99,
            batch_size = 32,
            learning_rate = 5e-4,
            train_for_n_iterations = 5,
            train_every_n_steps = 1,
            epsilon = eu.AttrDict(
                decay_type = "none",  #"none","linear" or "exponential"
                no_decay_params = eu.AttrDict(
                    value = 0.25,
                ),
                linear_decay_params = eu.AttrDict(
                    max = 1.0,
                    min = 0.1,
                    scheduled_episodes = 1000,
                ),
                exponential_decay_params = eu.AttrDict(
                    max = 1.0,
                    min = 0.1,
                    decay_factor = 0.99,
                ),
            ),
            target_network_update = eu.AttrDict(
                rule = "soft",  # "hard" or "soft"
                every_n_steps = 1, 
                alpha = 0.999,  # taraget network params will be updated as P_t = alpha * P_t + (1-alpha) * P_p   where P_p are params of policy network
            ),
            memory = eu.AttrDict(
                type = "normal",
                size = 1e6,
            ),
            network = eu.AttrDict(
                optimizer = torch.optim.Adam,
                loss = torch.nn.MSELoss,
            ),
            log = eu.AttrDict(
                loss_per_step = True,
                epsilon_per_episode = True,
            ),
            save = eu.AttrDict(
                directory = None,
                frequency = 10, # in episodes
            ),
        )
        return cnf


    def __init__(self, config = None, **kwargs):
        
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if self.config.device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                warnings.warn('Cuda not available. Using CPU as device ...')
        else:
            self.device = torch.device("cpu")

        self.policy_net = dnn.MultigoalDQNNetwork(state_size=self.config.env.obs_space_size, goal_size=self.config.goal_size, num_actions=self.config.env.num_actions)
        self.policy_net.to(self.device)

        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.to(self.device)

        self.loss = self.config.network.loss()
        self.optimizer = self.config.network.optimizer(self.policy_net.parameters(), lr = self.config.learning_rate)
        
        self.batch_size = self.config.batch_size      
        self.train_every_n_steps = self.config.train_every_n_steps
        self.steps_since_last_training = 0

        self.memory = mem.ExperienceReplayMemory(capacity = int(self.config.memory.size))

        self.discount_rate = self.config.discount_rate

        if self.config.epsilon.decay_type == "none":
            self.current_epsilon = self.config.epsilon.no_decay_params.value
        elif self.config.epsilon.decay_type == "linear":
            self.eps_max = self.epsilon.linear_decay_params.max
            self.eps_min = self.epsilon.linear_decay_params.min
            self.curent_epsilon = None
            self.scheduled_episodes = self.config.epsilon.linear_decay_params.scheduled_episodes
        elif self.config.epsilon.decay_type == "exponential":
            self.eps_max = self.epsilon.exponential_decay_params.max
            self.eps_min = self.epsilon.exponential_decay_params.min
            self.curent_epsilon = None
            self.epsilon_exponential_decay_factor = self.config.epsilon.exponential_decay_params.decay_factor
        else:
            raise ValueError("Unknown value for epsilon decay. Please select between none, linear, or exponential.")
            
        self.epsilon_decay_type = self.config.epsilon.decay_type
            
        if self.config.target_network_update.rule == "hard":
            if self.config.target_network_update.alpha != 0.0:
                warnings.warn("For hard update, alpha should be set to 0.0 ... proceeding with alpha = 0.0")
            self.update_alpha = 0.0
        elif self.config.target_network_update.rule == "soft":
            self.update_alpha = self.config.target_network_update.alpha

        self.update_target_network_every_n_steps = self.config.target_network_update.every_n_steps
        self.steps_since_last_network_update = 0

        self.current_episode = 0
        self.learning_starts_after = self.batch_size*2

    def _decay_epsilon(self):
        if self.epsilon_decay_type == "none":
            return
        elif self.epsilon_decay_type == "linear":
            self.__decay_epsilon_linearly()
        elif self.epsilon_decay_type == "exponential":
            self.__decay_epsilon_exponentially()

    def __decay_epsilon_linearly(self):
        fraction = min(self.current_episode/self.scheduled_episodes, 1.0)
        self.current_epsilon = (self.eps_min-self.eps_max)*fraction + self.eps_max

    def __decay_epsilon_exponentially(self):
        self.current_epsilon = min(self.eps_min, self.current_epsilon*self.epsilon_exponential_decay_factor) 
        
    def _remember(self, *args):
        self.memory.push(*args)
        return
    
    def _epsilon_greedy_action_selection(self, obs):
        """Epsilon greedy action selection"""
        if torch.rand(1).item() > self.current_epsilon:
            with torch.no_grad():
                return torch.argmax(self.policy_net(torch.tensor(obs).to(torch.float).to(self.device))).item()
        else:
            return torch.randint(0,self.config.env.num_actions,(1,)).item() 
            
    def _greedy_action_selection(self,obs):
        return torch.argmax(self.policy_net(torch.tensor(obs).to(torch.float).to(self.device))).item()

    def _sample_experiences(self):
        experiences = self.memory.sample(self.batch_size)
        return Experiences(*zip(*experiences))

    def _get_target_batch(self, next_state_batch, reward_batch, terminated_batch):
        with torch.no_grad():
            max_action = torch.argmax(self.policy_net(next_state_batch), axis = 1).unsqueeze(1).to(self.device)
            target = reward_batch + self.discount_rate * torch.mul(self.target_net(next_state_batch).gather(1,max_action).squeeze(), ~terminated_batch)
        return target

    def _build_observation_tensor(self, state, features):
        state = np.array(state)
        state = torch.from_numpy(state)

        features = np.array(features)
        features = torch.from_numpy(features)
        
        return torch.cat((state, features), dim = 1)
        
    def _train_one_batch(self):
        experiences = self._sample_experiences()

        next_state_batch = self._build_observation_tensor(experiences.next_state_batch, experiences.goal_batch)

        next_state_batch = next_state_batch.to(torch.float).to(self.device) 
        reward_batch = torch.tensor(experiences.reward_batch).to(self.device)
        terminated_batch = torch.tensor(experiences.terminated_batch).to(self.device)

        target_batch = self._get_target_batch(next_state_batch, reward_batch, terminated_batch)
    
        del next_state_batch
        del reward_batch
        del terminated_batch
        
        state_batch = self._build_observation_tensor(experiences.state_batch, experiences.goal_batch)
        state_batch = state_batch.to(torch.float).to(self.device)

        action_batch = torch.tensor(experiences.action_batch).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(target_batch, self.policy_net(state_batch).gather(1, action_batch).squeeze())
        
        loss.backward()
        self.optimizer.step()
        
        del state_batch
        del action_batch

        return loss.item()

    def _update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.update_alpha + target_net_state_dict[key]*(1-self.update_alpha)

        self.target_net.load_state_dict(target_net_state_dict)

    def start_episode(self, episode):
        self.current_episode = episode
        self._decay_epsilon()
        if self.config.log.epsilon_per_episode:
            log.add_value("agent_epsilon_per_episode", self.current_epsilon)

    def choose_action(self, obs, purpose):
        if purpose == "training":
            return self._epsilon_greedy_action_selection(obs)
        elif purpose == "inference":
            return self._greedy_action_selection(obs)
        else:
            raise ValueError("Unknown purpose. Choose either inference or training.")

    def train(self, transition, step):

        self._remember(*transition)
        
        if len(self.memory) < self.learning_starts_after:
            return

        if self.steps_since_last_training >= self.train_every_n_steps:
            self.steps_since_last_training = 0

            losses = []
            for _ in range(self.config.train_for_n_iterations):
                loss = self._train_one_batch()
                losses.append(loss)

            if self.config.log.loss_per_step:
                log.add_value("agent_loss", np.mean(losses))
                log.add_value("agent_loss_step", step)
        else:
            self.steps_since_last_training += 1
        
        if self.steps_since_last_network_update >= self.update_target_network_every_n_steps:
            self.steps_since_last_network_update = 0

            self._update_target_network()
        else:
            self.steps_since_last_network_update += 1

if __name__ == '__main__':
    pass