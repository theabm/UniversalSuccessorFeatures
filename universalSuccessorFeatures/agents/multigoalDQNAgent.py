import numpy as np
import torch
import exputils as eu
import exputils.data.logging as log
import warnings
import copy
from collections import namedtuple
import universalSuccessorFeatures.memory as mem
import universalSuccessorFeatures.networks.multigoalDQN as mdqn
import universalSuccessorFeatures.envs.gridWorld as envs

Experiences = namedtuple("Experiences", ("goal_batch", "state_batch", "action_batch", "reward_batch", "next_state_batch", "terminated_batch", "truncated_batch"))


class MultigoalDQNAgent():

    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            device = "cuda", # "cuda" or "cpu"
            discount_factor = 0.99,
            batch_size = 32,
            learning_rate = 5e-4,
            train_for_n_iterations = 1,
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
                rule = "hard",  # "hard" or "soft"
                every_n_steps = 10, 
                alpha = 0.0,  # target network params will be updated as P_t = alpha * P_t + (1-alpha) * P_p   where P_p are params of policy network
            ),
            memory = eu.AttrDict(
                cls = mem.ExperienceReplayMemory,
            ),
            #With this implementation, the choice of network completely determine the input size (i.e. the state and any additional info)
            #and the output size (num actions)
            network = eu.AttrDict(
                cls = mdqn.StateGoalPaperDQN,
                optimizer = torch.optim.Adam,
                loss = torch.nn.MSELoss,
            ),
            log = eu.AttrDict(
                loss_per_step = True,
                epsilon_per_episode = True,
            ),
            save = eu.AttrDict(
                # NEED TO HANDLE STILL
                directory = None,
                frequency = 10, # in episodes
            ),
        )
        return cnf


    def __init__(self, config = None, **kwargs):
        
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        #Setting the device
        if self.config.device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                warnings.warn('Cuda not available. Using CPU as device ...')
        else:
            self.device = torch.device("cpu")
        
        #Creating object instances
        if isinstance(self.config.network, dict):
            self.policy_net = eu.misc.create_object_from_config(self.config.network)
        else:
            raise ValueError("Network Config must be a dictionary.")

        if isinstance(self.config.memory, dict):
            self.memory = eu.misc.create_object_from_config(self.config.memory)
        else:
            raise ValueError("Memory config must be a dictionary.")

        #Setting other attributes

        self.target_net = copy.deepcopy(self.policy_net)

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        self.loss = self.config.network.loss()
        self.optimizer = self.config.network.optimizer(self.policy_net.parameters(), lr = self.config.learning_rate)
        
        self.batch_size = self.config.batch_size      
        self.train_every_n_steps = self.config.train_every_n_steps
        self.steps_since_last_training = 0

        self.discount_factor = self.config.discount_factor

        self.epsilon_decay_type = self.config.epsilon.decay_type

        if self.epsilon_decay_type == "none":
            self.current_epsilon = self.config.epsilon.no_decay_params.value
        elif self.epsilon_decay_type == "linear":
            self.eps_max = self.epsilon.linear_decay_params.max
            self.eps_min = self.epsilon.linear_decay_params.min
            self.curent_epsilon = self.eps_max
            self.scheduled_episodes = self.config.epsilon.linear_decay_params.scheduled_episodes
        elif self.epsilon_decay_type == "exponential":
            self.eps_max = self.epsilon.exponential_decay_params.max
            self.eps_min = self.epsilon.exponential_decay_params.min
            self.curent_epsilon = self.eps_max
            self.epsilon_exponential_decay_factor = self.config.epsilon.exponential_decay_params.decay_factor
        else:
            raise ValueError("Unknown value for epsilon decay. Please select between none, linear, or exponential.")
            
        if self.config.target_network_update.rule == "hard":
            if self.config.target_network_update.alpha != 0.0:
                warnings.warn("For hard update, alpha should be set to 0.0 ... proceeding with alpha = 0.0")
            self.update_alpha = 0.0
        elif self.config.target_network_update.rule == "soft":
            self.update_alpha = self.config.target_network_update.alpha
        else:
            raise ValueError("Unknown type of update rule.")

        self.update_target_network_every_n_steps = self.config.target_network_update.every_n_steps
        self.steps_since_last_network_update = 0

        self.current_episode = 0
        self.learning_starts_after = self.batch_size*2

    def start_episode(self, episode):
        self.current_episode = episode
        if self.config.log.epsilon_per_episode:
            log.add_value("agent_epsilon_per_episode", self.current_epsilon)

    def end_episode(self):
        self._decay_epsilon()

    def _decay_epsilon(self):
        if self.epsilon_decay_type == "none":
            return
        elif self.epsilon_decay_type == "linear":
            self.__decay_epsilon_linearly()
        elif self.epsilon_decay_type == "exponential":
            self.__decay_epsilon_exponentially()

    def __decay_epsilon_linearly(self):
        m = (self.eps_min-self.eps_max)/self.scheduled_episodes * (self.current_episode<=self.scheduled_episodes)
        self.curent_epsilon += m

    def __decay_epsilon_exponentially(self):
        self.current_epsilon = min(self.eps_min, self.current_epsilon*self.epsilon_exponential_decay_factor) 
    
    def choose_action(self, obs, goal, purpose):
        if purpose == "training":
            return self._epsilon_greedy_action_selection(obs, goal)
        elif purpose == "testing":
            return self._greedy_action_selection(obs, goal)
        else:
            raise ValueError("Unknown purpose. Choose either training or testing.")

    def _epsilon_greedy_action_selection(self, obs, goal):
        """Epsilon greedy action selection"""
        if torch.rand(1).item() > self.current_epsilon:
            return self._greedy_action_selection(obs, goal)
        else:
            return torch.randint(0,self.config.env.num_actions,(1,)).item() 

    def _greedy_action_selection(self,obs,goal):
            with torch.no_grad():
                return torch.argmax(
                                    self.policy_net(
                                                    self._make_compatible_with_nn(obs).to(self.device), 
                                                    self._make_compatible_with_nn(goal).to(self.device)
                                                    )
                                    ).item()

    def _make_compatible_with_nn(self, obs):
        return torch.tensor(obs).unsqueeze(0).to(torch.float)

    def _train_one_batch(self):
        experiences = self._sample_experiences()

        next_state_batch = self._build_observation_tensor(experiences.next_state_batch)
        goal_batch = self._build_observation_tensor(experiences.goal_batch)

        next_state_batch = next_state_batch.to(torch.float).to(self.device) 
        goal_batch = goal_batch.to(torch.float).to(self.device) 

        reward_batch = torch.tensor(experiences.reward_batch).to(self.device)
        terminated_batch = torch.tensor(experiences.terminated_batch).to(self.device)

        target_batch = self._get_target_batch(next_state_batch, goal_batch, reward_batch, terminated_batch)
    
        del next_state_batch
        del reward_batch
        del terminated_batch
        
        state_batch = self._build_observation_tensor(experiences.state_batch)
        state_batch = state_batch.to(torch.float).to(self.device)

        action_batch = torch.tensor(experiences.action_batch).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(target_batch, self.policy_net(state_batch, goal_batch).gather(1, action_batch).squeeze())
        
        loss.backward()
        self.optimizer.step()
        
        del state_batch
        del action_batch

        return loss.item()

    def _sample_experiences(self):
        experiences = self.memory.sample(self.batch_size)
        return Experiences(*zip(*experiences))

    def _build_observation_tensor(self, state):
        state = np.array(state)
        state = torch.from_numpy(state)
        return state

    def _get_target_batch(self, next_state_batch, goal_batch, reward_batch, terminated_batch):
        with torch.no_grad():
            max_action = torch.argmax(self.policy_net(next_state_batch, goal_batch), axis = 1).unsqueeze(1).to(self.device)
            target = reward_batch + self.discount_factor * torch.mul(self.target_net(next_state_batch, goal_batch).gather(1,max_action).squeeze(), ~terminated_batch)
        return target

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

    def _remember(self, *args):
        self.memory.push(*args)
        return

    def _update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.update_alpha + target_net_state_dict[key]*(1-self.update_alpha)

        self.target_net.load_state_dict(target_net_state_dict)


if __name__ == '__main__':
    my_dqn = MultigoalDQNAgent()