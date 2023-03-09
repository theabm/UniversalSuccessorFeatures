import numpy as np
import torch
import exputils as eu
import exputils.data.logging as log
import warnings
import copy
from collections import namedtuple
import universal_successor_features.memory as mem
import universal_successor_features.networks as nn
import universal_successor_features.envs as envs
import universal_successor_features.epsilon as eps


Experiences = namedtuple("Experiences", ("agent_position_batch", "goal_batch", "goal_weights_batch", "action_batch", "reward_batch", "next_agent_position_batch", "terminated_batch", "truncated_batch"))

class StateGoalWeightAgent():

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
                cls = eps.EpsilonConstant, 
            ),
            target_network_update = eu.AttrDict(
                rule = "hard",  # "hard" or "soft"
                every_n_steps = 10, 
                alpha = 0.0,  # target network params will be updated as P_t = alpha * P_t + (1-alpha) * P_p   where P_p are params of policy network
            ),
            memory = eu.AttrDict(
                cls = mem.ExperienceReplayMemory,
                #here i can add any other variable from the defaul config of the experience replay class.
            ),
            #With this implementation, the choice of network completely determine the input size (i.e. the state and any additional info)
            #and the output size (num actions)
            network = eu.AttrDict(
                cls = nn.StateGoalWeightUSF,
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


    def __init__(self, env, config = None, **kwargs):
        
        self.config = eu.combine_dicts(kwargs, config, StateGoalWeightAgent.default_config())
        self.action_space = env.action_space.n

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
            self.config.network.features_size = env.rows * env.columns
            self.policy_net = eu.misc.create_object_from_config(self.config.network)
        else:
            raise ValueError("Network Config must be a dictionary.")

        if isinstance(self.config.memory, dict):
            self.memory = eu.misc.create_object_from_config(self.config.memory)
        else:
            raise ValueError("Memory config must be a dictionary.")

        if isinstance(self.config.epsilon, dict):
            self.epsilon = eu.misc.create_object_from_config(self.config.epsilon)
        else:
            raise ValueError("Network Config must be a dictionary.")


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
            log.add_value("agent_epsilon_per_episode", self.epsilon.value)

    def end_episode(self):
        self.epsilon.decay()

    def choose_action(self, agent_position, goal_position, goal_weights, training = True):
        if training:
            return self._epsilon_greedy_action_selection(agent_position, goal_position, goal_weights).item()
        else:
            return self._greedy_action_selection(agent_position, goal_position, goal_weights).item()

    def _epsilon_greedy_action_selection(self, agent_position, goal_position, goal_weights):
        """Epsilon greedy action selection"""
        if torch.rand(1).item() > self.epsilon.value:
            return self._greedy_action_selection(agent_position, goal_position, goal_weights)
        else:
            return torch.randint(0,self.action_space,(1,)) 

    def _greedy_action_selection(self, agent_position, goal_position, goal_weights):
        with torch.no_grad():
            return torch.argmax(
                self.policy_net(
                    agent_position = torch.tensor(agent_position).to(torch.float).to(self.device),
                    goal_position = torch.tensor(goal_position).to(torch.float).to(self.device),
                    goal_weights = torch.tensor(goal_weights).to(torch.float).to(self.device)
                )
            )

    def _train_one_batch(self):
        experiences = self._sample_experiences()
        goal_batch = self._build_tensor_from_batch_of_np_arrays(experiences.goal_batch).to(self.device)
        goal_weights_batch = self._build_tensor_from_batch_of_np_arrays(experiences.goal_weights_batch).to(self.device)

        target_batch = self.__build_target_batch(experiences, goal_batch, goal_weights_batch)
        predicted_batch = self.__build_predicted_batch(experiences, goal_batch, goal_weights_batch)

        self.optimizer.zero_grad()
        loss = self.loss(target_batch, predicted_batch)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def _sample_experiences(self):
        experiences = self.memory.sample(self.batch_size)
        return Experiences(*zip(*experiences))

    def _build_tensor_from_batch_of_np_arrays(self, batch_of_np_arrays):
        # expected shape: [(1,n), (1,n), ..., (1,n)] where in total we have batch_size elements
        batch_of_np_arrays = np.array(batch_of_np_arrays)
        # batch of np_arrays has form (batch_size, 1, n) so after squeeze() we have (batch_size, n)
        batch_of_np_arrays = torch.tensor(batch_of_np_arrays).squeeze().to(torch.float)

        return batch_of_np_arrays

    def __build_target_batch(self, experiences, goal_batch, goal_weights_batch):
        next_agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.next_agent_position_batch).to(self.device) # shape (batch_size, n)

        # reward and terminated batch are handled differently because they are a list of floats and bools respectively and not a list of np.arrays
        reward_batch = torch.tensor(experiences.reward_batch).to(torch.float).to(self.device) # shape (n)
        terminated_batch = torch.tensor(experiences.terminated_batch).to(self.device) # shape (n)

        target_batch = self._get_dql_target_batch(next_agent_position_batch, goal_batch, goal_weights_batch, reward_batch, terminated_batch)
    
        del next_agent_position_batch
        del reward_batch
        del terminated_batch

        return target_batch

    def __build_predicted_batch(self, experiences, goal_batch, goal_weights_batch):
        agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.agent_position_batch).to(self.device)
        action_batch = torch.tensor(experiences.action_batch).unsqueeze(1).to(self.device)
        predicted_batch = self.policy_net(agent_position = agent_position_batch, goal_position = goal_batch, goal_weights = goal_weights_batch).gather(1, action_batch).squeeze()

        del agent_position_batch
        del action_batch

        return predicted_batch

    def _get_dql_target_batch(self, next_agent_position_batch, goal_batch, goal_weights_batch, reward_batch, terminated_batch):
        with torch.no_grad():
            max_action = torch.argmax(self.target_net(agent_position = next_agent_position_batch, goal_position = goal_batch, goal_weights = goal_weights_batch), axis = 1).unsqueeze(1).to(self.device)
            target = reward_batch + self.discount_factor * torch.mul(self.target_net(agent_position = next_agent_position_batch, goal_position = goal_batch, goal_weights = goal_weights_batch).gather(1,max_action).squeeze(), ~terminated_batch)
        return target

    def train(self, transition, step):
        
        self.memory.push(transition)
        
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

    def _update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = target_net_state_dict[key]*self.update_alpha + policy_net_state_dict[key]*(1-self.update_alpha)

        self.target_net.load_state_dict(target_net_state_dict)


if __name__ == '__main__':
    my_dqn = StateGoalWeightAgent()
