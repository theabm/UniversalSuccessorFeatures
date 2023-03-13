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


Experiences = namedtuple("Experiences", ("agent_position_batch", "goal_batch", "action_batch", "reward_batch", "next_agent_position_batch", "terminated_batch", "truncated_batch"))

class StateGoalAgent():

    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            device = "cuda", # "cuda" or "cpu"
            discount_factor = 0.99,
            batch_size = 32,
            learning_rate = 5e-4,
            train_for_n_iterations = 1,
            train_every_n_steps = 1,
            is_a_usf = False,
            loss_weight = 0.01,
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
                cls = nn.StateGoalPaperDQN,
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
        
        self.config = eu.combine_dicts(kwargs, config, StateGoalAgent.default_config())
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
        
        #  Creating object instances
        ## in the future, maybe I will have to incorporate some info from environment into config of network
        ## i.e the state size or goal size. But for now it works. (see what I did for feature based agents)
        if isinstance(self.config.network, dict):
            self.config.network.state_size = self.position_size
            self.config.network.goal_size = self.position_size
            self.config.network.features_size = self.features_size
            self.config.network.num_actions = self.action_space
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


        # Setting other attributes
        self.target_net = copy.deepcopy(self.policy_net)

        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        self.loss = self.config.network.loss()
        self.loss_weight = self.config.loss_weight
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

        self.is_a_usf = self.config.is_a_usf

    def start_episode(self, episode):
        self.current_episode = episode
        if self.config.log.epsilon_per_episode:
            log.add_value("agent_epsilon_per_episode", self.epsilon.value)

    def end_episode(self):
        self.epsilon.decay()

    def choose_action(self, agent_position, goal_position, training = True):
        if training:
            return self._epsilon_greedy_action_selection(agent_position, goal_position).item()
        else:
            return self._greedy_action_selection(agent_position, goal_position).item()

    def _epsilon_greedy_action_selection(self, agent_position, goal_position):
        """Epsilon greedy action selection"""
        if torch.rand(1).item() > self.epsilon.value:
            return self._greedy_action_selection(agent_position, goal_position)
        else:
            return torch.randint(0,self.action_space,(1,)) 

    def _greedy_action_selection(self, agent_position, goal_position):
        with torch.no_grad():
            return torch.argmax(
                self.policy_net(
                    torch.tensor(agent_position).to(torch.float).to(self.device),
                    torch.tensor(goal_position).to(torch.float).to(self.device)
                )
            )

    def _train_one_batch(self):
        experiences = self._sample_experiences()
        goal_batch = self._build_tensor_from_batch_of_np_arrays(experiences.goal_batch).to(self.device)

        target_batch = self._build_target_batch(experiences, goal_batch)
        predicted_batch = self._build_predicted_batch(experiences, goal_batch)

        self.optimizer.zero_grad()

        loss = self.loss(target_batch, predicted_batch)
        if self.is_a_usf:
            sf_target_batch = self._build_sf_target_batch(experiences, goal_batch)
            sf_predicted_batch = self._build_sf_predicted_batch(experiences, goal_batch)
            loss += self.loss_weight * self.loss(sf_target_batch, sf_predicted_batch)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def _build_sf_target_batch(self, experiences, goal_batch):
        next_agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.next_agent_position_batch).to(self.device) # shape (batch_size, n)
        with torch.no_grad():
            reward_batch = self.target_net.layer_state(next_agent_position_batch).to(self.device) # shape (batch_size, n)

        # Needs to be shape (batch_size, 1) so that broadcasting duplicates to (batch_size, n). Without this, torch.mul later will not work.
        # Also, (1, batch_size) is wrong because it will not broadcast in the correct way - it will broadcast to be (batch_size, batch_size) which will give error for torch.mul. 
        terminated_batch = torch.tensor(experiences.terminated_batch).unsqueeze(1).to(self.device) # shape (batch_size,1)

        target_batch = self._get_sf_target_batch(next_agent_position_batch, goal_batch, reward_batch, terminated_batch)
    
        del next_agent_position_batch
        del reward_batch
        del terminated_batch

        return target_batch

    def _get_sf_target_batch(self, next_agent_position_batch, goal_batch, reward_batch, terminated_batch):
        with torch.no_grad():
            # incomplete forward outputs only successor features of shape (batch_size, num_actions, features_size) and weights of shape (batch_size, features_size)
            sf_s_g, w = self.target_net.incomplete_forward(next_agent_position_batch, goal_batch) # shape (batch_size, num_actions, n) = (32, 4, 100)
            # complete forward takes sf_s_g and w, and multiplies them in the correct way to ouput the Q values of shape (batch_size, num_actions)
            q = self.target_net.complete_forward(sf_s_g, w)
            # we now take the max of the q function. Axis = 1 makes it so we collapse columns dimension to 1 - this results in shape (batch,). 
            _, action = torch.max(q, axis = 1)
            # Then, we reshape to (batch, 1, 1) and tile (duplicate) features_size times along the last dim. This is needed for the gather function to work later.
            action = action.reshape(self.batch_size, 1, 1).tile(self.features_size).to(self.device) # shape (batch_size,1,n)
            # The reward batch will have shape (batch, features_size), sf_s_g will have size (batch, num_actions, features_size), terminated will have (batch, 1)
            # The gather function is a fancy slicing operation. We use it to select the sf's corresponding to the action index we found before. i.e, if for batch 1 the 
            # max action is action 2, then we will pick the second (features_size,) matrix in that batch. 
            # The gather operator requires that sf_s_g and action be the same shape (so 3D) and for it to work correctly as we want it, we must repeat the action index
            # everywhere in the last dim, hence the tiling operation. This will produce (batch, 1, features_size) which we squeeze() to (batch, features_size) and then multiply
            # with terminated_batch which is (batch, 1) (which will get brodcasted). Final result is (batch, features_size). So it all works.
            target = reward_batch + self.discount_factor * torch.mul(sf_s_g.gather(1, action).squeeze(), ~terminated_batch)
        return target

    def _build_sf_predicted_batch(self, experiences, goal_batch):
        agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.agent_position_batch).to(self.device)
        action_batch = torch.tensor(experiences.action_batch).reshape(self.batch_size, 1, 1).tile(self.features_size).to(self.device)
        sf_s_g, _ = self.policy_net.incomplete_forward(agent_position_batch, goal_batch)
        predicted_batch = sf_s_g.gather(1, action_batch).squeeze() 

        del agent_position_batch
        del action_batch
        del sf_s_g

        return predicted_batch

    def _sample_experiences(self):
        experiences = self.memory.sample(self.batch_size)
        return Experiences(*zip(*experiences))

    def _build_tensor_from_batch_of_np_arrays(self, batch_of_np_arrays):
        # expected shape: [(1,n), (1,n), ..., (1,n)] where in total we have batch_size elements
        batch_of_np_arrays = np.array(batch_of_np_arrays)
        # batch of np_arrays has form (batch_size, 1, n) so after squeeze() we have (batch_size, n)
        batch_of_np_arrays = torch.tensor(batch_of_np_arrays).squeeze().to(torch.float)

        return batch_of_np_arrays

    def _build_target_batch(self, experiences, goal_batch):
        next_agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.next_agent_position_batch).to(self.device) # shape (batch_size, n)

        # reward and terminated batch are handled differently because they are a list of floats and bools respectively and not a list of np.arrays
        reward_batch = torch.tensor(experiences.reward_batch).to(torch.float).to(self.device) # shape (n)
        terminated_batch = torch.tensor(experiences.terminated_batch).to(self.device) # shape (n)

        target_batch = self._get_dql_target_batch(next_agent_position_batch, goal_batch, reward_batch, terminated_batch)
    
        del next_agent_position_batch
        del reward_batch
        del terminated_batch

        return target_batch

    def _build_predicted_batch(self, experiences, goal_batch):
        agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.agent_position_batch).to(self.device)
        action_batch = torch.tensor(experiences.action_batch).unsqueeze(1).to(self.device)
        predicted_batch = self.policy_net(agent_position_batch, goal_batch).gather(1, action_batch).squeeze()

        del agent_position_batch
        del action_batch

        return predicted_batch

    def _get_dql_target_batch(self, next_agent_position_batch, goal_batch, reward_batch, terminated_batch):
        with torch.no_grad():
            q, _ = torch.max(self.target_net(next_agent_position_batch, goal_batch), axis = 1)  # shape of q is (batch_size,)
            q.to(self.device)
            target = reward_batch + self.discount_factor * torch.mul(q, ~terminated_batch)
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
    my_dqn = StateGoalAgent()
