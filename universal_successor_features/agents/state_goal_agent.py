import numpy as np
import torch
import exputils as eu
import exputils.data.logging as log
import warnings
import copy
from collections import namedtuple
import universal_successor_features.memory as mem
import universal_successor_features.networks as nn
import universal_successor_features.epsilon as eps


Experiences = namedtuple("Experiences", ("agent_position_batch",
                                         "goal_batch",
                                         "action_batch",
                                         "reward_batch",
                                         "next_agent_position_batch",
                                         "terminated_batch",
                                         "truncated_batch"
                                         )
                         )

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
                loss_weight_psi = 0.01,
                loss_weight_phi = 0.00,
                network = eu.AttrDict(
                    cls = nn.StateGoalPaperDQN,
                    optimizer = torch.optim.Adam,
                    ),
                target_network_update = eu.AttrDict(
                    rule = "hard",  # "hard" or "soft"
                    every_n_steps = 10, 
                    alpha = 0.0,  # target network params will be updated as P_t = alpha * P_t + (1-alpha) * P_p   where P_p are params of policy network
                    ),
                epsilon = eu.AttrDict(
                    cls = eps.EpsilonConstant, 
                    ),
                memory = eu.AttrDict(
                    cls = mem.ExperienceReplayMemory,
                    # Need to be defined for prioritized experience replay
                    alpha = None,
                    beta0 = None,
                    schedule_length = None,
                    ),
                log = eu.AttrDict(
                    loss_per_step = True,
                    epsilon_per_episode = True,
                    log_name_epsilon = "epsilon_per_episode",
                    log_name_loss = "loss_per_step",
                    ),
                save = eu.AttrDict(
                    extension = ".pt"
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

        # Creating object instances
        if isinstance(self.config.network, dict):
            self.config.network.state_size = self.position_size
            self.config.network.goal_size = self.position_size
            self.config.network.features_size = self.features_size
            self.config.network.num_actions = self.action_space

            self.policy_net = eu.misc.create_object_from_config(self.config.network)
            self.is_a_usf = self.policy_net.is_a_usf 
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

        self.loss_weight_psi = self.config.loss_weight_psi
        self.loss_weight_phi = self.config.loss_weight_phi
        self.optimizer = self.config.network.optimizer(self.policy_net.parameters(), lr = self.config.learning_rate)

        self.batch_size = self.config.batch_size      
        self.train_every_n_steps = self.config.train_every_n_steps - 1
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

        self.update_target_network_every_n_steps = self.config.target_network_update.every_n_steps - 1
        self.steps_since_last_network_update = 0

        self.current_episode = 0
        self.step = 0
        self.learning_starts_after = self.batch_size*2

    def start_episode(self, episode):
        self.current_episode = episode
        if self.config.log.epsilon_per_episode:
            log.add_value(self.config.log.log_name_epsilon, self.epsilon.value)

    def end_episode(self):
        self.epsilon.decay()

    def choose_action(self,
                      agent_position,
                      list_of_goal_positions,
                      env_goal_position,
                      training
                      ):
        if training:
            return self._epsilon_greedy_action_selection(
                    agent_position,
                    list_of_goal_positions,
                    env_goal_position,
                    ).item()
        else:
            return self._greedy_action_selection(
                    agent_position,
                    list_of_goal_positions,
                    env_goal_position,
                    ).item()

    def _epsilon_greedy_action_selection(self,
                                         agent_position,
                                         list_of_goal_positions,
                                         env_goal_position
                                         ):
        """Epsilon greedy action selection"""
        if torch.rand(1).item() > self.epsilon.value:
            return self._greedy_action_selection(
                    agent_position,
                    list_of_goal_positions,
                    env_goal_position,
                    )
        else:
            return torch.randint(0,self.action_space,(1,)) 

    def _greedy_action_selection(self,
                                 agent_position,
                                 list_of_goal_positions,
                                 env_goal_position
                                 ):
        q_per_goal = torch.zeros(len(list_of_goal_positions))
        a_per_goal = torch.zeros(len(list_of_goal_positions), dtype=int)

        for i, goal_position in enumerate(list_of_goal_positions):
            with torch.no_grad():
                q, *_ = self.policy_net(
                        agent_position = torch.tensor(agent_position).to(torch.float).to(self.device),
                        policy_goal_position = torch.tensor(goal_position).to(torch.float).to(self.device),
                        env_goal_position = torch.tensor(env_goal_position).to(torch.float).to(self.device),
                        )
                qm, am = torch.max(q, axis = 1)
                q_per_goal[i] = qm.item()
                a_per_goal[i] = am.item() 
        # batch together for gpu in the future
        amm = torch.argmax(q_per_goal)

        return a_per_goal[amm.item()]

    def _sample_experiences(self):
        experiences, weights = self.memory.sample(self.batch_size)
        return Experiences(*zip(*experiences)), torch.tensor(weights)

    def _build_tensor_from_batch_of_np_arrays(self, batch_of_np_arrays):
        # expected shape: [(1,n), (1,n), ..., (1,n)] where in total we have batch_size elements
        batch_of_np_arrays = np.array(batch_of_np_arrays)
        # batch of np_arrays has form (batch_size, 1, n) so after squeeze() we have (batch_size, n)
        batch_of_np_arrays = torch.tensor(batch_of_np_arrays).squeeze().to(torch.float)

        return batch_of_np_arrays

    def _train_one_batch(self):
        experiences, sample_weights = self._sample_experiences()
        goal_batch = self._build_tensor_from_batch_of_np_arrays(experiences.goal_batch).to(self.device)
        sample_weights = sample_weights.to(self.device)

        self.optimizer.zero_grad()
        if self.is_a_usf:
            target_batch_q, target_batch_psi, r = self._build_target_batch(
                    experiences,
                    goal_batch,
                    )
            predicted_batch_q, predicted_batch_psi, phi_w = self._build_predicted_batch(
                    experiences,
                    goal_batch,
                    )

            td_error_q = torch.square(torch.abs(target_batch_q - predicted_batch_q)) # shape (batch_size,)
            # shape of target_batch_psi is (batch, size_features) so the td_error for that batch must be summed along first dim
            # which automatically squeezed dim = 1 and so the final shape is (batch,)
            td_error_psi = torch.mean(torch.square(torch.abs(target_batch_psi - predicted_batch_psi)), dim = 1) # shape (batch_size,)

            td_error_phi = torch.square(torch.abs(r-phi_w)) # shape (batch_size, )

            total_td_error = (td_error_q + self.loss_weight_psi*td_error_psi + self.loss_weight_phi*td_error_phi)

            # update the priority of batch samples in memory
            self.memory.update_samples(total_td_error.detach().cpu())

            self.memory.anneal_beta()

            loss = torch.mean(sample_weights*total_td_error)
        else:
            target_batch = self._build_target_batch(
                    experiences,
                    goal_batch,
                    )
            predicted_batch = self._build_predicted_batch(
                    experiences,
                    goal_batch,
                    )

            td_error_q = torch.square(torch.abs(target_batch - predicted_batch))

            self.memory.update_samples(td_error_q.detach().cpu())

            self.memory.anneal_beta()

            loss = torch.mean(sample_weights*td_error_q)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _build_target_batch(self,
                            experiences,
                            goal_batch
                            ):
        next_agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.next_agent_position_batch).to(self.device) # shape (batch_size, n)

        # reward and terminated batch are handled differently because they are a list of floats and bools respectively and not a list of np.arrays
        reward_batch = torch.tensor(experiences.reward_batch).to(torch.float).to(self.device) # shape (batch_size,)
        terminated_batch = torch.tensor(experiences.terminated_batch).to(self.device) # shape (batch_size,)

        if self.is_a_usf:
            with torch.no_grad():

                q, sf_s_g, w, reward_phi_batch = self.target_net(
                        agent_position = next_agent_position_batch,
                        policy_goal_position = goal_batch,
                        env_goal_position = goal_batch,
                        )

                qm, action = torch.max(q, axis = 1)

                target_q = reward_batch + self.discount_factor * torch.mul(qm, ~terminated_batch) # shape (batch_size,)

                terminated_batch = terminated_batch.unsqueeze(1)
                action = action.reshape(self.batch_size, 1, 1).tile(self.features_size).to(self.device) # shape (batch_size,1,n)

                target_psi = reward_phi_batch + self.discount_factor * torch.mul(sf_s_g.gather(1, action).squeeze(), ~terminated_batch) # shape (batch, features_size)

            return target_q, target_psi, reward_batch

        else:
            with torch.no_grad():
                q, *_ = self.target_net(
                        agent_position = next_agent_position_batch,
                        policy_goal_position = goal_batch,
                        env_goal_position = goal_batch,
                        )
                q, _ = torch.max(q, axis = 1)
                target_q = reward_batch + self.discount_factor * torch.mul(q, ~terminated_batch)

            return target_q 

    def _build_predicted_batch(self,
                               experiences,
                               goal_batch
                               ):
        agent_position_batch = self._build_tensor_from_batch_of_np_arrays(experiences.agent_position_batch).to(self.device)
        action_batch = torch.tensor(experiences.action_batch).unsqueeze(1).to(self.device)

        if self.is_a_usf:
            q, sf_s_g, w, phi = self.policy_net(
                    agent_position = agent_position_batch,
                    policy_goal_position = goal_batch,
                    env_goal_position = goal_batch,
                    )

            predicted_q = q.gather(1,action_batch).squeeze() # shape (batch_size,)

            action_batch = action_batch.reshape(self.batch_size, 1, 1).tile(self.features_size)
            predicted_psi = sf_s_g.gather(1, action_batch).squeeze() # shape (batch_size, features_size)

            return predicted_q, predicted_psi, torch.sum(phi * w, dim = 1)

        else:
            predicted_q, *_ = self.policy_net(
                    agent_position = agent_position_batch,
                    policy_goal_position = goal_batch,
                    env_goal_position = goal_batch
                    )

            return predicted_q.gather(1, action_batch).squeeze()

    def train(self, transition):

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
                log.add_value(self.config.log.log_name_loss, np.mean(losses))
        else:
            self.steps_since_last_training += 1

        if self.steps_since_last_network_update >= self.update_target_network_every_n_steps:
            self.steps_since_last_network_update = 0

            self._update_target_network()
        else:
            self.steps_since_last_network_update += 1

    def prepare_for_eval_phase(self):
        self.train_memory_buffer = copy.deepcopy(self.memory)

        self.eval_memory_buffer = eu.misc.create_object_from_config(self.config.memory)

    def train_during_eval_phase(self, transition, p_pick_new_memory_buffer):

        self.eval_memory_buffer.push(transition)

        if len(self.eval_memory_buffer) < self.learning_starts_after:
            self.memory = self.train_memory_buffer
        else:
            if torch.rand(1).item() <= p_pick_new_memory_buffer:
                self.memory = self.eval_memory_buffer
            else:
                self.memory = self.train_memory_buffer

        if self.steps_since_last_training >= self.train_every_n_steps:
            self.steps_since_last_training = 0

            losses = []
            for _ in range(self.config.train_for_n_iterations):
                loss = self._train_one_batch()
                losses.append(loss)

            if self.config.log.loss_per_step:
                log.add_value(self.config.log.log_name_loss, np.mean(losses))
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

    def save(self, episode, step, total_reward):
        filename = "checkpoint" + self.config.save.extension
        torch.save(
                {
                    "config": self.config,
                    "episode": episode,
                    "step": step,
                    "total_reward": total_reward,
                    "model_state_dict": self.policy_net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "memory": self.memory,
                    },
                filename
                )

    @classmethod
    def load_from_checkpoint(cls, env, filename):
        checkpoint = torch.load(filename)

        agent = cls(env, config = checkpoint["config"])

        agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
        agent.target_net = copy.deepcopy(agent.policy_net)

        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        agent.memory = checkpoint["memory"]
        agent.current_episode = checkpoint["episode"]
        agent.total_reward = checkpoint["total_reward"]

        return agent












