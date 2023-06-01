import numpy as np
import torch
import exputils as eu
import exputils.data.logging as log
import warnings
import copy
from collections import namedtuple
from gradient_descent_the_ultimate_optimizer import gdtuo
import universal_successor_features.memory as mem
import universal_successor_features.epsilon as eps
from abc import ABC, abstractmethod


Experiences = namedtuple(
    "Experiences",
    (
        "agent_position_batch",
        "agent_position_features_batch",
        "goal_batch",
        "goal_weights_batch",
        "action_batch",
        "reward_batch",
        "next_agent_position_batch",
        "next_agent_position_features_batch",
        "terminated_batch",
        "truncated_batch",
    ),
)


class BaseAgent(ABC):
    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            # "cuda" or "cpu"
            device="cuda",
            discount_factor=0.99,
            batch_size=32,
            learning_rate=5e-4,
            train_for_n_iterations=1,
            train_every_n_steps=1,
            loss_weight_q=1.0,
            loss_weight_psi=0.01,
            loss_weight_phi=0.00,
            network=eu.AttrDict(
                cls=None,
                # whether to use hypergradients
                # setting to true will use SGD
                # to optimize ADAM
                use_gdtuo=None,
                optimizer=None,
            ),
            target_network_update=eu.AttrDict(
                rule="hard",  # "hard" or "soft"
                every_n_steps=10,
                alpha=0.0,  # target network params will be updated as
                # P_t = alpha * P_t + (1-alpha) * P_p
                # where P_p are params of policy network
            ),
            epsilon=eu.AttrDict(
                cls=eps.EpsilonConstant,
            ),
            memory=eu.AttrDict(
                cls=mem.ExperienceReplayMemory,
                # Need to be defined only for prioritized experience replay
                alpha=None,
                beta0=None,
                schedule_length=None,
            ),
            log=eu.AttrDict(
                loss_per_step=True,
                epsilon_per_episode=True,
                log_name_epsilon="epsilon_per_episode",
                log_name_loss="loss_per_step",
            ),
            save=eu.AttrDict(extension=".pt"),
        )
        return cnf

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, BaseAgent.default_config())
        self.env = env
        self.action_space = env.action_space.n
        self.position_size = env.observation_space["agent_position"].shape[1]
        self.features_size = env.observation_space["agent_position_features"].shape[1]

        # Setting the device
        if self.config.device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                warnings.warn("Cuda not available. Using CPU as device ...")
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

        self.loss_weight_q = self.config.loss_weight_q
        self.loss_weight_psi = self.config.loss_weight_psi
        self.loss_weight_phi = self.config.loss_weight_phi

        self.use_gdtuo = self.config.network.use_gdtuo
        if self.use_gdtuo:
            self.optimizer = gdtuo.ModuleWrapper(
                self.policy_net, optimizer=gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
            )
            self.optimizer.initialize()
        else:
            self.optimizer = self.config.network.optimizer(
                self.policy_net.parameters(), lr=self.config.learning_rate
            )

        self.batch_size = self.config.batch_size
        self.train_every_n_steps = self.config.train_every_n_steps - 1
        self.steps_since_last_training = 0

        self.discount_factor = self.config.discount_factor

        if self.config.target_network_update.rule == "hard":
            if self.config.target_network_update.alpha != 0.0:
                warnings.warn(
                    "For hard update, alpha should be set to 0.0 ... proceeding with alpha = 0.0"
                )
            self.update_alpha = 0.0
        elif self.config.target_network_update.rule == "soft":
            self.update_alpha = self.config.target_network_update.alpha
        else:
            raise ValueError("Unknown type of update rule.")

        self.update_target_network_every_n_steps = (
            self.config.target_network_update.every_n_steps - 1
        )
        self.steps_since_last_network_update = 0

        self.current_episode = 0
        self.step = 0
        self.learning_starts_after = self.batch_size * 2

    def start_episode(self, episode):
        self.current_episode = episode
        if self.config.log.epsilon_per_episode:
            log.add_value(self.config.log.log_name_epsilon, self.epsilon.value)

    def end_episode(self):
        self.epsilon.decay()

    def choose_action(
        self,
        obs,
        list_of_goal_positions,
        training,
    ):
        if training:
            return self._epsilon_greedy_action_selection(
                obs,
                list_of_goal_positions,
            ).item()
        else:
            return self._greedy_action_selection(
                obs,
                list_of_goal_positions,
            ).item()

    def _epsilon_greedy_action_selection(self, obs, list_of_goal_positions):
        """Epsilon greedy action selection"""
        if torch.rand(1).item() > self.epsilon.value:
            return self._greedy_action_selection(
                obs,
                list_of_goal_positions,
            )
        else:
            return torch.randint(0, self.action_space, (1,))

    def _greedy_action_selection(self, obs, list_of_goal_positions):
        q_per_goal = torch.zeros(len(list_of_goal_positions))
        a_per_goal = torch.zeros(len(list_of_goal_positions), dtype=int)

        obs_dict = self._build_arguments_from_obs(obs)
        for i, goal_position in enumerate(list_of_goal_positions):
            with torch.no_grad():
                q, *_ = self.policy_net(
                    policy_goal_position=torch.tensor(goal_position)
                    .to(torch.float)
                    .to(self.device),
                    **obs_dict,
                )
                qm, am = torch.max(q, axis=1)
                q_per_goal[i] = qm.item()
                a_per_goal[i] = am.item()
        # batch together for gpu in the future
        amm = torch.argmax(q_per_goal)

        return a_per_goal[amm.item()]

    @abstractmethod
    def _build_arguments_from_obs(self):
        pass

    def _print_successor_features(self, obs, list_of_goal_positions):
        if self.is_a_usf:
            obs_dict = self._build_arguments_from_obs(obs)
            for i, goal_position in enumerate(list_of_goal_positions):
                with torch.no_grad():
                    q, sf, *_ = self.policy_net(
                        policy_goal_position=torch.tensor(goal_position)
                        .to(torch.float)
                        .to(self.device),
                        **obs_dict,
                    )
                    sf = sf.squeeze().reshape(
                        self.action_space, self.env.rows, self.env.columns
                    )
                    print(
                        f"Sucessor features at: {obs['agent_position_features']}\nFor goal{goal_position}\n",
                        sf,
                    )
        else:
            raise "This function is only available for USF's"

    def _sample_experiences(self):
        experiences, weights = self.memory.sample(self.batch_size)
        return Experiences(*zip(*experiences)), torch.tensor(weights)

    @staticmethod
    def _build_tensor_from_batch_of_np_arrays(batch_of_np_arrays):
        # expected shape: [(1,n), (1,n), ..., (1,n)] where in total we
        # have batch_size elements
        batch_of_np_arrays = np.array(batch_of_np_arrays)
        # batch of np_arrays has form (batch_size, 1, n) so after squeeze()
        # we have (batch_size, n)
        batch_of_np_arrays = torch.tensor(batch_of_np_arrays).squeeze().to(torch.float)

        return batch_of_np_arrays

    def _build_dictionary_of_batch_from_experiences(self, experiences):
        batch_dict = {
            # shape (batch_size, position_size)
            "agent_position_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.agent_position_batch
            ).to(self.device),
            # shape (batch_size, feature_size)
            "agent_position_features_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.agent_position_features_batch
            ).to(self.device),
            # shape (batch_size, position_size)
            "goal_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.goal_batch
            ).to(self.device),
            # shape (batch_size, feature_size)
            "goal_weights_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.goal_weights_batch
            ).to(self.device),
            # shape (batch_size, 1)
            "action_batch": torch.tensor(experiences.action_batch)
            .unsqueeze(1)
            .to(self.device),
            # reward and terminated batch are handled differently because they are
            # a list of floats and bools respectively and not a list of np.arrays
            # shape (batch_size,1)
            "reward_batch": (
                torch.tensor(experiences.reward_batch).to(torch.float).to(self.device)
            ),
            # shape (batch_size, position_size)
            "next_agent_position_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.next_agent_position_batch
            ).to(self.device),
            # shape (batch_size, feature_size)
            "next_agent_position_features_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.next_agent_position_features_batch
            ).to(
                self.device
            ),
            # shape (batch_size,)
            "terminated_batch": torch.tensor(experiences.terminated_batch).to(
                self.device
            ),
        }

        return batch_dict

    @staticmethod
    @abstractmethod
    def _build_target_args(batch_args):
        """Extract the needed arguments for the network from the dictionary of batch
        arguments
        """
        pass

    def _build_q_target(self, batch_args):
        """Build the q function target from batch args.
        Input: batch_args (dict) is a dictionary of batches of experience.
        Output: target of the Q values.
        """
        # We only assert the data that we use for computations. Any other 
        # data not explicitly used in the body of this function, such as 
        # sf_s_g will be checked in the appropriate function that uses it.

        assert len(batch_args)==9

        q, sf_s_g, w, reward_phi_batch = self.target_net(
            **self._build_target_args(batch_args)
        )
        
        assert q.shape == (self.batch_size, self.action_space)

        q_max, action = torch.max(q, axis=1)

        assert q_max.shape == (self.batch_size,)
        assert action.shape == (self.batch_size,)
        assert batch_args["reward_batch"].shape == (self.batch_size,)
        assert batch_args["terminated_batch"].shape == (self.batch_size,)

        target_q = batch_args["reward_batch"] + self.discount_factor * torch.mul(
            q_max, ~batch_args["terminated_batch"]
        )
        assert target_q.shape == (self.batch_size,)

        return target_q, action, sf_s_g, w, reward_phi_batch

    def _build_psi_target(self, batch_args, action, sf_s_g, reward_phi_batch):
        assert batch_args["terminated_batch"].shape == (self.batch_size,)
        assert action.shape == (self.batch_size,)
        assert sf_s_g.shape == (self.batch_size, self.action_space, self.features_size)
        assert reward_phi_batch.shape == (self.batch_size, self.features_size)

        terminated_batch = batch_args["terminated_batch"].unsqueeze(1)

        assert terminated_batch.shape == (self.batch_size, 1)

        # shape (batch_size,1,n)
        action = (
            action.reshape(self.batch_size, 1, 1)
            .tile(self.features_size)
            .to(self.device)
        )

        assert action.shape == (self.batch_size, 1, self.features_size)

        target_psi = reward_phi_batch + self.discount_factor * torch.mul(
            sf_s_g.gather(1, action).squeeze(), ~terminated_batch
        )

        assert target_psi.shape == (self.batch_size, self.features_size)

        return target_psi

    def _build_target_batch(self, batch_args):
        if self.is_a_usf:
            with torch.no_grad():
                target_q, action, sf_s_g, _, reward_phi_batch = self._build_q_target(
                    batch_args
                )

                target_psi = self._build_psi_target(
                    batch_args, action, sf_s_g, reward_phi_batch
                )

                target_r = batch_args["reward_batch"]

            return target_q, target_psi, target_r

        else:
            with torch.no_grad():
                target_q, *_ = self._build_q_target(batch_args)

            return target_q

    @staticmethod
    @abstractmethod
    def _build_predicted_args(batch_args):
        pass

    def _build_q_predicted(self, batch_args):
        assert len(batch_args) == 9

        q, sf_s_g, w, phi = self.policy_net(**self._build_predicted_args(batch_args))

        assert q.shape == (self.batch_size, self.action_space)
        assert batch_args["action_batch"].shape == (self.batch_size, )

        # shape (batch_size,)
        predicted_q = q.gather(1, batch_args["action_batch"]).squeeze()

        assert predicted_q.shape == (self.batch_size, )

        return predicted_q, sf_s_g, w, phi

    def _build_psi_predicted(self, batch_args, sf_s_g):
        assert len(batch_args) == 9
        assert sf_s_g.shape == (self.batch_size, self.action_space, self.features_size)
        assert batch_args["action_batch"].shape == (self.batch_size, )

        action_batch = (
            batch_args["action_batch"]
            .reshape(self.batch_size, 1, 1)
            .tile(self.features_size)
        )

        assert action_batch.shape == (self.batch_size, 1, self.features_size)

        # shape (batch_size, features_size)
        predicted_psi = sf_s_g.gather(1, action_batch).squeeze()

        assert predicted_psi.shape == (self.batch_size, self.features_size)

        return predicted_psi

    def _build_predicted_batch(self, batch_args):
        if self.is_a_usf:
            predicted_q, sf_s_g, w, phi = self._build_q_predicted(batch_args)

            predicted_psi = self._build_psi_predicted(batch_args, sf_s_g)

            predicted_r = torch.sum(phi * w, dim=1)

            return predicted_q, predicted_psi, predicted_r

        else:
            predicted_q, *_ = self._build_q_predicted(batch_args)

            return predicted_q

    def _train_one_batch(self):
        experiences, sample_weights = self._sample_experiences()
        sample_weights = sample_weights.to(self.device)

        batch_args = self._build_dictionary_of_batch_from_experiences(experiences)

        if self.use_gdtuo:
            self.optimizer.begin()

        self.optimizer.zero_grad()
        if self.is_a_usf:
            target_batch_q, target_batch_psi, r = self._build_target_batch(
                batch_args,
            )
            predicted_batch_q, predicted_batch_psi, phi_w = self._build_predicted_batch(
                batch_args,
            )

            # shape (batch_size,)
            td_error_q = torch.square(torch.abs(target_batch_q - predicted_batch_q))

            # shape of target_batch_psi is (batch, size_features) so the td_error for
            # that batch must be summed along first dim which automatically squeezes
            # the dim = 1 and so the final shape is (batch_size,)
            td_error_psi = torch.mean(
                torch.square(torch.abs(target_batch_psi - predicted_batch_psi)), dim=1
            )

            # shape (batch_size, )
            td_error_phi = torch.square(torch.abs(r - phi_w))

            total_td_error = (
                self.loss_weight_q * td_error_q
                + self.loss_weight_psi * td_error_psi
                + self.loss_weight_phi * td_error_phi
            )

            # update the priority of batch samples in memory
            self.memory.update_samples(total_td_error.detach().cpu())

            self.memory.anneal_beta()

            loss = torch.mean(sample_weights * total_td_error)
        else:
            target_batch_q = self._build_target_batch(
                batch_args,
            )
            predicted_batch_q = self._build_predicted_batch(
                batch_args,
            )

            td_error_q = torch.square(torch.abs(target_batch_q - predicted_batch_q))

            self.memory.update_samples(td_error_q.detach().cpu())

            self.memory.anneal_beta()

            loss = torch.mean(sample_weights * td_error_q)

        if self.use_gdtuo:
            loss.backward(create_graph=True)
        else:
            loss.backward()

        self.optimizer.step()

        return loss.item()

    def _train(self):
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

        if (
            self.steps_since_last_network_update
            >= self.update_target_network_every_n_steps
        ):
            self.steps_since_last_network_update = 0

            self._update_target_network()
        else:
            self.steps_since_last_network_update += 1

    def train(self, transition):
        self.memory.push(transition)

        if len(self.memory) < self.learning_starts_after:
            return

        self._train()

    def prepare_for_eval_phase(self):
        self.train_memory_buffer = copy.deepcopy(self.memory)

        self.eval_memory_buffer = eu.misc.create_object_from_config(self.config.memory)

    def train_during_eval_phase(self, transition, p_pick_new_memory_buffer):
        self.eval_memory_buffer.push(transition)

        if len(self.eval_memory_buffer) < self.learning_starts_after:
            return
        else:
            if torch.rand(1).item() <= p_pick_new_memory_buffer:
                self.memory = self.eval_memory_buffer
            else:
                self.memory = self.train_memory_buffer

            self._train()

    def _update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = target_net_state_dict[
                key
            ] * self.update_alpha + policy_net_state_dict[key] * (1 - self.update_alpha)

        self.target_net.load_state_dict(target_net_state_dict)

    # Not needed since I will use exputils functionality
    # def save(self, episode, step, total_reward):
    #     filename = "checkpoint" + self.config.save.extension
    #     torch.save(
    #         {
    #             "cls": self.__class__,
    #             "config": self.config,
    #             "episode": episode,
    #             "step": step,
    #             "total_reward": total_reward,
    #             "model_state_dict": self.policy_net.state_dict(),
    #             "optimizer_state_dict": self.optimizer.state_dict(),
    #             "memory": self.memory,
    #             "env_goals_source": self.env.goal_list_source_tasks,
    #             "env_goals_target": self.env.goal_list_target_tasks,
    #             "env_goals_eval": self.env.goal_list_evaluation_tasks,
    #         },
    #         filename,
    #     )
    #
    # @classmethod
    # def load_from_checkpoint(cls, env, filename):
    #     checkpoint = torch.load(filename)
    #
    #     agent = cls(env, config=checkpoint["config"])
    #
    #     agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
    #     agent.target_net = copy.deepcopy(agent.policy_net)
    #
    #     agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #
    #     agent.memory = checkpoint["memory"]
    #     agent.current_episode = checkpoint["episode"]
    #     agent.total_reward = checkpoint["total_reward"]
    #
    #     return agent
