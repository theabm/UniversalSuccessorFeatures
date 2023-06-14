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

FullTransition = namedtuple(
    "FullTransition",
    (
        "agent_position",
        "agent_position_features",
        "goal",
        "goal_weights",
        "action",
        "reward",
        "next_agent_position",
        "next_agent_position_features",
        "terminated",
        "truncated",
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

        # The internal batch size I will work with
        # I set it to the same value, but it will be determined during training.
        # by the list of goal positions over which I want to evaluate.
        # The entry point is the _train_one_batch_function
        self._augmented_batch_size = self.batch_size

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

    def _sample_experiences(self, list_of_goal_positions_for_augmentation):
        # one of the arguments is used only in overriden version

        # since in some agents I augment batch size, I need to initialize the
        # new batch size here.

        experiences, weights = self.memory.sample(self.batch_size)

        assert type(experiences) == list
        assert len(experiences) == self.batch_size

        self._augmented_batch_size = self.batch_size

        return Experiences(*zip(*experiences)), torch.tensor(weights)

    def _build_tensor_from_batch_of_np_arrays(self, list_of_np_arrays):
        # This function is called from the training function where I will have
        # defined the true augmented batch size.
        # So it needs to use the augmented batch size parameter.
        assert len(list_of_np_arrays) == self._augmented_batch_size
        assert len(list_of_np_arrays[0].shape) == 2

        len_input = list_of_np_arrays[0].shape[1]

        # expected shape: [(1,n), (1,n), ..., (1,n)] where in total we
        # have batch_size elements
        list_of_np_arrays = np.array(list_of_np_arrays)
        assert list_of_np_arrays.shape == (self._augmented_batch_size, 1, len_input)

        # batch of np_arrays has form (batch_size, 1, n) so after squeeze()
        # we have (batch_size, n)
        batch_of_np_arrays = (
            torch.tensor(list_of_np_arrays)
            .reshape(self._augmented_batch_size, len_input)
            .to(torch.float)
        )

        assert batch_of_np_arrays.shape == (self._augmented_batch_size, len_input)

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
        # sf's will be checked in the appropriate function that uses it.

        assert len(batch_args) == 9

        q, sf, w, reward_phi_batch = self.target_net(
            **self._build_target_args(batch_args)
        )

        assert q.shape == (self._augmented_batch_size, self.action_space)

        q_max, action = torch.max(q, axis=1)

        assert q_max.shape == (self._augmented_batch_size,)
        assert action.shape == (self._augmented_batch_size,)
        assert batch_args["reward_batch"].shape == (self._augmented_batch_size,)
        assert batch_args["terminated_batch"].shape == (self._augmented_batch_size,)

        target_q = batch_args["reward_batch"] + self.discount_factor * torch.mul(
            q_max, ~batch_args["terminated_batch"]
        )
        assert target_q.shape == (self._augmented_batch_size,)

        return target_q, action, sf, w, reward_phi_batch

    def _build_psi_target(self, batch_args, action, sf, reward_phi_batch):
        # This function is only called inside the train function which has
        # properly defined the augmented batch size
        assert batch_args["terminated_batch"].shape == (self._augmented_batch_size,)
        assert action.shape == (self._augmented_batch_size,)
        assert sf.shape == (
            self._augmented_batch_size,
            self.action_space,
            self.features_size,
        )
        assert reward_phi_batch.shape == (
            self._augmented_batch_size,
            self.features_size,
        )

        terminated_batch = batch_args["terminated_batch"].unsqueeze(1)

        assert terminated_batch.shape == (self._augmented_batch_size, 1)

        # shape (batch_size,1,n)
        action = (
            action.reshape(self._augmented_batch_size, 1, 1)
            .tile(self.features_size)
            .to(self.device)
        )

        assert action.shape == (self._augmented_batch_size, 1, self.features_size)

        max_sf = sf.gather(1, action)
        assert max_sf.shape == (self._augmented_batch_size, 1, self.features_size)

        max_sf = max_sf.reshape(self._augmented_batch_size, self.features_size)
        assert max_sf.shape == (self._augmented_batch_size, self.features_size)

        target_psi = reward_phi_batch + self.discount_factor * torch.mul(
            max_sf, ~terminated_batch
        )

        assert target_psi.shape == (self._augmented_batch_size, self.features_size)

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

        assert q.shape == (self._augmented_batch_size, self.action_space)
        assert batch_args["action_batch"].shape == (self._augmented_batch_size, 1)

        # shape (batch_size,)
        predicted_q = q.gather(1, batch_args["action_batch"]).reshape(
            self._augmented_batch_size
        )

        assert predicted_q.shape == (self._augmented_batch_size,)

        return predicted_q, sf_s_g, w, phi

    def _build_psi_predicted(self, batch_args, sf_s_g):
        assert len(batch_args) == 9
        assert sf_s_g.shape == (
            self._augmented_batch_size,
            self.action_space,
            self.features_size,
        )
        assert batch_args["action_batch"].shape == (self._augmented_batch_size, 1)

        action_batch = (
            batch_args["action_batch"]
            .reshape(self._augmented_batch_size, 1, 1)
            .tile(self.features_size)
        )

        assert action_batch.shape == (self._augmented_batch_size, 1, self.features_size)

        # shape (batch_size, features_size)
        predicted_psi = sf_s_g.gather(1, action_batch).reshape(
            self._augmented_batch_size, self.features_size
        )

        assert predicted_psi.shape == (self._augmented_batch_size, self.features_size)

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

    def _get_td_error_for_usf(
        self,
        target_batch_q,
        target_batch_psi,
        target_batch_r,
        predicted_batch_q,
        predicted_batch_psi,
        predicted_batch_r,
    ):
        """Compute MSE loss for batches"""

        td_error_q = torch.square(torch.abs(target_batch_q - predicted_batch_q))

        assert td_error_q.shape == (self._augmented_batch_size,)

        # shape of target_batch_psi is (batch, size_features) so the td_error for
        # that batch must be summed along first dim which automatically squeezes
        # the dim = 1 and so the final shape is (batch_size,)
        td_error_psi = torch.mean(
            torch.square(torch.abs(target_batch_psi - predicted_batch_psi)), dim=1
        )

        assert td_error_psi.shape == (self._augmented_batch_size,)

        # shape (batch_size, )
        td_error_phi = torch.square(torch.abs(target_batch_r - predicted_batch_r))

        assert td_error_phi.shape == (self._augmented_batch_size,)

        total_td_error = (
            self.loss_weight_q * td_error_q
            + self.loss_weight_psi * td_error_psi
            + self.loss_weight_phi * td_error_phi
        )
        assert total_td_error.shape == (self._augmented_batch_size,)

        return total_td_error

    # def _sample_and_augment_experiences(self, list_of_goal_positions_for_augmentation):
    #     # list of goal positions holds all the goals over which I want
    #     # to augment my training. It can be a single goal, in which case
    #     # I am doing no augmentation, or it can be the full set of goals
    #     # over which I am learning.
    #
    #     assert type(list_of_goal_positions_for_augmentation) == list
    #
    #     # first I sample batch_size experiences.
    #     # these experiences are named tuples of the following structure:
    #     # (agent_position, agent_position_features, action, next_agent_position,
    #     # next_agent_position_features)
    #
    #     # Here I must always use the true batch size I have determined
    #     experiences, weights = self.memory.sample(self.batch_size)
    #     assert type(experiences) == list
    #     assert len(experiences) == self.batch_size
    #
    #     # Then, from these, we will construct a full transition tuple for each
    #     # of the goals by doing the following:
    #     # goal - given by goal list
    #     # goal_weight - calculated by environment
    #     # reward - goal_weight*next_agent_position_features
    #     # terminated - true if position == goal
    #     # truncated - we dont need for training.
    #
    #     augmented_experiences = []
    #
    #     for experience in experiences:
    #         for goal_position in list_of_goal_positions_for_augmentation:
    #             goal_weights = self.env._get_goal_weights_at(goal_position)
    #             assert goal_weights.shape == (1, self.features_size)
    #
    #             reward = int(
    #                 np.sum(experience.next_agent_position_features * goal_weights)
    #             )
    #             assert type(reward) == int
    #
    #             terminated = (
    #                 True
    #                 if (goal_position == experience.next_agent_position).all()
    #                 else False
    #             )
    #             assert type(terminated) == bool
    #
    #             new_experience = FullTransition(
    #                 experience.agent_position,
    #                 experience.agent_position_features,
    #                 goal_position,
    #                 goal_weights,
    #                 experience.action,
    #                 reward,
    #                 experience.next_agent_position,
    #                 experience.next_agent_position_features,
    #                 terminated,
    #                 experience.truncated
    #             )
    #
    #             augmented_experiences.append(new_experience)
    #
    #     # this is the real batch size I need to work with.
    #     len_list_goals = len(list_of_goal_positions_for_augmentation)
    #
    #     self._augmented_batch_size = len_list_goals * self.batch_size
    #
    #     assert len(augmented_experiences) == self._augmented_batch_size
    #
    #     assert weights.shape == (self.batch_size,)
    #
    #     # We take the weights, reshape to (batch,1) so that we can tile in the
    #     # first dimension, then tile to obtain (batch, 12) then we reshape to
    #     # batch*12.
    #     augmented_weights = weights.reshape((self.batch_size, 1))
    #
    #     augmented_weights = np.tile(augmented_weights, (1, len_list_goals))
    #
    #     assert augmented_weights.shape == (self.batch_size, len_list_goals)
    #
    #     augmented_weights = augmented_weights.reshape((self._augmented_batch_size,))
    #
    #     assert augmented_weights.shape == (self._augmented_batch_size,)
    #
    #     return Experiences(*zip(*augmented_experiences)), torch.tensor(
    #         augmented_weights
    #     )

    def _update_memory(self, td_error, list_of_goal_positions_for_augmentation):
        # update the priority of batch samples in memory
        self.memory.update_samples(td_error.detach().cpu())

        self.memory.anneal_beta()

    def _train_one_batch(self, list_of_goal_positions_for_augmentation):
        # for fgw they will be augmented
        experiences, sample_weights = self._sample_experiences(
            list_of_goal_positions_for_augmentation
        )

        sample_weights = sample_weights.to(self.device)

        batch_args = self._build_dictionary_of_batch_from_experiences(experiences)

        if self.use_gdtuo:
            self.optimizer.begin()

        self.optimizer.zero_grad()

        if self.is_a_usf:
            target_batch_q, target_batch_psi, target_batch_r = self._build_target_batch(
                batch_args,
            )

            (
                predicted_batch_q,
                predicted_batch_psi,
                predicted_batch_r,
            ) = self._build_predicted_batch(
                batch_args,
            )

            total_td_error = self._get_td_error_for_usf(
                target_batch_q,
                target_batch_psi,
                target_batch_r,
                predicted_batch_q,
                predicted_batch_psi,
                predicted_batch_r,
            )

            loss = torch.mean(sample_weights * total_td_error)

            self._update_memory(total_td_error, list_of_goal_positions_for_augmentation)

        else:
            target_batch_q = self._build_target_batch(
                batch_args,
            )
            predicted_batch_q = self._build_predicted_batch(
                batch_args,
            )

            td_error_q = torch.square(torch.abs(target_batch_q - predicted_batch_q))

            loss = torch.mean(sample_weights * td_error_q)

            self._update_memory(td_error_q, list_of_goal_positions_for_augmentation)

        if self.use_gdtuo:
            loss.backward(create_graph=True)
        else:
            loss.backward()

        self.optimizer.step()

        return loss.item()

    def _train(self, list_of_goal_positions_for_augmentation):
        if self.steps_since_last_training >= self.train_every_n_steps:
            self.steps_since_last_training = 0

            losses = []
            for _ in range(self.config.train_for_n_iterations):
                loss = self._train_one_batch(list_of_goal_positions_for_augmentation)
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

    def train(self, transition, list_of_goal_positions_for_augmentation):
        self.memory.push(transition)

        if len(self.memory) < self.learning_starts_after:
            return

        self._train(list_of_goal_positions_for_augmentation)

    def prepare_for_eval_phase(self):
        self.train_memory_buffer = copy.deepcopy(self.memory)

        self.eval_memory_buffer = eu.misc.create_object_from_config(self.config.memory)

    def train_during_eval_phase(
        self,
        transition,
        p_pick_new_memory_buffer,
        list_of_goal_positions_for_augmentation,
    ):
        self.eval_memory_buffer.push(transition)

        if len(self.eval_memory_buffer) < self.learning_starts_after:
            return
        else:
            if torch.rand(1).item() <= p_pick_new_memory_buffer:
                self.memory = self.eval_memory_buffer
            else:
                self.memory = self.train_memory_buffer

            self._train(list_of_goal_positions_for_augmentation)

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
