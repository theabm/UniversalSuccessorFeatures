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
        "agent_position_rbf_batch",
        "features_batch",
        "goal_position_batch",
        "goal_position_rbf_batch",
        "goal_weights_batch",
        "action_batch",
        "reward_batch",
        "next_agent_position_batch",
        "next_agent_position_rbf_batch",
        "next_features_batch",
        "terminated_batch",
        "truncated_batch",
    ),
)

FullTransition = namedtuple(
    "FullTransition",
    (
        "agent_position",
        "agent_position_rbf",
        "features",
        "goal_position",
        "goal_position_rbf",
        "goal_weights",
        "action",
        "reward",
        "next_agent_position",
        "next_agent_position_rbf",
        "next_features",
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
            optimal_target_net = None,
            pretrained_agent=None,
            learning_starts_after = None,
            network=eu.AttrDict(
                cls=None,
                # whether to use gradient_descent_the_ultimate_optimizer which 
                # uses hypergradients. Setting to true will use SGD to optimize ADAM
                use_gdtuo=None,
                optimizer=None,
            ),
            target_network_update=eu.AttrDict(
                # "hard" or "soft"
                rule="hard",  
                every_n_steps=10,
                # target network params will be updated as
                # P_t = alpha * P_t + (1-alpha) * P_p
                # where P_p are params of policy network
                alpha=0.0,
            ),
            epsilon=eu.AttrDict(
                cls=eps.EpsilonConstant,
            ),
            memory=eu.AttrDict(
                cls=mem.ExperienceReplayMemory,
                # Need to be defined only for prioritized experience replay
                # alpha is how much I prioritize
                alpha=None,
                # beta0 is how much I correct for skewed distribution
                # should be between 0 and 1
                beta0=None,
                # If we define a schedule, beta0 will be annealed linearly to 
                # 1 at the end of the schedule
                # should be the length of the training steps
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
        # how many actions
        self.action_space = env.action_space.n
        # how many directions (in our case it is 2)
        self.position_size = env.observation_space["agent_position"].shape[1]
        # how many features (we are using one hot encoding so len*width of grid)
        self.features_size = env.features_size
        # size of rbf features (usually the same as features size)
        self.rbf_size = env.rbf_size

        # Setting the device
        if self.config.device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                warnings.warn("Cuda not available. Using CPU as device ...")
        else:
            self.device = torch.device("cpu")

        # Creating object instance of neural network
        if isinstance(self.config.network, dict):
            self.config.network.state_size = self.position_size
            self.config.network.goal_size = self.position_size
            self.config.network.features_size = self.features_size
            self.config.network.num_actions = self.action_space
            self.config.network.rbf_size = self.rbf_size

            self.policy_net = eu.misc.create_object_from_config(self.config.network)
            # whether the policy network is a USF or not (could be a DQN)
            self.is_a_usf = self.policy_net.is_a_usf
        else:
            raise ValueError("Network Config must be a dictionary.")

        # creating instance of memory object
        # we can have normal experience replay, prioritized, and combined
        if isinstance(self.config.memory, dict):
            self.memory = eu.misc.create_object_from_config(self.config.memory)
        else:
            raise ValueError("Memory config must be a dictionary.")

        # the epsilon we use
        # it can be constant, or linearly annealed, or exponentially annealed.
        if isinstance(self.config.epsilon, dict):
            self.epsilon = eu.misc.create_object_from_config(self.config.epsilon)
        else:
            raise ValueError("Network Config must be a dictionary.")

        # the paramters for the loss which is a weighted combination of 
        # L = w_q*Lq + w_psi*L_psi + w_phi * L_phi
        self.loss_weight_q = self.config.loss_weight_q
        self.loss_weight_psi = self.config.loss_weight_psi
        self.loss_weight_phi = self.config.loss_weight_phi

        # use gradient_descent_the_ultimate_optimizer or not
        self.use_gdtuo = self.config.network.use_gdtuo
        if self.use_gdtuo:
            warnings.warn("Using gdtuo... learning rate ignored.")
            self.optimizer = gdtuo.ModuleWrapper(
                self.policy_net, optimizer=gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
            )
            self.optimizer.initialize()
        else:
            self.optimizer = self.config.network.optimizer(
                self.policy_net.parameters(), lr=self.config.learning_rate
            )

        # the batch size we use
        self.batch_size = self.config.batch_size

        # The internal batch size I will work with. Some agents augment the
        # data training automatically so this is the reason this is hidden.

        # The entry point for this variable is in the _sample_experiences function
        self._augmented_batch_size = self.batch_size

        # how often we train. The default case is to train every step.
        self.train_every_n_steps = self.config.train_every_n_steps - 1
        self.steps_since_last_training = 0

        # the discounting factor. The default is 0.99
        self.discount_factor = self.config.discount_factor

        # the rule to update the target network. We can either do a hard update 
        # where we simply swap the two networks after a certain amount of timesteps
        # or we can do a soft update, where at each step we update a small percentage 
        # of the weights.
        if self.config.target_network_update.rule == "hard":
            if self.config.target_network_update.alpha != 0.0:
                warnings.warn(
                    "Alpha = 0.0 for hard update..."
                )
            self.update_alpha = 0.0
        elif self.config.target_network_update.rule == "soft":
            warnings.warn("Using soft update.")
            self.update_alpha = self.config.target_network_update.alpha
        else:
            raise ValueError("Unknown type of update rule.")

        # how often we update the network. The default is every 10 steps
        # for soft updating, it should be set to 1
        self.update_target_network_every_n_steps = (
            self.config.target_network_update.every_n_steps - 1
        )
        self.steps_since_last_network_update = 0

        self.current_episode = 0
        self.step = 0

        # how soon we start learning. This is set to twice the batch size by 
        # default. However, we can set it manually to any value.
        if self.config.learning_starts_after is None:
            self.learning_starts_after = 2*self.batch_size
        else:
            self.learning_starts_after = self.config.learning_starts_after

        # If we want to use a pretrained optimal target net. 
        # This is used when analyzing the stability of the agent that is only 
        # learning the successor features. 
        # in particular, we first train an optimal agent using DQN. This agent 
        # converges very well for gridworld. Then, we use this agent as the 
        # optimal target agent which the policy network has to learn to imitate.
        if self.config.optimal_target_net:
            # This is an instance of an agent - it is passed to this class in the
            # training module.
            self.target_net = self.config.optimal_target_net

            # if we provide an optimal target net, then we cannot 
            # update the target network ever since this is already the "ultimate"
            # truth
            warnings.warn("Using optimal target net. Update frequency = 0")
            self.update_target_network_every_n_steps = np.inf
        else:
            # If I dont use an optimal target net, then initialy, the target 
            # net is just an independent copy of the policy net
            self.target_net = copy.deepcopy(self.policy_net)

        # sending the nets to the device (should be gpu)
        self.target_net.to(self.device)
        self.policy_net.to(self.device)

        if self.config.pretrained_agent:
            self.choose_action = self.config.pretrained_agent.choose_action

    def start_episode(self, episode):
        """The function used to start an episode. It simply logs some 
        initial information such as the current episode, the current epsilon 
        value"""
        self.current_episode = episode
        if self.config.log.epsilon_per_episode:
            log.add_value(self.config.log.log_name_epsilon, self.epsilon.value)

    def end_episode(self):
        """Ends the episode. It is used to make epsilon decay."""
        self.epsilon.decay()

    def choose_action(
        self,
        obs,
        list_of_goal_positions,
        training,
    ):
        """Choose an action based on a single observation, a list of goal positions, 
        and whether we are in training mode or not.

        If we are in training mode, we will do epsilon greedy strategy, where 
        we select a random action with a certain probability. Otherwise we 
        select the best action.
        If we are not in training mode, we only select the best action.
        """
        if training:
            return self._epsilon_greedy_action_selection(
                obs,
                list_of_goal_positions,
            )#.item()
        else:
            return self._greedy_action_selection(
                obs,
                list_of_goal_positions,
            )#.item()

    def _epsilon_greedy_action_selection(self, obs, list_of_goal_positions):
        """Epsilon greedy action selection. We perform a random action a certain 
        amount of time.

        This function returns a tuple (action, goal_chosen). 
        The goal chosen is useful when using GPI to understand which goal gave 
        rise to that action. 
        If we are not using GPI, then the goal chosen is the current goal we 
        are trying to reach.
        """
    
        # if the random value is greater than epsilon, we take a specific action
        if torch.rand(1).item() > self.epsilon.value:
            return self._greedy_action_selection(
                obs,
                list_of_goal_positions,
            )
        else:
            # we encode the chosen goal as -1 when the action was picked 
            # randomly and not with GPI
            return torch.randint(0, self.action_space, (1,)), -1

    def _greedy_action_selection(self, obs, list_of_goal_positions):
        agent_position = obs["agent_position"]
        # A key idea: at each iteration, we need to check that the position of
        # agent is not one of the goals in my list. This is because goal
        # positions are terminal states, so the agent never learns the psi
        # function there. So the value could be arbitrarily high and interfere
        # with the GPI procedure in selecting the correct action.
        # include in thesis.

        list_of_goal_positions = [
            goal for goal in list_of_goal_positions if (goal != agent_position).any()
        ]

        q_per_goal = torch.zeros(len(list_of_goal_positions))
        a_per_goal = torch.zeros(len(list_of_goal_positions), dtype=int)

        for i, goal_position in enumerate(list_of_goal_positions):
            obs_dict = self._build_arguments_from_obs(obs, goal_position)
            with torch.no_grad():
                q, *_ = self.policy_net(
                    **obs_dict,
                )
                qm, am = torch.max(q, axis=1)
                q_per_goal[i] = qm.item()
                a_per_goal[i] = am.item()
        # batch together for gpu in the future
        max_policy = torch.argmax(q_per_goal)

        return a_per_goal[max_policy.item()].item(), max_policy.item()

    @abstractmethod
    def _build_arguments_from_obs(self, obs, goal_position):
        pass

    def _print_successor_features(self, obs, list_of_goal_positions):
        if self.is_a_usf:
            for i, goal_position in enumerate(list_of_goal_positions):
                obs_dict = self._build_arguments_from_obs(obs, goal_position)
                with torch.no_grad():
                    q, sf, *_ = self.policy_net(
                        **obs_dict,
                    )
                    sf = sf.squeeze().reshape(
                        self.action_space, self.env.rows, self.env.columns
                    )
                    print(
                        f"SF at: {obs['features']}\nFor goal{goal_position}\n",
                        sf,
                    )
        else:
            raise "This function is only available for USF's"

    def _sample_experiences(self, list_of_goal_positions_for_augmentation):
        # The list_of_goal_positions_for_augmentation argument is used only in
        # overriden version

        # Since in some agents I augment batch size, I need to initialize the
        # new batch size here.

        experiences, weights = self.memory.sample(self.batch_size)

        assert type(experiences) == list
        assert len(experiences) == self.batch_size

        # by default _augmented_batch_size = batch_size so I dont need to set it
        # here. However, in featuregoalagent where I have overriden this
        # function, I need to set it as appropriate

        # self._augmented_batch_size = self.batch_size

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
            # shape (batch_size, rbf_size)
            "agent_position_rbf_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.agent_position_rbf_batch
            ).to(self.device),
            # shape (batch_size, feature_size)
            "features_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.features_batch
            ).to(self.device),
            # shape (batch_size, position_size)
            "goal_position_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.goal_position_batch
            ).to(self.device),
            "goal_position_rbf_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.goal_position_rbf_batch
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
            "next_agent_position_rbf_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.next_agent_position_rbf_batch
            ).to(self.device),
            # shape (batch_size, feature_size)
            "next_features_batch": self._build_tensor_from_batch_of_np_arrays(
                experiences.next_features_batch
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

        assert len(batch_args) == 12

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
        assert len(batch_args) == 12

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
        assert len(batch_args) == 12

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
        assert len(batch_args) == 12
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
        """Prepares the agent for the second phase where we 
        evaluate a second (or third) set of goals. As described 
        in the paper, we keep the previous memory buffer and start 
        building a new one. When we train, we select either memory 
        buffer with equal probability.
        """
        # do a deep copy of the memory buffer
        self.train_memory_buffer = copy.deepcopy(self.memory)

        # build a new memory from the config. 
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
            ## mod starts here
            # the modification is to train the agent on both memory buffers 
            # instead of only on one. 
            # However it would be very bad if I tried to train more times per 
            # iteration.
            self.memory = self.eval_memory_buffer
            self._train(list_of_goal_positions_for_augmentation)
            self.memory = self.train_memory_buffer
            self._train(list_of_goal_positions_for_augmentation)
            # mod ends here

            # uncomment for original
            # if torch.rand(1).item() <= p_pick_new_memory_buffer:
            #     self.memory = self.eval_memory_buffer
            # else:
            #     self.memory = self.train_memory_buffer
            #
            # self._train(list_of_goal_positions_for_augmentation)

    def _update_target_network(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = target_net_state_dict[
                key
            ] * self.update_alpha + policy_net_state_dict[key] * (1 - self.update_alpha)

        self.target_net.load_state_dict(target_net_state_dict)

    def save(self, filename, episode, step, total_reward):
        torch.save(
            {
                "cls": self.__class__,
                "config": self.config,
                "episode": episode,
                "step": step,
                "total_reward": total_reward,
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "memory": self.memory,
                "env_goals_source": self.env.goal_list_source_tasks,
                "env_goals_target": self.env.goal_list_target_tasks,
                "env_goals_eval": self.env.goal_list_evaluation_tasks,
            },
            filename,
        )

    @staticmethod
    def load_from_checkpoint(env, filename, config = None):
        checkpoint = torch.load(filename)

        agent_class = checkpoint["cls"]
        agent_config = eu.combine_dicts(config, checkpoint["config"])

        agent = agent_class(env, config=agent_config)

        agent.policy_net.load_state_dict(checkpoint["model_state_dict"])
        agent.target_net = copy.deepcopy(agent.policy_net)

        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        agent.memory = checkpoint["memory"]
        agent.current_episode = checkpoint["episode"]
        agent.total_reward = checkpoint["total_reward"]

        agent._env_primary_goals = checkpoint["env_goals_source"]
        agent._env_secondary_goals = checkpoint["env_goals_target"]
        agent._env_tertiary_goals = checkpoint["env_goals_eval"]

        return agent
