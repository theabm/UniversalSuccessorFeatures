import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import exputils as eu
import random
import warnings
from universal_successor_features.envs.directions import Directions
import pickle

# Note that RBF vectors can be used as features only if r = rbf * w is true. 
# this does not hold for a reward of -0.1 at every time step and 0 at the 
# goal.
# On the other hand, it can always be used as a substitute for the position. 
# in other words, instead of (x,y) we give an rbf vector of fixed size. 

class GridWorld(gym.Env):
    @staticmethod
    def default_config():
        return eu.AttrDict(
            rows=9,
            columns=9,
            nmax_steps=31,
            penalization=0,
            reward_at_goal_position=20,
            one_hot_weight=1,
            n_goals=12,
            rbf_points_in_x_direction=9,
            rbf_points_in_y_direction=9,
        )

    def __init__(self, config=None, **kwargs):
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, GridWorld.default_config())

        # length and width of the grid
        self.rows = self.config.rows
        self.columns = self.config.columns
        self.features_size = self.rows*self.columns

        # how many rbf points
        self.rbf_x = self.config.rbf_points_in_x_direction
        self.rbf_y = self.config.rbf_points_in_y_direction
        self.rbf_size = self.rbf_x*self.rbf_y

        # sigma of gaussian for rbf points
        self.sigma = 1

        # Action space of the environment.
        # Agent can only move up, down, left, right, or stay
        self.action_space = gym.spaces.Discrete(4)

        # observation space of the agent are the x,y coordinates
        self.observation_space = gym.spaces.Dict(
            {
                "agent_position": gym.spaces.MultiDiscrete(
                    np.array([[self.rows, self.columns]])
                ),
                "agent_position_features": gym.spaces.MultiBinary(
                    [1, self.features_size]
                ),
                "agent_position_features_rbf": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1, self.rbf_size)
                ),
                "goal_position": gym.spaces.MultiDiscrete(
                    np.array([[self.rows, self.columns]])
                ),
                "goal_position_features": gym.spaces.MultiBinary(
                    [1, self.features_size]
                ),
                "goal_position_features_rbf": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1, self.rbf_size)
                ),
                "goal_weights": gym.spaces.MultiBinary([1, self.features_size]),
            }
        )

        dim1_components = np.linspace(1, self.rows, self.rbf_x)
        dim2_components = np.linspace(1, self.columns, self.rbf_y)

        self.rbf_grid = np.array(
            np.meshgrid(dim1_components, dim2_components)
        ).T.reshape(-1, 2)
        assert self.rbf_grid.shape == (self.rbf_size, 2)

        self.reward_range = (-np.inf, 0)

        # Position
        self.agent_i = None
        self.agent_j = None

        # Old position
        self.old_agent_i = None
        self.old_agent_j = None

        # Goal position
        self.goal_i = None
        self.goal_j = None

        self.cur_step = None
        self.nmax_steps = self.config.nmax_steps

        # generated randomly once for 9x9 and fixed forever.
        self.n_goals = self.config.n_goals
        (
            self.goal_list_source_tasks,
            self.goal_list_target_tasks,
            self.goal_list_evaluation_tasks,
        ) = self._create_three_disjoint_goal_lists()

        # what will make up observation space
        self.agent_position = None
        self.agent_position_features = None
        self.agent_position_features_rbf = None
        self.goal_position = None
        self.goal_position_features = None
        self.goal_position_features_rbf = None
        self.goal_weights = None

    def _sample_position_in_grid(self):
        """
        Samples a row and a column from the predefined matrix dimensions
        and returns them.
        """
        i = np.random.choice(self.rows)
        j = np.random.choice(self.columns)

        return i, j

    def _create_three_disjoint_goal_lists(self):
        """
        Create three disjoint set of goals.

        Primary/Source goals are the initial goals to train on.

        Secondary/Target goals are the goals trained on in second phase

        Tertiary/Evaluation goals are goals we evaluate on in the first
        phase
        """
        all_possible_goals = [
            np.array([[i, j]]) for i in range(self.rows) for j in range(self.columns)
        ]
        goals = random.sample(all_possible_goals, 3 * self.n_goals)
        return (
            goals[: self.n_goals],
            goals[self.n_goals : 2 * self.n_goals],
            goals[2 * self.n_goals :],
        )

    def sample_source_goal(self):
        """Randomly sample a goal from the primary/source goals."""
        return self.sample_a_goal_position_from_list(self.goal_list_source_tasks)

    def sample_target_goal(self):
        """Randomly sample a goal from the secondary/target goals."""
        return self.sample_a_goal_position_from_list(self.goal_list_target_tasks)

    def sample_eval_goal(self):
        """Randomly sample a goal from the tertiary/evaluation goals."""
        return self.sample_a_goal_position_from_list(self.goal_list_evaluation_tasks)

    def sample_a_goal_position_from_list(self, goal_list):
        """Randomly sample a goal position from a list of goals."""
        idx = random.randrange(len(goal_list))
        return goal_list[idx]

    def reset(
        self,
        start_agent_position: np.ndarray = None,
        goal_position: np.ndarray = None,
        seed=None,
    ):
        """Reset the environment."""
        super().reset(seed=seed)
        self.cur_step = 0

        # position of the agent and goal in x,y coordinates
        if start_agent_position is not None:
            self.agent_i = start_agent_position[0][0]
            self.agent_j = start_agent_position[0][1]
        else:
            self.agent_i, self.agent_j = self._sample_position_in_grid()

        if goal_position is not None:
            self.goal_i = goal_position[0][0]
            self.goal_j = goal_position[0][1]

            # if same, give priority to goal that was set
            while (self.agent_i, self.agent_j) == (self.goal_i, self.goal_j):
                warnings.warn(
                    "Start and goal position cannot be the same. Using different start"
                )
                self.agent_i, self.agent_j = self._sample_position_in_grid()
        else:
            self.goal_i, self.goal_j = self._sample_position_in_grid()

            while (self.agent_i, self.agent_j) == (self.goal_i, self.goal_j):
                self.goal_i, self.goal_j = self._sample_position_in_grid()

        # good sanity check
        if (self.agent_i, self.agent_j) == (self.goal_i, self.goal_j):
            raise ValueError("Start and Goal position cannot be the same.")

        #######################################################################
        #               Until here, goal and position have been set
        #                   Now I need to create observation
        #######################################################################

        # These are constants that are calculated just once.

        # the position of the goal
        self.goal_position = np.array([[self.goal_i, self.goal_j]])

        # the one hot encoding features of the goal (just to maintain symmetry
        # with position features)
        self.goal_position_features = self._get_one_hot_vector_at(self.goal_position)
        assert self.goal_position_features.shape == (1, self.features_size)

        # the rbf goal position features
        self.goal_position_features_rbf = self._get_rbf_vector_at(self.goal_position)
        assert self.goal_position_features_rbf.shape == (1, self.rbf_size)

        self.goal_weights = self._get_goal_weights_at(self.goal_position)
        assert self.goal_position_features.shape == (1, self.features_size)

        obs = self.build_new_observation()

        return obs, {}

    def check_boundary_conditions(self):
        "Check the boundary conditions of the grid."
        if self.agent_i < 0:
            self.agent_i = 0
        if self.agent_i > self.rows - 1:
            self.agent_i = self.rows - 1

        if self.agent_j < 0:
            self.agent_j = 0
        if self.agent_j > self.columns - 1:
            self.agent_j = self.columns - 1

    def modify_agent_position_according_to_action(self, action):
        # Actions:                   o o o o o               o x o o o
        #                           o x o o o   -GO "UP"-   o o o o o
        # 0 - up                     o o o o o               o o o o o
        # 1 - down                   o o o o o               o o o o o
        # 2 - right
        # 3 - left
        if action == Directions.UP:
            self.agent_i -= 1
        elif action == Directions.DOWN:
            self.agent_i += 1
        elif action == Directions.RIGHT:
            self.agent_j += 1
        elif action == Directions.LEFT:
            self.agent_j -= 1
        else:
            raise ValueError("Agent can only perform: up:0, down:1, right:2, left:3.")

    def get_step_info(self):
        terminated = (self.agent_i, self.agent_j) == (self.goal_i, self.goal_j)

        truncated = self.cur_step >= self.nmax_steps

        reward = (
            self.config.reward_at_goal_position
            if terminated
            else self.config.penalization
        )

        return reward, terminated, truncated

    def build_new_observation(self):
        self.agent_position = np.array([[self.agent_i, self.agent_j]])
        self.agent_position_features = self._get_one_hot_vector_at(self.agent_position)
        self.agent_position_features_rbf = self._get_rbf_vector_at(self.agent_position)

        assert self.agent_position.shape == (1, 2)
        assert self.agent_position_features.shape == (1, self.features_size)
        assert self.agent_position_features_rbf.shape == (1, self.rbf_size)

        obs = {
            "agent_position": self.agent_position,
            "agent_position_features": self.agent_position_features,
            "agent_position_features_rbf": self.agent_position_features_rbf,
            "goal_position": self.goal_position,
            "goal_position_features": self.goal_position_features,
            "goal_position_features_rbf": self.goal_position_features_rbf,
            "goal_weights": self.goal_weights,
        }

        return obs

    def step(self, action):
        self.cur_step += 1
        self.action = action

        self.old_agent_i = self.agent_i
        self.old_agent_j = self.agent_j

        self.modify_agent_position_according_to_action(action)
        self.check_boundary_conditions()

        reward, terminated, truncated = self.get_step_info()
        self.reward = reward

        obs = self.build_new_observation()

        return obs, reward, terminated, truncated, {}

    def render(self):
        print(
            f"Goal: ({self.goal_i},{self.goal_j})\t",
            f"old position: ({self.old_agent_i},{self.old_agent_j})\t",
            f"action: {Directions(self.action).name}\t",
            f"new position: ({self.agent_i},{self.agent_j})\t",
            f"reward: {self.reward}",
        )

    def _make_full_grid_and_place_val_in(self, i, j, full_val, val):
        grd = np.full((self.rows, self.columns), full_val)
        grd[i][j] = val
        return grd.reshape((1, self.rows * self.columns))

    # formerly _get_agent_position_features_at
    def _get_one_hot_vector_at(self, position: np.ndarray):
        """
        Return one hot vector corresponding to a given position
        """
        i = position[0][0]
        j = position[0][1]
        return self._make_full_grid_and_place_val_in(
            i, j, 0, self.config.one_hot_weight
        )

    def _get_rbf_vector_at(self, position: np.ndarray):
        """
        Return vector of size rbf_x*rbf_y. Each component is the
        gaussian density of the rbf points wrt to the position.
        """
        rbf_vector = np.exp(
            -1
            * np.sum((position - self.rbf_grid) * (position - self.rbf_grid), axis=1)
            / self.sigma
        )
        assert rbf_vector.shape == (self.rbf_x * self.rbf_y,)

        rbf_vector = rbf_vector.reshape(1, self.rbf_x * self.rbf_y)

        return rbf_vector

    def _get_goal_weights_at(self, goal_pos: np.ndarray):
        i = goal_pos[0][0]
        j = goal_pos[0][1]
        return self._make_full_grid_and_place_val_in(
            i, j, self.config.penalization, self.config.reward_at_goal_position
        )

    def save(self):
        filename = "env_config.cfg"

        self.config.goal_list_source_tasks = self.goal_list_source_tasks
        self.config.goal_list_target_tasks = self.goal_list_target_tasks
        self.config.goal_list_evaluation_tasks = self.goal_list_evaluation_tasks

        with open(filename, "wb") as fp:
            pickle.dump(self.config, fp)

    @classmethod
    def load_from_checkpoint(cls, filename):
        with open(filename, "rb") as fp:
            config = pickle.load(fp)

        env = cls(config)
        env.goal_list_source_tasks = config["goal_list_source_tasks"]
        env.goal_list_target_tasks = config["goal_list_target_tasks"]
        env.goal_list_evaluation_tasks = config["goal_list_evaluation_tasks"]

        return env


if __name__ == "__main__":
    eu.misc.seed(0)

    grid_world_env = GridWorld()
    # reset missing **kwargs argument but I dont want this functionality.
    # check_env(grid_world_env)
    print(grid_world_env.observation_space["agent_position_features"].shape[1])
    print(grid_world_env.observation_space["agent_position"].shape[1])
    l1, l2, l3 = grid_world_env._create_three_disjoint_goal_lists()
    print(l1)
    print(l2)
    print(l3)
    grid_world_env.reset()
    for _ in range(100):
        action = grid_world_env.action_space.sample()
        obs, reward, terminated, truncated, info = grid_world_env.step(action)
        grid_world_env.render()
