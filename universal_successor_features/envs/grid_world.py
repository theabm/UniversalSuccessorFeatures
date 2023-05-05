import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import exputils as eu
import random
import warnings
from universal_successor_features.envs.directions import Directions


class GridWorld(gym.Env):
    @staticmethod
    def default_config():
        return eu.AttrDict(
            rows=10,
            columns=10,
            nmax_steps=1e6,
            penalization=-0.1,
            reward_at_goal_position=0,
            n_goals=1,
        )

    def __init__(self, config=None, **kwargs):
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, GridWorld.default_config())

        # length and width of the grid
        self.rows = self.config.rows
        self.columns = self.config.columns

        # action space of the environment. Agent can only move up, down, left, right, or stay
        self.action_space = gym.spaces.Discrete(4)

        # observation space of the agent are the x,y coordinates
        self.observation_space = gym.spaces.Dict(
            {
                "agent_position": gym.spaces.MultiDiscrete(
                    np.array([[self.rows, self.columns]])
                ),
                "agent_position_features": gym.spaces.MultiBinary(
                    [1, self.rows * self.columns]
                ),
                "goal_position": gym.spaces.MultiDiscrete(
                    np.array([[self.rows, self.columns]])
                ),
                "goal_weights": gym.spaces.MultiBinary([1, self.rows * self.columns]),
            }
        )

        self.reward_range = (-np.inf, 0)

        self.agent_i = None
        self.agent_j = None

        self.goal_i = None
        self.goal_j = None

        self.cur_step = None
        self.nmax_steps = self.config.nmax_steps

        # generated randomly once for 10x10 and fixed forever.
        self.n_goals = self.config.n_goals
        (
            self.goal_list_source_tasks,
            self.goal_list_target_tasks,
            self.goal_list_evaluation_tasks,
        ) = self._create_three_disjoint_goal_lists()

    def _sample_position_in_grid(self):
        """Samples a row and a column from the predefined matrix dimensions.
        Returns a tuple of int.
        """
        i = np.random.choice(self.rows)
        j = np.random.choice(self.columns)

        return i, j

    def _create_three_disjoint_goal_lists(self):
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
        return self.sample_a_goal_position_from_list(self.goal_list_source_tasks)

    def sample_target_goal(self):
        return self.sample_a_goal_position_from_list(self.goal_list_target_tasks)

    def sample_eval_goal(self):
        return self.sample_a_goal_position_from_list(self.goal_list_evaluation_tasks)

    def sample_a_goal_position_from_list(self, goal_list):
        idx = random.randrange(len(goal_list))
        return goal_list[idx]

    def reset(
        self,
        start_agent_position: np.ndarray = None,
        goal_position: np.ndarray = None,
        seed=None,
    ):
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
                    "Start position and goal position cannot be the same. Proceeding with different start position..."
                )
                self.agent_i, self.agent_j = self._sample_position_in_grid()
        else:
            self.goal_i, self.goal_j = self._sample_position_in_grid()

            while (self.agent_i, self.agent_j) == (self.goal_i, self.goal_j):
                self.goal_i, self.goal_j = self._sample_position_in_grid()

        # good sanity check
        if (self.agent_i, self.agent_j) == (self.goal_i, self.goal_j):
            raise ValueError("Start and Goal position cannot be the same.")

        position = np.array([[self.agent_i, self.agent_j]])
        position_features = self._get_current_agent_position_features()
        self.goal = np.array([[self.goal_i, self.goal_j]])
        self.goal_weights = self._get_current_goal_weights()

        obs = {
            "agent_position": position,
            "agent_position_features": position_features,
            "goal_position": self.goal,
            "goal_weights": self.goal_weights,
        }

        return obs, {}

    def check_boundary_conditions(self):
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
            raise ValueError(
                "Unrecognized action. Agent can only perform the following actions: up:0, down:1, right:2, left:3."
            )

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
        position = np.array([[self.agent_i, self.agent_j]])
        position_features = self._get_current_agent_position_features()

        obs = {
            "agent_position": position,
            "agent_position_features": position_features,
            "goal_position": self.goal,
            "goal_weights": self.goal_weights,
        }

        return obs

    def step(self, action):
        self.cur_step += 1

        self.modify_agent_position_according_to_action(action)
        self.check_boundary_conditions()

        reward, terminated, truncated = self.get_step_info()

        obs = self.build_new_observation()

        return obs, reward, terminated, truncated, {}

    def render(self, action, reward):
        print(
            f"Action: {Directions(action).name},\t position: ({self.agent_i},{self.agent_j}),\t reward: {reward}"
        )

    def _make_full_grid_and_place_val_in(self, i, j, full_val, val):
        grd = np.full((self.rows, self.columns), full_val)
        grd[i][j] = val
        return grd.reshape((1, self.rows * self.columns))

    def _get_agent_position_features_at(self, position: np.ndarray):
        i = position[0][0]
        j = position[0][1]
        return self._make_full_grid_and_place_val_in(i, j, 0, 1)

    def _get_current_agent_position_features(self):
        return self._make_full_grid_and_place_val_in(self.agent_i, self.agent_j, 0, 1)

    def _get_current_goal_weights(self):
        return self._make_full_grid_and_place_val_in(
            self.goal_i,
            self.goal_j,
            self.config.penalization,
            self.config.reward_at_goal_position,
        )

    def _get_goal_weights_at(self, goal_pos: np.ndarray):
        i = goal_pos[0][0]
        j = goal_pos[0][1]
        return self._make_full_grid_and_place_val_in(
            i, j, self.config.penalization, self.config.reward_at_goal_position
        )

if __name__ == "__main__":
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
        grid_world_env.render(action, reward)
