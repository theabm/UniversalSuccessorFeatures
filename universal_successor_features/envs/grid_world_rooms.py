import numpy as np
import exputils as eu
import random
import pickle
from universal_successor_features.envs.grid_world import GridWorld


class RoomGridWorld(GridWorld):
    @staticmethod
    def default_config():
        return eu.AttrDict(
            rows=9,
            columns=9,
            nmax_steps=31,
            penalization=-0.1,
            reward_at_goal_position=0,
            one_hot_weight=1,
        )

    def __init__(self, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, RoomGridWorld.default_config())
        self.config.rows = 9
        self.config.columns = 9

        super().__init__(config=self.config)

        assert self.config.rows == 9
        assert self.config.columns == 9

        self.forbidden_cells = [
            (4, 0),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 8),
            (0, 4),
            (2, 4),
            (3, 4),
            (5, 4),
            (6, 4),
            (8, 4),
        ]

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
        while (i, j) in self.forbidden_cells:
            i = np.random.choice(self.rows)
            j = np.random.choice(self.columns)

        return i, j

    def _create_three_disjoint_goal_lists(self):
        """Create three disjoint set of goals.
        Primary/Source goals are the initial goals to train on.

        Secondary/Target goals are the goals trained on in second phase

        Tertiary/Evaluation goals are goals we evaluate on in the first
        phase
        """
        possible_goals_room_0 = [np.array([[i, j]]) for i in range(4) for j in range(4)]
        possible_goals_room_1 = [
            np.array([[i, j]]) for i in range(4) for j in range(5, 9)
        ]
        possible_goals_room_2 = [
            np.array([[i, j]]) for i in range(5, 9) for j in range(4)
        ]
        possible_goals_room_3 = [
            np.array([[i, j]]) for i in range(5, 9) for j in range(5, 9)
        ]

        goals_room0 = random.sample(possible_goals_room_0, 3 * 3)
        goals_room1 = random.sample(possible_goals_room_1, 3 * 3)
        goals_room2 = random.sample(possible_goals_room_2, 3 * 3)
        goals_room3 = random.sample(possible_goals_room_3, 3 * 3)

        source_tasks = (
            goals_room0[:3] + goals_room1[:3] + goals_room2[:3] + goals_room3[:3]
        )
        target_tasks = (
            goals_room0[3:6] + goals_room1[3:6] + goals_room2[3:6] + goals_room3[3:6]
        )
        eval_tasks = (
            goals_room0[6:9] + goals_room1[6:9] + goals_room2[6:9] + goals_room3[6:9]
        )

        return source_tasks, target_tasks, eval_tasks

    def check_boundary_conditions(self):
        """
        Check agent cannot violate physical constraints of the grid
        world.
        """
        super().check_boundary_conditions()
        if (self.agent_i, self.agent_j) in self.forbidden_cells:
            self.agent_i = self.old_agent_i
            self.agent_j = self.old_agent_j

    def step(self, action):
        self.old_agent_i = self.agent_i
        self.old_agent_j = self.agent_j

        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, info

    def save(self):
        self.config.forbidden_cells = self.forbidden_cells
        super().save()

    @classmethod
    def load_from_checkpoint(cls, filename):
        with open(filename, "rb") as fp:
            config = pickle.load(fp)

        env = cls(config)
        env.goal_list_source_tasks = config["goal_list_source_tasks"]
        env.goal_list_target_tasks = config["goal_list_target_tasks"]
        env.goal_list_evaluation_tasks = config["goal_list_evaluation_tasks"]
        env.forbidden_cells = config["forbidden_cells"]

        return env


if __name__ == "__main__":
    grid_world_env = RoomGridWorld()
    grid_world_env.reset()

    for _ in range(100):
        action = grid_world_env.action_space.sample()
        obs, reward, terminated, truncated, info = grid_world_env.step(action)
        grid_world_env.render()

    grid_world_env.save()
    new_grid_world_env = RoomGridWorld.load_from_checkpoint("./env_config.cfg")

    assert all(
        [
            (goal1 == goal2).all()
            for goal1, goal2 in zip(
                grid_world_env.goal_list_source_tasks,
                new_grid_world_env.goal_list_source_tasks,
            )
        ]
    )
    assert all(
        [
            (goal1 == goal2).all()
            for goal1, goal2 in zip(
                grid_world_env.goal_list_target_tasks,
                new_grid_world_env.goal_list_target_tasks,
            )
        ]
    )
    assert all(
        [
            (goal1 == goal2).all()
            for goal1, goal2 in zip(
                grid_world_env.goal_list_evaluation_tasks,
                new_grid_world_env.goal_list_evaluation_tasks,
            )
        ]
    )
    assert grid_world_env.forbidden_cells == new_grid_world_env.forbidden_cells
