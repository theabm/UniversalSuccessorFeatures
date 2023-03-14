import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import exputils as eu
import random
import warnings

class GridWorld(gym.Env):
    
    @staticmethod
    def default_config():
        return eu.AttrDict(
            rows = 10,
            columns = 10,
            nmax_steps = 1e6,
            penalization = -0.1,
            reward_at_goal_position = 0,
            n_goals = 1
            )

    def __init__(self, config = None, **kwargs):

        super().__init__()
        
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        #length and width of the grid
        self.rows = self.config.rows 
        self.columns = self.config.columns
        
        #action space of the environment. Agent can only move up, down, left, right, or stay
        self.action_space = gym.spaces.Discrete(4)

        #observation space of the agent are the x,y coordinates 
        self.observation_space = gym.spaces.Dict(
            {
            "agent_position" : gym.spaces.MultiDiscrete(np.array([[self.rows, self.columns]])),
            "agent_position_features" : gym.spaces.MultiBinary([1,self.rows*self.columns]),
            "goal_position" : gym.spaces.MultiDiscrete(np.array([[self.rows, self.columns]])),
            "goal_weights" : gym.spaces.MultiBinary([1,self.rows*self.columns]), 
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
        self.goal_list_training, self.goal_list_testing = self._create_disjoint_goal_list_for_training_and_testing()

    def _sample_position_in_matrix(self):
        """Samples a row and a column from the predefined matrix dimensions.
           Returns a tuple of int.
        """
        i = np.random.choice(self.rows)
        j = np.random.choice(self.columns)
        
        return i,j
    
    def _create_disjoint_goal_list_for_training_and_testing(self):
        all_possible_goals = [np.array([[i,j]]) for i in range(self.rows) for j in range(self.columns)]
        goals = random.sample(all_possible_goals, 2*self.n_goals)
        return goals[:self.n_goals], goals[self.n_goals:]
    
    def sample_a_goal_position(self, training):
        if training:
            idx = random.randrange(len(self.goal_list_training))
            return self.goal_list_training[idx]
        else:
            idx = random.randrange(len(self.goal_list_testing))
            return self.goal_list_testing[idx]
    
    def sample_a_goal_position_from_list(self, goal_list):
        idx = random.randrange(len(goal_list))
        return goal_list[idx]

    def reset(self, start_agent_position : np.ndarray = None, goal_position : np.ndarray = None, seed = None):

        super().reset(seed=seed)

        self.cur_step = 0

        #position of the agent and goal in x,y coordinates 
        if start_agent_position is not None:
            self.agent_i = start_agent_position[0][0]
            self.agent_j = start_agent_position[0][1]
        else:
            self.agent_i, self.agent_j = self._sample_position_in_matrix()
            
        if goal_position is not None:
            self.goal_i = goal_position[0][0]
            self.goal_j = goal_position[0][1]

            #if same, give priority to goal that was set
            while (self.agent_i,self.agent_j)==(self.goal_i,self.goal_j):
                warnings.warn("Start position and goal position cannot be the same. Proceeding with different start position...")
                self.agent_i, self.agent_j = self._sample_position_in_matrix()
        else:
            self.goal_i, self.goal_j = self._sample_position_in_matrix()

            while (self.agent_i,self.agent_j)==(self.goal_i,self.goal_j):
                self.goal_i, self.goal_j = self._sample_position_in_matrix()
            
        #good sanity check
        if (self.agent_i,self.agent_j)==(self.goal_i,self.goal_j):
            raise ValueError("Start and Goal position cannot be the same.")

        position = np.array([[self.agent_i, self.agent_j]])
        position_features = self._get_current_agent_position_features()
        self.goal = np.array([[self.goal_i, self.goal_j]])
        self.goal_weights = self._get_current_goal_weights()

        info = {}
        obs = {
            "agent_position": position,
            "agent_position_features": position_features,
            "goal_position": self.goal,
            "goal_weights": self.goal_weights
        }

        return obs, info

    def step(self, action):
        
        self.cur_step += 1

        #Actions:                   o o o o o               o x o o o               
        #                           o x o o o   -GO "UP"-   o o o o o 
        #0 - up                     o o o o o               o o o o o
        #1 - down                   o o o o o               o o o o o
        #2 - right
        #3 - left
        if action == 0:
            self.agent_i -= 1
        elif action == 1:
            self.agent_i += 1
        elif action == 2:
            self.agent_j += 1
        elif action == 3:
            self.agent_j -= 1
        else: 
            raise ValueError("Unrecognized action. Agent can only perform the following actions: up:0, down:1, right:2, left:3.") 

        if self.agent_i < 0: self.agent_i = 0
        if self.agent_i > self.rows - 1: self.agent_i = self.rows - 1
        
        if self.agent_j < 0: self.agent_j = 0
        if self.agent_j > self.columns - 1: self.agent_j = self.columns - 1

        terminated = (self.agent_i, self.agent_j) == (self.goal_i, self.goal_j)

        truncated = self.cur_step >= self.nmax_steps

        reward = self.config.reward_at_goal_position if terminated else self.config.penalization

        position = np.array([[self.agent_i, self.agent_j]])
        position_features = self._get_current_agent_position_features()

        info = {}
        obs = {
            "agent_position": position,
            "agent_position_features": position_features,
            "goal_position": self.goal,
            "goal_weights": self.goal_weights
        }

        return obs, reward, terminated, truncated, info

    def render(self, action, reward):
        print(f"Action: {action}, position: ({self.agent_i},{self.agent_j}), reward: {reward}")

    def _make_grid_and_place_one_in(self,i,j):
        grd = np.zeros((self.rows, self.columns))
        grd[i][j] = 1.
        return grd.reshape((1,self.rows*self.columns))
    
    def get_agent_position_features_at(self,position: np.ndarray):
        i = position[0][0]
        j = position[0][1]
        return self._make_grid_and_place_one_in(i,j)

    def _get_current_agent_position_features(self):
        position_features = self._make_grid_and_place_one_in(self.agent_i, self.agent_j)
        return position_features
        
    def _get_current_goal_weights(self):
        task_weights = self._make_grid_and_place_one_in(self.goal_i, self.goal_j)
        return task_weights


if __name__ == '__main__':
    grid_world_env = GridWorld()
    # check_env(grid_world_env) #reset missing **kwargs argument but I dont want this functionality.
    print(grid_world_env.observation_space["agent_position_features"].shape[1])
    print(grid_world_env.observation_space["agent_position"].shape[1])
    l1, l2 = grid_world_env._create_disjoint_goal_list_for_training_and_testing()
    print(l1)
    print(l2)
    num_episodes = 1
    for _ in range(num_episodes):
        grid_world_env.reset()

        # steps
        for _ in range(100):
            action = grid_world_env.action_space.sample()
            obs, reward, terminated, truncated, info = grid_world_env.step(action)
            # grid_world_env.render(action, reward)

