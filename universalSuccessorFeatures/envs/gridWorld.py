import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import exputils as eu
import random

class GridWorld(gym.Env):
    
    @staticmethod
    def default_config():
        return eu.AttrDict(
            rows = 10,
            columns = 10,
            nmax_steps = 1e6,
            penalization = -0.1,
            reward_at_goal_state = 0
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
        self.observation_space = gym.spaces.MultiDiscrete([self.rows, self.columns])

        self.reward_range = (-np.inf, 0)

        self.agent_i = None
        self.agent_j = None

        self.goal_i = None
        self.goal_j = None

        self.cur_step = None 
        self.nmax_steps = self.config.nmax_steps

    def _sample_position_in_matrix(self):
        """Samples a row and a column from the predefined matrix dimensions.
           Returns a tuple of int (row, col).
        """
        i = np.random.choice(self.rows)
        j = np.random.choice(self.columns)
        
        return i,j
    
    def sample_a_goal_position(self, goal_list : list = None):
        """Takes as input a list of goals and samples a goal position. If None, the sampling is done uniformly
           rows and columns. 
           Returns a tuple of int.
        """
        if goal_list is None:
            return self._sample_position_in_matrix()
        else:
            idx = random.randrange(len(goal_list))
            return goal_list[idx]

    def get_current_goal_position_in_matrix(self):
        return np.array([self.goal_i, self.goal_j])

    def get_current_agent_position_in_matrix(self):
        return np.array([self.agent_i, self.agent_j])

    def reset(self, start_position = None, goal = None, seed = None, **kwargs):

        super().reset(seed=seed)

        self.cur_step = 0

        #position of the agent and goal in x,y coordinates 
        if start_position is None:
            self.agent_i, self.agent_j = self._sample_position_in_matrix()
        else:
            self.agent_i, self.agent_j = start_position

        if goal is None:
            self.goal_i, self.goal_j = self._sample_position_in_matrix()

            while (self.agent_i,self.agent_j)==(self.goal_i,self.goal_j):
                self.goal_i, self.goal_j = self._sample_position_in_matrix()
        else:
            self.goal_i, self.goal_j = goal
            
        if (self.agent_i,self.agent_j)==(self.goal_i,self.goal_j):
            raise ValueError("Start and Goal position cannot be the same.")

        info = {}

        return np.array([self.agent_i, self.agent_j]), info

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

        reward = self.config.reward_at_goal_state if terminated else self.config.penalization

        info = {}

        return np.array([self.agent_i, self.agent_j]), reward, terminated, truncated, info

    def render(self, action, reward):
        print(f"Action: {action}, position: ({self.agent_i},{self.agent_j}), reward: {reward}")

    def _make_grid_and_place_one_in(self,i,j):
        grd = np.zeros((self.rows, self.columns))
        grd[i][j] = 1.
        return grd.reshape(self.rows*self.columns)
    
    #maybe add method which takes as input the state you want the features for

    def get_current_state_features(self):
        state_features = self._make_grid_and_place_one_in(self.agent_i, self.agent_j)
        return state_features
        
    def get_goal_weights(self):
        goal_weights = self._make_grid_and_place_one_in(self.goal_i, self.goal_j)
        return goal_weights


if __name__ == '__main__':
    grid_world_env = GridWorld()
    check_env(grid_world_env)

    num_episodes = 1
    for _ in range(num_episodes):
        grid_world_env.reset()

        # steps
        for _ in range(100):
            action = grid_world_env.action_space.sample()
            obs, reward, terminated, truncated, info = grid_world_env.step(action)
            grid_world_env.render(action, reward)

