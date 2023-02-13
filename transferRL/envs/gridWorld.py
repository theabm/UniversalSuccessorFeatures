import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np
import exputils as eu
import torch

class GridWorld(gym.Env):
    
    @staticmethod
    def default_config():
        return eu.AttrDict(
            length_y = 10,
            length_x = 10,
            nmax_steps = 1e6,
            )

    def __init__(self, config = None, **kwargs):

        super().__init__()
        
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        #length and width of the grid
        self.length_y = self.config.length_y 
        self.length_x = self.config.length_x
        
        #action space of the environment. Agent can only move up, down, left, right, or stay
        self.action_space = gym.spaces.Discrete(4)

        #observation space of the agent are the x,y coordinates 
        self.observation_space = gym.spaces.MultiDiscrete([self.length_x, self.length_y])

        self.reward_range = (-np.inf, 0)


        self.cur_step = None 
        self.nmax_steps = self.config.nmax_steps

    def _sample_xy_coordinates(self):
        x = np.random.choice(self.length_x)
        y = np.random.choice(self.length_y)
        
        return x,y
    
    def get_current_goal_coordinates(self):
        return np.array([self.goal_x, self.goal_y])

    def _goal_is_same_as_initial_position(self):
        return (self.goal_x,self.goal_y) == (self.agent_x,self.agent_y)

    def reset(self, seed = None, **kwargs):

        super().reset(seed=seed)
        np.random.seed(seed=seed)

        self.cur_step = 0

        #position of the agent and goal in x,y coordinates 
        self.agent_x, self.agent_y = self._sample_xy_coordinates()
        self.goal_x, self.goal_y = self._sample_xy_coordinates()

        while self._goal_is_same_as_initial_position():
            self.goal_x, self.goal_y = self._sample_xy_coordinates()
            
        info = {}

        return np.array([self.agent_x, self.agent_y]), info

    def step(self, action):

        self.cur_step += 1

        #Actions:                   o o o o o               o x o o o               
        #                           o x o o o   -GO "UP"-   o o o o o 
        #0 - up                     o o o o o               o o o o o
        #1 - down                   o o o o o               o o o o o
        #2 - right
        #3 - left
        if action == 0:
            self.agent_y += 1
        elif action == 1:
            self.agent_y -= 1
        elif action == 2:
            self.agent_x += 1
        elif action == 3:
            self.agent_x -= 1
        else: 
            raise ValueError("Unrecognized action. Agent can only perform the following actions: up:0, down:1, right:2, left:3.") 

        if self.agent_x < 0: self.agent_x = 0
        if self.agent_x > self.length_x - 1: self.agent_x = self.length_x - 1
        
        if self.agent_y < 0: self.agent_y = 0
        if self.agent_y > self.length_y - 1: self.agent_y = self.length_y - 1

        terminated = self.agent_x == self.goal_x and self.agent_y == self.goal_y

        truncated = self.cur_step >= self.nmax_steps

        reward = 0 if terminated else -1

        info = {}

        return np.array([self.agent_x, self.agent_y]), reward, terminated, truncated, info

    def render(self, action, reward):
        print(f"Action: {action}, position: ({self.agent_x},{self.agent_y}), reward: {reward}")

    def _make_grid_and_place_one_in(self,x,y):
        grid = np.zeros_like((self.length_x, self.length_y))
        grid[x][y] = 1.
        return grid.reshape(self.length_x*self.length_y)

    def get_state_features(self):
        state_features = self._make_grid_and_place_one_in(self.agent_x, self.agent_y)
        return state_features
        
    def get_goal_weights(self):
        goal_weights = self._make_grid_and_place_one_in(self.goal_x, self.goal_y)
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

