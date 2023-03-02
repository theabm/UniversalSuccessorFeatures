import numpy as np
import universalSuccessorFeatures.agents as a
import universalSuccessorFeatures.envs.gridWorld as env
import torch
import pytest

def test_choose_action():
    my_env = env.GridWorld()
    my_env.reset()

    agent = a.FeatureGoalAgent()

    obs, *_ = my_env.step(my_env.action_space.sample())

    action = agent.choose_action(training=False,agent_position_features=obs["agent_position_features"], goal_position=obs["goal_position"])
    assert action is not None 
    assert isinstance(action, int) 
    
def test_build_tensor_from_batch_of_np_arrays(batch_size = 32):
    agent = a.FeatureGoalAgent()

    batch = []
    for i in range(batch_size):
        batch.append(np.random.rand(1,100))
    batch = agent._build_tensor_from_batch_of_np_arrays(batch)

    assert tuple(batch.shape) == (batch_size, 100) 
    assert batch.dtype == torch.float

def test_few_rounds_of_training(num_episodes = 10):
    my_env = env.GridWorld()
    agent = a.FeatureGoalAgent()

    step = 0
    for episode in range(num_episodes):
        agent.start_episode(episode = episode)
        obs, _ = my_env.reset()
        while True:
            action = agent.choose_action(agent_position_features = obs["agent_position_features"], goal_position = obs["goal_position"], training= False)
            next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

            transition = (obs["agent_position_features"], obs["goal_position"], action, reward, next_obs["agent_position_features"], terminated, truncated)

            agent.train(transition=transition, step = step)

            if terminated or truncated:
                agent.end_episode()
                break

            obs = next_obs
            step += 1



    
