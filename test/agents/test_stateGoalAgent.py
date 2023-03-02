import numpy as np
import universalSuccessorFeatures.agents as a
import universalSuccessorFeatures.networks as nn
import universalSuccessorFeatures.epsilon as eps
import universalSuccessorFeatures.envs.gridWorld as env
import random
import torch
import pytest
from math import isclose

@pytest.mark.parametrize(
    "eps_type",
    [
        (eps.EpsilonConstant), 
        (eps.EpsilonLinearDecay), 
        (eps.EpsilonExponentialDecay),
    ]
)
def test_epsilon_is_instatiated(eps_type):
    my_dqn = a.StateGoalAgent(epsilon = {"cls" : eps_type})
    assert isinstance(my_dqn.epsilon, eps_type)
    assert my_dqn.epsilon.value is not None

def test_choose_action():
    my_env = env.GridWorld()
    my_env.reset()

    agent = a.StateGoalAgent()

    obs, *_ = my_env.step(my_env.action_space.sample())

    action = agent.choose_action(training=False,agent_position=obs["agent_position"], goal_position=obs["goal_position"])
    assert action is not None 
    assert isinstance(action, int) 
    
def test_build_tensor_from_batch_of_np_arrays(batch_size = 32):
    agent = a.StateGoalAgent()

    batch = []
    for i in range(batch_size):
        batch.append(np.random.rand(1,2))
    batch = agent._build_tensor_from_batch_of_np_arrays(batch)

    assert tuple(batch.shape) == (batch_size, 2) 
    assert batch.dtype == torch.float

def test_few_rounds_of_training(num_episodes = 20):
    my_env = env.GridWorld()
    agent = a.StateGoalAgent()

    step = 0
    for episode in range(num_episodes):
        obs, _ = my_env.reset()
        agent.start_episode(episode = episode)
        while True:
            action = agent.choose_action(agent_position = obs["agent_position"], goal_position = obs["goal_position"], training= False)
            next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

            transition = (obs["agent_position"], obs["goal_position"], action, reward, next_obs["agent_position"], terminated, truncated)

            agent.train(transition=transition, step = step)

            if terminated or truncated:
                break

            obs = next_obs
            step += 1
        agent.end_episode()



    
