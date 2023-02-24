import numpy as np
import universalSuccessorFeatures.agents as a
import universalSuccessorFeatures.networks as nn
import universalSuccessorFeatures.epsilon as eps
import universalSuccessorFeatures.envs as envs
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
    my_env = envs.GridWorld()
    my_env.reset()

    agent = a.StateGoalAgent()

    obs, *_ = my_env.step(action = my_env.action_space.sample())

    action = agent.choose_action(training=False ,agent_position=obs["agent_position"], goal_position=obs["goal_position"]) is not None
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


    
