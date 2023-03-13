import numpy as np
import universal_successor_features.agents as a
import universal_successor_features.networks as nn
import universal_successor_features.epsilon as eps
import universal_successor_features.envs.grid_world as env
import torch
import pytest

@pytest.mark.parametrize(
    "eps_type",
    [
        (eps.EpsilonConstant), 
        (eps.EpsilonLinearDecay), 
        (eps.EpsilonExponentialDecay),
    ]
)
def test_epsilon_is_instatiated(eps_type):
    my_env = env.GridWorld()
    my_dqn = a.StateGoalAgent(env = my_env, epsilon = {"cls" : eps_type})
    assert isinstance(my_dqn.epsilon, eps_type)
    assert my_dqn.epsilon.value is not None

def test_choose_action():
    my_env = env.GridWorld()
    my_env.reset()

    agent = a.StateGoalAgent(env = my_env)

    obs, *_ = my_env.step(my_env.action_space.sample())

    action = agent.choose_action(training=False,agent_position=obs["agent_position"], goal_position=obs["goal_position"])
    assert action is not None 
    assert isinstance(action, int) 
    
def test_build_tensor_from_batch_of_np_arrays(batch_size = 32):
    my_env = env.GridWorld()
    agent = a.StateGoalAgent(env = my_env)

    batch = []
    for i in range(batch_size):
        batch.append(np.random.rand(1,2))
    batch = agent._build_tensor_from_batch_of_np_arrays(batch)

    assert tuple(batch.shape) == (batch_size, 2) 
    assert batch.dtype == torch.float




    
