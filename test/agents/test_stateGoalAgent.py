import numpy as np
import universalSuccessorFeatures.agents as a
import universalSuccessorFeatures.networks as nn
import universalSuccessorFeatures.epsilon as eps
import universalSuccessorFeatures.envs as envs
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

    action = agent.choose_action(training=False ,agent_position=obs["position"], goal_position=obs["goal"]) is not None
    assert action is not None 
    assert isinstance(action, int) 
    
    
