import torch
import universalSuccessorFeatures.agents as dqn
import universalSuccessorFeatures.envs as env
import pytest
from math import isclose


def test_when_episode_starts_epsilon_defined():
    my_dqn = dqn.MultigoalDQNAgent()
    my_dqn.start_episode(episode = 0)
    assert my_dqn.current_epsilon is not None
    
@pytest.mark.parametrize(
    "epsilon_decay_type, epsilon_expected",
    [("none", 0.33), ("linear", 0.98), ("exponential", 0.7)]
)
def test_when_episode_starts_epsilon_is_correct_value(epsilon_decay_type, epsilon_expected):
    my_dqn = dqn.MultigoalDQNAgent(epsilon = epsilon_expected, epsilon_decay = {"type": epsilon_decay_type, "params": {"max": epsilon_expected}}) 
    my_dqn.start_episode(episode=0)
    assert my_dqn.current_epsilon == epsilon_expected and my_dqn.eps_min == 0.1 and my_dqn.epsilon_decay_type == epsilon_decay_type

@pytest.mark.parametrize(
    "decay, epsilon_expected",
    [
    ("none", 0.25),
    ("linear", 0.1),
    ("exponential", 0.99**10),
    ]
)
def test_decay_type(decay, epsilon_expected, num_episodes = 10):
    my_dqn = dqn.MultigoalDQNAgent(epsilon_decay = {"type":decay, "params": {"scheduled_episodes": num_episodes}})
    for ep in range(num_episodes):
        my_dqn.start_episode(episode=ep)
        my_dqn.end_episode()
    assert isclose(my_dqn.current_epsilon, epsilon_expected, abs_tol=0.0001)