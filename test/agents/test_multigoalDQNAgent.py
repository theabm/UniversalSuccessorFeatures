import numpy as np
import universalSuccessorFeatures.agents as a
import universalSuccessorFeatures.networks as nn
import pytest
from math import isclose


def test_when_episode_starts_epsilon_defined():
    my_dqn = a.StateGoalAgent()
    my_dqn.start_episode(episode = 0)
    assert my_dqn.current_epsilon is not None
    
@pytest.mark.parametrize(
    "epsilon_decay_type, epsilon_expected",
    [
        ("none", 0.33),
        ("linear", 0.98),
        ("exponential", 0.7)
    ]
)
def test_when_episode_starts_epsilon_is_correct_value(epsilon_decay_type, epsilon_expected):
    my_dqn = a.StateGoalAgent(epsilon = epsilon_expected, epsilon_decay = {"type": epsilon_decay_type, "params": {"max": epsilon_expected}}) 
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
    my_dqn = a.StateGoalAgent(epsilon_decay = {"type":decay, "params": {"scheduled_episodes": num_episodes}})
    for ep in range(num_episodes):
        my_dqn.start_episode(episode=ep)
        my_dqn.end_episode()
    assert isclose(my_dqn.current_epsilon, epsilon_expected, abs_tol=0.0001)

def test_coherency_of_types():
    my_dqn = a.StateGoalAgent(network = {"cls":nn.StateGoalPaperDQN})
    state = np.array([2,2])
    goal = np.array([5,5])
    assert my_dqn.policy_net(**my_dqn._make_compatible_with_nn(s = state, g = goal)) is not None
    assert my_dqn.choose_action(s = state, g = goal, purpose="testing") is not None
    
    
def test_choose_action():
    ##Test if I can use DQN agent with MDQN network
    my_dqn = a.StateGoalAgent(network = {"cls":nn.StateGoalPaperDQN})
    assert my_dqn.choose_action(s = np.random.rand(2), g = np.random.rand(2), purpose="testing") is not None
    my_dqn = a.StateGoalAgent(network = {"cls":nn.FeatureGoalPaperDQN})
    assert my_dqn.choose_action(phi_s = np.random.rand(100), g = np.random.rand(2), purpose="testing") is not None

    ## Testing if I can use DQN agent with USF network
    my_dqn = a.StateGoalAgent(network = {"cls":nn.StateGoalUSF})
    assert my_dqn.choose_action(s = np.random.rand(2), g = np.random.rand(2), purpose="testing") is not None
    my_dqn = a.StateGoalAgent(network = {"cls":nn.StateGoalWeightUSF})
    assert my_dqn.choose_action(s = np.random.rand(2), g = np.random.rand(2), w = np.random.rand(100), purpose="testing") is not None
    my_dqn = a.StateGoalAgent(network = {"cls":nn.FeatureGoalWeightUSF})
    assert my_dqn.choose_action(phi_s = np.random.rand(100), g = np.random.rand(2), w = np.random.rand(100), purpose="testing") is not None
    my_dqn = a.StateGoalAgent(network = {"cls":nn.FeatureGoalUSF})
    assert my_dqn.choose_action(phi_s = np.random.rand(100), g = np.random.rand(2), purpose="testing") is not None

def test_choose_action_can_take_all_arguments():
    ##Test if I can use DQN agent with MDQN network
    my_dqn = a.StateGoalAgent(network = {"cls":nn.StateGoalPaperDQN})
    assert my_dqn.choose_action(s = np.random.rand(2), phi_s = np.random.rand(100), g = np.random.rand(2), w = np.random.rand(100), purpose="testing") is not None
    my_dqn = a.StateGoalAgent(network = {"cls":nn.FeatureGoalPaperDQN})
    assert my_dqn.choose_action(s = np.random.rand(2), phi_s = np.random.rand(100), g = np.random.rand(2), w = np.random.rand(100), purpose="testing") is not None

    ## Testing if I can use DQN agent with USF network
    my_dqn = a.StateGoalAgent(network = {"cls":nn.StateGoalUSF})
    assert my_dqn.choose_action(s = np.random.rand(2), phi_s = np.random.rand(100), g = np.random.rand(2), w = np.random.rand(100), purpose="testing") is not None
    my_dqn = a.StateGoalAgent(network = {"cls":nn.StateGoalWeightUSF})
    assert my_dqn.choose_action(s = np.random.rand(2), phi_s = np.random.rand(100), g = np.random.rand(2), w = np.random.rand(100), purpose="testing") is not None
    my_dqn = a.StateGoalAgent(network = {"cls":nn.FeatureGoalWeightUSF})
    assert my_dqn.choose_action(s = np.random.rand(2), phi_s = np.random.rand(100), g = np.random.rand(2), w = np.random.rand(100), purpose="testing") is not None
    my_dqn = a.StateGoalAgent(network = {"cls":nn.FeatureGoalUSF})
    assert my_dqn.choose_action(s = np.random.rand(2), phi_s = np.random.rand(100), g = np.random.rand(2), w = np.random.rand(100), purpose="testing") is not None