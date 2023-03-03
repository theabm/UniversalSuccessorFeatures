import universalSuccessorFeatures.epsilon as eps
from math import isclose

def test_epsilon_is_correct_value(max = 0.5, min = 0.1 ):
    decaying_eps = eps.EpsilonLinearDecay(max = max, min = min)
    assert decaying_eps.max == max and decaying_eps.min == min and decaying_eps.value == max

def test_epsilon_decays_properly(scheduled_episodes = 5, max = 0.6, min = 0.1):
    decaying_eps = eps.EpsilonLinearDecay(scheduled_episodes = scheduled_episodes, max = max, min = min)
    for ep in range(scheduled_episodes):
        decaying_eps.decay()
    
    assert isclose(decaying_eps.value, min)

    for ep in range(5):
        decaying_eps.decay()
    
    assert decaying_eps.value == min