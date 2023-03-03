import universalSuccessorFeatures.epsilon as eps

def test_epsilon_is_correct_value(max = 0.5, min = 0.1 ):
    decaying_eps = eps.EpsilonExponentialDecay(max = max, min = min)
    assert decaying_eps.max == max and decaying_eps.min == min and decaying_eps.value == max

def test_epsilon_decays_properly(decay_factor = 0.5):
    decaying_eps = eps.EpsilonExponentialDecay(decay_factor = decay_factor, min = 0)
    for _ in range(4):
        decaying_eps.decay()
    
    assert decaying_eps.value == decay_factor**4
