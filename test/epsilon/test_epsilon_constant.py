import universalSuccessorFeatures.epsilon as eps

def test_epsilon_is_correct_value(value = 0.3):
    lineareps = eps.EpsilonConstant(value = value) 
    assert lineareps.value == value

def test_epsilon_remains_constant(value = 0.3):
    lineareps = eps.EpsilonConstant(value = value) 
    for i in range(10):
        lineareps.decay()
    
    assert lineareps.value == value