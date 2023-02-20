import universalSuccessorFeatures.envs.gridWorld as env
import numpy as np

my_env = env.GridWorld()

def test_goal_weights(goal = (0,0)):
    my_env.reset(goal = goal) 

    theoretical_goal_w = np.zeros(my_env.rows*my_env.columns)
    theoretical_goal_w[0] = 1

    comp = theoretical_goal_w == my_env.get_goal_weights()

    assert comp.all() 

def test_state_features(state = (0,0)):

    my_env.reset(start_position= state)
    theoretical_features = np.zeros(my_env.rows*my_env.columns)
    theoretical_features[0] = 1

    comp = theoretical_features == my_env.get_current_state_features()

    assert comp.all()