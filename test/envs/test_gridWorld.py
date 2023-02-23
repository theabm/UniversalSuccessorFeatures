import universalSuccessorFeatures.envs.gridWorld as env
import numpy as np

my_env = env.GridWorld()

def test_action_space_size():
    my_env.action_space.shape == 4

def test_observation_space():
    obs, _ = my_env.reset()
    assert obs["position"].shape == (1,2)
    assert obs["position_features"].shape == (1,100)
    assert obs["goal"].shape == (1,2)
    assert obs["goal_weights"].shape == (1,100)

def test_goal_weights(goal = np.array([0,0])):
    my_env.reset(goal_position= goal) 

    theoretical_goal_w = np.zeros((1,my_env.rows*my_env.columns))
    theoretical_goal_w[0][0] = 1
    print(my_env.goal_i,my_env.goal_j)
    print(theoretical_goal_w)
    print(my_env.get_current_goal_weights())

    comp = theoretical_goal_w == my_env.get_current_goal_weights()

    assert comp.all() 

def test_position_features(state = np.array([0,0])):

    my_env.reset(start_position= state)
    theoretical_features = np.zeros(my_env.rows*my_env.columns)
    theoretical_features[0] = 1

    comp = theoretical_features == my_env.get_current_position_features()

    assert comp.all()

def test_boundaries_going_up_left(start_position = np.array([0,0])):
    my_env.reset(start_position=start_position)
    obs, *_ = my_env.step(0)
    assert (obs["position"] == start_position).all()
    obs, *_ = my_env.step(3)
    assert (obs["position"] == start_position).all()

def test_boundaries_going_down_right(start_position = np.array([9,9])):
    my_env.reset(start_position= start_position)
    obs, *_ = my_env.step(1)
    assert (obs["position"] == start_position).all()
    obs, *_ = my_env.step(2)
    assert (obs["position"] == start_position).all()
