import universalSuccessorFeatures.envs.gridWorld as env
import numpy as np

def test_action_space_size():
    my_env = env.GridWorld()
    my_env.action_space.shape == 4

def test_observation_space():
    my_env = env.GridWorld()
    obs, _ = my_env.reset()
    assert obs["agent_position"].shape == (1,2) and obs["agent_position"] is not None
    assert obs["agent_position_features"].shape == (1,100) and obs["agent_position_features"] is not None
    assert obs["goal_position"].shape == (1,2) and obs["goal_position"] is not None
    assert obs["goal_weights"].shape == (1,100) and obs["goal_weights"] is not None
    my_env.step(0)
    

def test_goal_weights(goal = np.array([0,0])):
    my_env = env.GridWorld()
    obs, *_ = my_env.reset(goal_position=goal) 

    theoretical_goal_w = np.zeros((1,my_env.rows*my_env.columns))
    theoretical_goal_w[0][0] = 1

    comp = theoretical_goal_w == obs["goal_weights"]

    assert comp.all() 

def test_position_features(start_position = np.array([0,0])):
    my_env = env.GridWorld()

    obs, *_ = my_env.reset(start_agent_position= start_position)
    theoretical_features = np.zeros((1,my_env.rows*my_env.columns))
    theoretical_features[0][0] = 1

    comp = theoretical_features == obs["agent_position_features"]

    assert comp.all()

def test_boundaries_going_up_left(start_position = np.array([0,0])):
    my_env = env.GridWorld()
    my_env.reset(start_agent_position=start_position)
    obs, *_ = my_env.step(0)
    assert (obs["agent_position"] == start_position).all()
    obs, *_ = my_env.step(3)
    assert (obs["agent_position"] == start_position).all()

def test_boundaries_going_down_right(start_position = np.array([9,9])):
    my_env = env.GridWorld()
    my_env.reset(start_agent_position=start_position)
    obs, *_ = my_env.step(1)
    assert (obs["agent_position"] == start_position).all()
    obs, *_ = my_env.step(2)
    assert (obs["agent_position"] == start_position).all()
