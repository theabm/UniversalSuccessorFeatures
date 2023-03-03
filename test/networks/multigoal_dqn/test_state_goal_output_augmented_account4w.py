import torch
import universal_successor_features.networks.multigoal_dqn as mdqn

state_size = 2
goal_size = 2
features_size = 100
num_actions = 4
batch_size = 32

my_dqn = mdqn.StateGoalAugmentedDQN(state_size=state_size, goal_size=goal_size, features_size=features_size,num_actions=num_actions)

def test_batch_of_input_for_network():
    s_batch = torch.rand(batch_size,state_size)
    g_batch = torch.rand(batch_size,goal_size)
    
    output = my_dqn(s_batch,g_batch)
    assert output is not None and output.shape == (batch_size,num_actions)
    
def test_single_input_of_network():
    s = torch.rand(state_size).unsqueeze(0)
    g = torch.rand(goal_size).unsqueeze(0)
    output = my_dqn(s,g)

    assert output is not None and output.shape == (1,num_actions)