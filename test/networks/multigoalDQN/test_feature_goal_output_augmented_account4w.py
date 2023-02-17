import torch
import universalSuccessorFeatures.networks.multigoalDQN as mdqn

state_size = 2
goal_size = 2
features_size = 100
num_actions = 4
batch_size = 32

my_dqn = mdqn.FeatureGoalAugumentedDQN(state_size = state_size, goal_size=goal_size, features_size = features_size, num_actions = num_actions)

def test_batch_of_input_for_network():
    feature_batch = torch.rand(batch_size,features_size)
    g_batch = torch.rand(batch_size,goal_size)
    
    output = my_dqn(feature_batch,g_batch)
    assert output is not None and output.shape == (batch_size,num_actions)
    
def test_single_input_of_network():
    feature = torch.rand(features_size).unsqueeze(0)
    g = torch.rand(2).unsqueeze(0)
    output = my_dqn(feature,g)

    assert output is not None and output.shape == (1,num_actions)