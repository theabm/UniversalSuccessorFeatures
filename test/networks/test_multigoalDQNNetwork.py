import torch
import universalSuccessorFeatures.networks.multigoalDQN.state_goal_output_like_paper as dqn

my_dqn = dqn.StateGoalPaperDQN()

def test_input_of_network():
    s = torch.rand(1,2)
    g = torch.rand(1,2)
    assert my_dqn(s,g) is not None

def test_output_size_of_network(N = 10, s_size = 2, g_size = 2, num_actions = 4):
    N = N
    s = torch.rand(N,s_size)
    g = torch.rand(N,g_size)
    output = my_dqn(s,g)
    assert output.shape[0] == N and output.shape[1] == num_actions