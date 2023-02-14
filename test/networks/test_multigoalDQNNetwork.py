import torch
import transferRL.networks.multigoalDQNNetwork as dqn

def test_input_of_network():
    my_dqn = dqn.MultigoalDQNNetwork()
    s = torch.rand(1,2)
    g = torch.rand(1,2)
    assert my_dqn(s,g) is not None

def test_output_size_of_network(N = 10, s_size = 2, g_size = 2, num_actions = 4):
    N = N
    my_dqn = dqn.MultigoalDQNNetwork()
    s = torch.rand(N,s_size)
    g = torch.rand(N,g_size)
    output = my_dqn(s,g)
    assert output.shape[0] == N and output.shape[1] == num_actions