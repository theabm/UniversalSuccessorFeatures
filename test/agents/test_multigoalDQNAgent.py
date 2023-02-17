import torch
import universalSuccessorFeatures.agents as dqn
import universalSuccessorFeatures.envs as env

my_dqn = dqn.MultigoalDQNAgent()
my_env = env.GridWorld()

def test_when_episode_starts_epsilon_defined():
    my_dqn.start_episode(episode = 0)
    assert my_dqn.current_epsilon is not None
    
def test_when_episode_starts_epsilon_is_correct_value(epsilon = 0.33):
    my_dqn = dqn.MultigoalDQNAgent(epsilon = epsilon, epsilon_decay = {"type": "linear"}) 
    assert my_dqn.current_epsilon == epsilon