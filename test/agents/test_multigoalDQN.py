import torch
import transferRL.agents.multigoalDQN as dqn
import transferRL.envs.gridWorld as env

my_dqn = dqn.MultigoalDQN()
my_env = env.GridWorld()


def test_when_episode_starts_epsilon_is_not_none(episode = 0):
    my_dqn.start_episode(episode = episode)
    assert my_dqn.current_epsilon is not None and my_dqn.current_epsilon == 0.25 and my_dqn.current_episode == episode