import numpy as np
import universal_successor_features.agents as a
import universal_successor_features.envs.grid_world as env
import torch


def test_choose_action():
    my_env = env.GridWorld()
    my_env.reset()

    agent = a.FeatureGoalAgent(env=my_env)

    obs, *_ = my_env.step(my_env.action_space.sample())

    action = agent.choose_action(
        agent_position_features=obs["agent_position_features"],
        list_of_goal_positions=[obs["goal_position"]],
        env_goal_position=obs["goal_position"],
        training=False,
    )
    assert action is not None
    assert isinstance(action, int)


def test_build_tensor_from_batch_of_np_arrays(batch_size=32):
    my_env = env.GridWorld()
    agent = a.FeatureGoalAgent(env=my_env)

    batch = []
    for i in range(batch_size):
        batch.append(np.random.rand(1, 100))
    batch = agent._build_tensor_from_batch_of_np_arrays(batch)

    assert tuple(batch.shape) == (batch_size, 100)
    assert batch.dtype == torch.float
