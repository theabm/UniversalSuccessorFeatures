import universal_successor_features.agents as a
import universal_successor_features.envs.grid_world as env


def test_choose_action():
    my_env = env.GridWorld()
    agent = a.StateGoalAgent(env=my_env)

    obs, *_ = my_env.reset()

    action = agent.choose_action(
        obs=obs,
        list_of_goal_positions=[obs["goal_position"]],
        training=False,
    )
    assert action is not None
    assert isinstance(action, int)
