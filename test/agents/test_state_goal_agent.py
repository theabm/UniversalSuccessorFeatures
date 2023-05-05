import universal_successor_features.agents as a
import universal_successor_features.envs.grid_world as env


def test_choose_action():
    my_env = env.GridWorld()
    my_env.reset()

    agent = a.StateGoalAgent(env=my_env)

    obs, *_ = my_env.step(my_env.action_space.sample())

    action = agent.choose_action(
        agent_position=obs["agent_position"],
        list_of_goal_positions=obs["goal_position"],
        env_goal_position=obs["goal_position"],
        training=False,
    )
    assert action is not None
    assert isinstance(action, int)
