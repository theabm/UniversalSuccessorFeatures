import universal_successor_features.envs as env
import numpy as np
import pytest


def test_setup_is_as_expected(
    rows=5, columns=8, nmax_steps=100, penalization=3, reward_at_goal_position=4
):
    my_env = env.GridWorld(
        rows=rows,
        columns=columns,
        nmax_steps=nmax_steps,
        penalization=penalization,
        reward_at_goal_position=reward_at_goal_position,
        n_goals=1,
    )
    assert my_env.rows == rows
    assert my_env.columns == columns
    assert my_env.nmax_steps == nmax_steps
    assert my_env.config.penalization == penalization
    assert my_env.config.reward_at_goal_position == reward_at_goal_position


def test_action_space_size():
    my_env = env.GridWorld()
    assert my_env.action_space.n == 4


def test_observation_space():
    my_env = env.GridWorld()
    obs, _ = my_env.reset()
    assert obs["agent_position"].shape == (1, 2)
    assert obs["agent_position_features"].shape == (1, 81)
    assert obs["goal_position"].shape == (1, 2)
    assert obs["goal_weights"].shape == (1, 81)
    my_env.step(0)


@pytest.mark.parametrize(
    "i,j",
    [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ],
)
def test_goal_weights(i, j):
    my_env = env.GridWorld(
        rows=3, columns=3, penalization=-5, reward_at_goal_position=100, n_goals=1
    )
    goal_position = np.array([[i, j]])
    obs, *_ = my_env.reset(goal_position=goal_position)

    theoretical_goal_w = np.full(
        (1, my_env.rows * my_env.columns), my_env.config.penalization
    )
    idx = my_env.rows * i + j
    theoretical_goal_w[0][idx] = my_env.config.reward_at_goal_position

    comp = theoretical_goal_w == obs["goal_weights"]

    assert comp.all()


@pytest.mark.parametrize(
    "i,j",
    [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ],
)
def test_position_features(i, j):
    my_env = env.GridWorld(
        rows=3, columns=3, penalization=-5, reward_at_goal_position=100, n_goals=1
    )
    start_agent_position = np.array([[i, j]])
    obs, *_ = my_env.reset(start_agent_position=start_agent_position)

    theoretical_features = np.zeros((1, my_env.rows * my_env.columns))
    idx = my_env.rows * i + j
    theoretical_features[0][idx] = 1

    comp = theoretical_features == obs["agent_position_features"]

    assert comp.all()


def test_reward_is_dot_product_of_feature_and_weight():
    my_env = env.GridWorld(
        rows=3, columns=3, penalization=-5, reward_at_goal_position=100, n_goals=1
    )
    # set the goal at 0,0 -> reward for going here should be 100
    # everywhere else it will be -5
    obs, *_ = my_env.reset()
    for i in range(100000):
        action = my_env.action_space.sample()
        obs, reward, terminated, *_ = my_env.step(action)
        dot_prod = np.sum(obs["agent_position_features"] * obs["goal_weights"])
        assert dot_prod == reward


@pytest.mark.parametrize(
    "i,j",
    [
        (0, 0),
        (0, 1),
        (0, 2),
    ],
)
def test_boundaries_going_up(i, j):
    my_env = env.GridWorld(rows=3, columns=3, n_goals=1)
    start_agent_position = np.array([[i, j]])
    my_env.reset(start_agent_position=start_agent_position)
    obs, *_ = my_env.step(env.Directions.UP)
    assert (obs["agent_position"] == start_agent_position).all()


@pytest.mark.parametrize(
    "i,j",
    [
        (0, 0),
        (1, 0),
        (2, 0),
    ],
)
def test_boundaries_going_left(i, j):
    my_env = env.GridWorld(rows=3, columns=3, n_goals=1)
    start_agent_position = np.array([[i, j]])
    my_env.reset(start_agent_position=start_agent_position)
    obs, *_ = my_env.step(env.Directions.LEFT)
    assert (obs["agent_position"] == start_agent_position).all()


@pytest.mark.parametrize(
    "i,j",
    [
        (2, 0),
        (2, 1),
        (2, 2),
    ],
)
def test_boundaries_going_down(i, j):
    my_env = env.GridWorld(rows=3, columns=3, n_goals=1)
    start_agent_position = np.array([[i, j]])
    my_env.reset(start_agent_position=start_agent_position)
    obs, *_ = my_env.step(env.Directions.DOWN)
    assert (obs["agent_position"] == start_agent_position).all()


@pytest.mark.parametrize(
    "i,j",
    [
        (0, 2),
        (1, 2),
        (2, 2),
    ],
)
def test_boundaries_going_right(i, j):
    my_env = env.GridWorld(rows=3, columns=3, n_goals=1)
    start_agent_position = np.array([[i, j]])
    my_env.reset(start_agent_position=start_agent_position)
    obs, *_ = my_env.step(env.Directions.RIGHT)
    assert (obs["agent_position"] == start_agent_position).all()


@pytest.mark.parametrize(
    "i,j",
    [
        (0, 0),(0, 1),(0, 2),
        (1, 0),(1, 1),(1, 2),
        (2, 0),(2, 1),(2, 2)
    ],
)
def test_rbf_vector_is_correct(i,j):
    agent_position = np.array([[i, j]])
    my_env = env.GridWorld(
        rows=3, columns=3, n_goals=1,
        rbf_points_in_x_direction = 3, rbf_points_in_y_direction = 3
    )
    rbf_vector = []
    for k in range(my_env.rows):
        for m in range(my_env.columns):
            exponent = np.sum(
                (agent_position - np.array([k, m]))
                * (agent_position - np.array([k, m]))
            )
            component = np.exp(-1 * exponent / my_env.sigma)
            rbf_vector.append(component)

    rbf_vector = np.array(rbf_vector)

    obs, _ = my_env.reset(start_agent_position=agent_position)

    rbf_vector_env = obs["agent_position_features_rbf"]
    assert rbf_vector_env.shape == (1, 9)

    rbf_vector_env = rbf_vector_env.reshape((9,))

    assert (rbf_vector_env == rbf_vector).all()
    assert rbf_vector_env[i*3+j] == 1.0

