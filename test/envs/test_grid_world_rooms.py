import universal_successor_features.envs as env
import numpy as np
import pytest


@pytest.mark.parametrize(
    "i,j",
    [
        (4 - 1, 0),
        (4 - 1, 2),
        (4 - 1, 3),
        (4 - 1, 5),
        (4 - 1, 6),
        (4 - 1, 8),
    ],
)
def test_boundaries_forbidden_cells_going_down(i, j):
    my_env = env.RoomGridWorld()
    start_agent_position = np.array([[i, j]])
    my_env.reset(start_agent_position=start_agent_position)
    obs, *_ = my_env.step(env.Directions.DOWN)
    assert (obs["agent_position"] == start_agent_position).all()


@pytest.mark.parametrize(
    "i,j",
    [
        (4 + 1, 0),
        (4 + 1, 2),
        (4 + 1, 3),
        (4 + 1, 5),
        (4 + 1, 6),
        (4 + 1, 8),
    ],
)
def test_boundaries_forbidden_cells_going_up(i, j):
    my_env = env.RoomGridWorld()
    start_agent_position = np.array([[i, j]])
    my_env.reset(start_agent_position=start_agent_position)
    obs, *_ = my_env.step(env.Directions.UP)
    assert (obs["agent_position"] == start_agent_position).all()


@pytest.mark.parametrize(
    "i,j", [(0, 4 - 1), (2, 4 - 1), (3, 4 - 1), (5, 4 - 1), (6, 4 - 1), (8, 4 - 1)]
)
def test_boundaries_forbidden_cells_going_right(i, j):
    my_env = env.RoomGridWorld()
    start_agent_position = np.array([[i, j]])
    my_env.reset(start_agent_position=start_agent_position)
    obs, *_ = my_env.step(env.Directions.RIGHT)
    assert (obs["agent_position"] == start_agent_position).all()


@pytest.mark.parametrize(
    "i,j", [(0, 4 + 1), (2, 4 + 1), (3, 4 + 1), (5, 4 + 1), (6, 4 + 1), (8, 4 + 1)]
)
def test_boundaries_forbidden_cells_going_left(i, j):
    my_env = env.RoomGridWorld()
    start_agent_position = np.array([[i, j]])
    my_env.reset(start_agent_position=start_agent_position)
    obs, *_ = my_env.step(env.Directions.LEFT)
    assert (obs["agent_position"] == start_agent_position).all()


def test_no_boundaries_are_ever_crossed():
    my_env = env.RoomGridWorld()
    my_env.reset()
    pos = []
    for i in range(1000000):
        action = my_env.action_space.sample()
        obs, *_ = my_env.step(action)
        pos.append(tuple(obs["agent_position"][0]))

    for i in my_env.forbidden_cells:
        assert i not in pos
