import universal_successor_features.envs as envs
import universal_successor_features.memory as mem
import universal_successor_features.agents as a
import universal_successor_features.networks as nn
import universal_successor_features.exp as exp
import pytest
import exputils as eu
import test.agents.utils as u  # for pytest

# import utils as u # for python


@pytest.mark.parametrize(
    "network, memory, n_steps",
    [
        (nn.StateGoalPaperDQN, mem.ExperienceReplayMemory, 700),
        (nn.StateGoalAugmentedDQN, mem.ExperienceReplayMemory, 500),
        (nn.StateGoalUSF, mem.ExperienceReplayMemory, 1000),
        (nn.StateGoalUSF, mem.CombinedExperienceReplayMemory, 1000),
        (nn.StateGoalUSF, mem.PrioritizedExperienceReplayMemory, 1000),
    ],
)
def test_training(network, memory, n_steps, seed=0):
    if seed is not None:
        eu.misc.seed(seed)

    my_env = envs.GridWorld(
        rows=3,
        columns=3,
        penalization=0,
        reward_at_goal_position=1,
        nmax_steps=31,
        n_goals=1,
    )

    agent = a.StateGoalAgent(
        env=my_env,
        epsilon={"value": 1.0},
        train_for_n_iterations=2,
        discount_factor=0.5,
        network={"cls": network},
        memory=eu.AttrDict(cls=memory, alpha=0.3, beta0=0.2, schedule_length=n_steps),
    )

    cmp = u.test_training(
        agent,
        my_env,
        n_steps,
        u.q_ground_truth,
        exp.general_step_function,
        use_pos=True,
        use_weight=False,
    )
    assert cmp
