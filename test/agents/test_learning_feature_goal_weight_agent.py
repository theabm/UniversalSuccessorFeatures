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
        (nn.FeatureGoalWeightKUSF, mem.ExperienceReplayMemory, 300),
        (nn.FeatureGoalWeightUSF, mem.ExperienceReplayMemory, 200),
    ],
)
def test_learned_q_function_matches_expected_q_function(
    network, memory, n_steps, seed=0
):
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

    agent = a.FeatureGoalWeightAgent(
        env=my_env,
        epsilon={"value": 1.0},
        train_for_n_iterations=2,
        discount_factor=0.5,
        network={"cls": network},
        memory=eu.AttrDict(cls=memory, alpha=0.5, beta0=0.5, schedule_length=n_steps),
    )

    cmp = u.test_training(
        agent,
        my_env,
        n_steps,
        u.q_ground_truth,
        exp.general_step_function,
        use_pos=False,
        use_weight=True,
    )
    assert cmp


# test_learned_q_funtion_matches_expected_q_function(nn.FeatureGoalWeightUSF, mem.ExperienceReplayMemory, 1000)
