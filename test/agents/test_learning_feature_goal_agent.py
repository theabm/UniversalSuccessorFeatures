import universal_successor_features.envs as envs
import universal_successor_features.memory as mem
import universal_successor_features.agents as a
import universal_successor_features.networks as nn
import pytest
import exputils as eu
import torch
import agents.utils as u

# Ground truth values for the following configuration (discount = 0.5)
# o o o
# o o o
# o o g         #UP  #DOWN #RIGHT #LEFT
q_gt_g1_s1 = [0.0625, 0.125, 0.125, 0.0625]
q_gt_g1_s2 = [0.125, 0.250, 0.250, 0.0625]
q_gt_g1_s3 = [0.250, 0.500, 0.250, 0.125]
q_gt_g1_s4 = [0.0625, 0.250, 0.250, 0.125]
q_gt_g1_s5 = [0.125, 0.500, 0.500, 0.125]
q_gt_g1_s6 = [0.250, 1.000, 0.500, 0.250]
q_gt_g1_s7 = [0.125, 0.250, 0.500, 0.250]
q_gt_g1_s8 = [0.250, 0.500, 1.000, 0.250]
q_gt_g1_s9 = [0.0, 0.0, 0.0, 0.0]
# q_gt_g1_s9 = [0.500,1.000,1.000,0.500] #corresponding Q(goal state, actions)

# Ground truth values for the following configuration (discount = 0.5)
# o o o
# o o o
# g o o
q_gt_g2_s1 = [0.250, 0.500, 0.125, 0.250]
q_gt_g2_s2 = [0.125, 0.250, 0.0625, 0.250]
q_gt_g2_s3 = [0.0625, 0.125, 0.0625, 0.125]
q_gt_g2_s4 = [0.250, 1.000, 0.250, 0.500]
q_gt_g2_s5 = [0.125, 0.500, 0.125, 0.500]
q_gt_g2_s6 = [0.0625, 0.250, 0.125, 0.250]
q_gt_g2_s7 = [0.0, 0.0, 0.0, 0.0]
# q_gt_g2_s7 = [0.500,1.000,0.500,1.000]
q_gt_g2_s8 = [0.250, 0.500, 0.250, 1.000]
q_gt_g2_s9 = [0.125, 0.250, 0.250, 0.500]

q_ground_truth = torch.tensor(
    [
        q_gt_g1_s1,
        q_gt_g1_s2,
        q_gt_g1_s3,
        q_gt_g1_s4,
        q_gt_g1_s5,
        q_gt_g1_s6,
        q_gt_g1_s7,
        q_gt_g1_s8,
        q_gt_g1_s9,
        q_gt_g2_s1,
        q_gt_g2_s2,
        q_gt_g2_s3,
        q_gt_g2_s4,
        q_gt_g2_s5,
        q_gt_g2_s6,
        q_gt_g2_s7,
        q_gt_g2_s8,
        q_gt_g2_s9,
    ]
)


@pytest.mark.parametrize(
    "network, n_steps",
    [(nn.FeatureGoalPaperDQN, 500), (nn.FeatureGoalAugmentedDQN, 500)],
)
def test_training(network, n_steps, seed=0):
    if seed is not None:
        eu.misc.seed(seed)

    my_env = envs.GridWorld(
        rows=3, columns=3, penalization=0, reward_at_goal_position=1
    )

    agent = a.FeatureGoalAgent(
        env=my_env,
        epsilon={"value": 1.0},
        train_for_n_iterations=2,
        discount_factor=0.5,
        network={"cls": network},
    )

    cmp = u.test_training(
        agent, my_env, n_steps, q_ground_truth, u.step_feature_goal_agent
    )
    assert cmp


@pytest.mark.parametrize(
    "network, memory, n_steps",
    [
        (nn.FeatureGoalUSF, mem.ExperienceReplayMemory, 500),
        (nn.FeatureGoalUSF, mem.CombinedExperienceReplayMemory, 500),
        (nn.FeatureGoalUSF, mem.PrioritizedExperienceReplayMemory, 500),
    ],
)
def test_training_usf(network, memory, n_steps, seed=0):
    if seed is not None:
        eu.misc.seed(seed)

    my_env = envs.GridWorld(
        rows=3, columns=3, penalization=0, reward_at_goal_position=1
    )

    agent = a.FeatureGoalAgent(
        env=my_env,
        epsilon={"value": 1.0},
        train_for_n_iterations=2,
        discount_factor=0.5,
        network={"cls": network},
        loss_weight_psi=0.1,
        memory=eu.AttrDict(cls=memory, alpha=0.5, beta0=0.5, schedule_length=n_steps),
    )
    cmp = u.test_training(
        agent, my_env, n_steps, q_ground_truth, u.step_feature_goal_agent
    )
    assert cmp
