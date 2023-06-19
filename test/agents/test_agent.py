import numpy as np
import exputils as eu
import universal_successor_features.agents as a
import universal_successor_features.envs.grid_world as env
import universal_successor_features.networks as nn
import universal_successor_features.epsilon as eps
import universal_successor_features.memory as mem
import torch
import pytest
from test.agents.stub_classes import StubFeatureGoalWeightNetwork
from universal_successor_features.agents.base_agent import FullTransition, Experiences
import copy


@pytest.mark.parametrize(
    "agent_type, network_type",
    [
        (a.FeatureGoalAgent, nn.FeatureGoalPaperDQN),
        (a.FeatureGoalAgent, nn.FeatureGoalAugmentedDQN),
        (a.FeatureGoalAgent, nn.FeatureGoalUSF),
        (a.FeatureGoalWeightAgent, nn.FeatureGoalWeightUSF),
        (a.StateGoalAgent, nn.StateGoalPaperDQN),
        (a.StateGoalAgent, nn.StateGoalAugmentedDQN),
        (a.StateGoalAgent, nn.StateGoalUSF),
        (a.StateGoalWeightAgent, nn.StateGoalWeightUSF),
    ],
)
def test_default_configuration(agent_type, network_type):
    my_env = env.GridWorld(rows=3, columns=3, n_goals=1)
    agent = agent_type(env=my_env, network=eu.AttrDict(cls=network_type))

    # Assert default config is as expected

    expected_config = eu.AttrDict(
        device="cuda",
        discount_factor=0.99,
        batch_size=32,
        learning_rate=5e-4,
        train_for_n_iterations=1,
        train_every_n_steps=1,
        loss_weight_q=1.0,
        loss_weight_psi=0.01,
        loss_weight_phi=0.00,
        network=eu.AttrDict(
            cls=network_type,
            use_gdtuo=False,
            optimizer=torch.optim.Adam,
            state_size=2,
            goal_size=2,
            features_size=9,
            num_actions=4,
            rbf_size=81,
        ),
        target_network_update=eu.AttrDict(
            rule="hard",
            every_n_steps=10,
            alpha=0.0,
        ),
        epsilon=eu.AttrDict(
            cls=eps.EpsilonConstant,
        ),
        memory=eu.AttrDict(
            cls=mem.ExperienceReplayMemory,
            alpha=None,
            beta0=None,
            schedule_length=None,
        ),
        log=eu.AttrDict(
            loss_per_step=True,
            epsilon_per_episode=True,
            log_name_epsilon="epsilon_per_episode",
            log_name_loss="loss_per_step",
        ),
        save=eu.AttrDict(extension=".pt"),
    )
    print(agent.config, "\n", expected_config)

    assert agent.config == expected_config


@pytest.mark.parametrize(
    "agent_type, network_type",
    [
        (a.FeatureGoalAgent, nn.FeatureGoalPaperDQN),
        (a.FeatureGoalAgent, nn.FeatureGoalAugmentedDQN),
        (a.FeatureGoalAgent, nn.FeatureGoalUSF),
        (a.FeatureGoalWeightAgent, nn.FeatureGoalWeightUSF),
        (a.StateGoalAgent, nn.StateGoalPaperDQN),
        (a.StateGoalAgent, nn.StateGoalAugmentedDQN),
        (a.StateGoalAgent, nn.StateGoalUSF),
        (a.StateGoalWeightAgent, nn.StateGoalWeightUSF),
    ],
)
def test_agent_matches_custom_config(agent_type, network_type):
    my_env = env.GridWorld(rows=3, columns=3, n_goals=1)

    expected_config = eu.AttrDict(
        device="cuda",
        discount_factor=0.1,
        batch_size=2,
        learning_rate=3e-4,
        train_for_n_iterations=4,
        train_every_n_steps=6,
        loss_weight_q=3,
        loss_weight_psi=4.01,
        loss_weight_phi=-3.50,
        network=eu.AttrDict(
            cls=network_type,
            use_gdtuo=False,
            optimizer=torch.optim.Adam,
            state_size=2,
            goal_size=2,
            features_size=9,
            num_actions=4,
            rbf_size=81,
        ),
        target_network_update=eu.AttrDict(
            rule="hard",
            every_n_steps=20,
            alpha=0.0,
        ),
        epsilon=eu.AttrDict(
            cls=eps.EpsilonConstant,
        ),
        memory=eu.AttrDict(
            cls=mem.ExperienceReplayMemory,
            alpha=None,
            beta0=None,
            schedule_length=None,
        ),
        log=eu.AttrDict(
            loss_per_step=True,
            epsilon_per_episode=True,
            log_name_epsilon="epsilon_per_episode",
            log_name_loss="loss_per_step",
        ),
        save=eu.AttrDict(extension=".pt"),
    )
    agent = agent_type(env=my_env, config=expected_config)

    assert agent.config == expected_config


def test_dimensions_of_network_match_env(rows=4, columns=5):
    my_env = env.GridWorld(rows=rows, columns=columns, n_goals=1)

    agent = a.FeatureGoalAgent(env=my_env)
    assert agent.policy_net.config.state_size == 2
    assert agent.policy_net.config.goal_size == 2
    assert agent.policy_net.features_size == rows * columns
    assert agent.policy_net.num_actions == 4


# Having tested the basic setup, I need to now test for the following:
# 1. How the agent picks an action
# 2. How the gpi procedure works (tested on test gpi)
# 4. Sample and augment experiences


def test_choose_action():
    eu.misc.seed(0)
    my_env = env.GridWorld(
        rows=3, columns=3, reward_at_goal_position=10, penalization=1, n_goals=1
    )
    agent = a.FeatureGoalWeightAgent(env=my_env)

    # This network takes features, repeats them, and multiplies the second entry
    # by 5. So the max action should always be 2 (unless rewards are negative)
    # however, to observe this, I need to choose all the weights to one to
    # activate the weights.
    agent.policy_net = StubFeatureGoalWeightNetwork()

    obs, *_ = my_env.reset()

    action = agent.choose_action(
        obs=obs,
        list_of_goal_positions=[obs["goal_position"]],
        training=False,
    )
    assert action == 2


# test that sample and augment experience for FGW is working. 
# the idea is that we push a single transition, which will be augmented 
# by the number of goals internally. So we check that the output is 
# what we expect
def test_sample_and_augment_experiences():
    my_env = env.GridWorld(
        rows=3, columns=3, reward_at_goal_position=10, penalization=0, n_goals=1
    )

    # we select to augment over these two goals
    goal_1 = np.array([[2, 0]])
    goal_2 = np.array([[0, 1]])

    goal_1_w = my_env._get_goal_weights_at(goal_1)
    goal_2_w = my_env._get_goal_weights_at(goal_2)

    goal_1_rbf = my_env._get_rbf_vector_at(goal_1)
    goal_2_rbf = my_env._get_rbf_vector_at(goal_2)

    list_of_goal_positions_for_augmentation = [goal_1, goal_2]

    obs, _ = my_env.reset(start_agent_position=np.array([[0, 0]]))

    # make my agent go right, this means that it will hit one of the two goals.
    # therefore, in the augmentation, I expect that for goal (0,1) the reward 
    # calculated as phi*w = 10
    # while for the other goal, it will be 0
    action = 2
    next_obs, *_ = my_env.step(action)

    # We create a full transitio with the relevant elements. Note that 
    # the elements that are set to none are ignored by the augmentation strategy
    # from the non null elements, it will compute everything else.
    example_transition = FullTransition(
        obs["agent_position"],
        None,
        obs["features"],
        None,
        None,
        None,
        action,
        None,
        next_obs["agent_position"],
        None,
        next_obs["features"],
        None,
        None,
    )

    # we create a memory buffer
    example_memory = mem.ExperienceReplayMemory(capacity=1)

    # and push our example transition inside
    example_memory.push(example_transition)

    # then we create an agent and set its memory to the one we created.
    agent = a.FeatureGoalWeightAgent(env=my_env, batch_size=1)
    agent.memory = example_memory

    # finally we sample from it
    experiences, weights = agent._sample_experiences(
        list_of_goal_positions_for_augmentation
    )

    expected_experiences = copy.deepcopy(
        Experiences(
            (obs["agent_position"], obs["agent_position"]),
            (obs["agent_position_rbf"], obs["agent_position_rbf"]),
            (obs["features"], obs["features"]),
            (goal_1, goal_2),
            (goal_1_rbf, goal_2_rbf),
            (goal_1_w, goal_2_w),
            (action, action),
            (my_env.config.penalization, my_env.config.reward_at_goal_position),
            (next_obs["agent_position"], next_obs["agent_position"]),
            (next_obs["agent_position_rbf"], next_obs["agent_position_rbf"]),
            (next_obs["features"], next_obs["features"]),
            (False, True),
            (None, None),
        )
    )

    print("\nObtained: ", experiences)
    print("\nExpected ", expected_experiences)

    assert len(experiences.agent_position_batch) == 2

    assert [
        (elem1 == elem2).all()
        for elem1, elem2 in zip(
            expected_experiences.agent_position_batch, experiences.agent_position_batch
        )
    ]
    assert [
        (elem1 == elem2).all()
        for elem1, elem2 in zip(
            expected_experiences.agent_position_rbf_batch, experiences.agent_position_rbf_batch
        )
    ]
    assert [
        (elem1 == elem2).all()
        for elem1, elem2 in zip(
            expected_experiences.features_batch,
            experiences.features_batch,
        )
    ]
    assert [
        (elem1 == elem2).all()
        for elem1, elem2 in zip(
            expected_experiences.goal_position_batch, experiences.goal_position_batch
        )
    ]
    assert [
        (elem1 == elem2).all()
        for elem1, elem2 in zip(
            expected_experiences.goal_position_rbf_batch, experiences.goal_position_rbf_batch
        )
    ]
    assert [
        (elem1 == elem2).all()
        for elem1, elem2 in zip(
            expected_experiences.goal_weights_batch, experiences.goal_weights_batch
        )
    ]
    assert [
        (elem1 == elem2)
        for elem1, elem2 in zip(
            expected_experiences.action_batch, experiences.action_batch
        )
    ]
    assert [
        (elem1 == elem2)
        for elem1, elem2 in zip(
            expected_experiences.reward_batch, experiences.reward_batch
        )
    ]
    assert [
        (elem1 == elem2).all()
        for elem1, elem2 in zip(
            expected_experiences.next_agent_position_batch,
            experiences.next_agent_position_batch,
        )
    ]
    assert [
        (elem1 == elem2).all()
        for elem1, elem2 in zip(
            expected_experiences.next_agent_position_rbf_batch,
            experiences.next_agent_position_rbf_batch,
        )
    ]
    assert [
        (elem1 == elem2).all()
        for elem1, elem2 in zip(
            expected_experiences.next_features_batch,
            experiences.next_features_batch,
        )
    ]
    assert [
        (elem1 == elem2)
        for elem1, elem2 in zip(
            expected_experiences.terminated_batch, experiences.terminated_batch
        )
    ]


def test_build_tensor_from_batch_of_np_arrays(batch_size=32):
    my_env = env.GridWorld()
    agent = a.FeatureGoalAgent(env=my_env)

    batch = []
    for i in range(batch_size):
        batch.append(np.random.rand(1, 100))
    batch = agent._build_tensor_from_batch_of_np_arrays(batch)

    assert tuple(batch.shape) == (batch_size, 100)
    assert batch.dtype == torch.float
