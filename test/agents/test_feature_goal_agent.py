import numpy as np
import exputils as eu
import universal_successor_features.agents as a
import universal_successor_features.envs.grid_world as env
import universal_successor_features.networks as nn
import universal_successor_features.epsilon as eps
import universal_successor_features.memory as mem
import torch


def test_default_configuration():
    my_env = env.GridWorld(
        rows=3,
        columns=3,
    )
    agent = a.FeatureGoalAgent(env=my_env)

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
            cls=nn.FeatureGoalUSF,
            use_gdtuo=False,
            optimizer=torch.optim.Adam,
            state_size=2,
            goal_size=2,
            features_size=9,
            num_actions=4,
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

    assert agent.config == expected_config


def test_agent_matches_custom_config():
    my_env = env.GridWorld(
        rows=3,
        columns=3,
    )

    expected_config = eu.AttrDict(
        device="cuda",
        discount_factor=0.1,
        batch_size=2,
        learning_rate=3e-4,
        train_for_n_iterations=4,
        train_every_n_steps=6,
        loss_weight_q = 3,
        loss_weight_psi=4.01,
        loss_weight_phi=-3.50,
        network=eu.AttrDict(
            cls=nn.FeatureGoalUSF,
            use_gdtuo=False,
            optimizer=torch.optim.Adam,
            state_size=2,
            goal_size=2,
            features_size=9,
            num_actions=4,
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
    agent = a.FeatureGoalAgent(env=my_env, config=expected_config)

    assert agent.config == expected_config

def test_dimensions_of_network_match_env(rows = 4, columns = 5):
    my_env = env.GridWorld(
        rows=rows,
        columns=columns,
    )

    agent = a.FeatureGoalAgent(env=my_env)
    assert agent.policy_net.config.state_size==2
    assert agent.policy_net.config.goal_size==2
    assert agent.policy_net.features_size==rows*columns
    assert agent.policy_net.num_actions==4


def test_choose_action():
    my_env = env.GridWorld()
    agent = a.FeatureGoalAgent(env=my_env)

    obs, *_ = my_env.reset()

    action = agent.choose_action(
        obs=obs,
        list_of_goal_positions=[obs["goal_position"]],
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
