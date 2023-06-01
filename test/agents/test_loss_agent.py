import universal_successor_features as usf
import pytest
import torch
import exputils as eu
import numpy as np


class stub_feature_goal_network(torch.nn.Module):
    @staticmethod
    def default_config():
        return eu.AttrDict(
            state_size=2,
            goal_size=2,
            features_size=None,
            num_actions=4,
        )

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, agent_position_features, policy_goal_position, env_goal_weights):
        # This network simply takes the agent features, repeats it 4 times in the
        # last dimension, and then multiplies the third element by 5.
        # Therefore, when calculating the max, it should always be the third action
        #
        # Expected shape (1,features_size)
        agent_position_features = (agent_position_features).tile(4, 1)
        agent_position_features[2] *= 5
        sf = agent_position_features.unsqueeze(0)

        q = torch.sum(torch.mul(sf, (env_goal_weights).unsqueeze(1)), dim=2)

        return q, sf, None, agent_position_features


def test_build_q_target():
    env = usf.envs.GridWorld(
        rows=3,
        columns=3,
        penalization=0,
        reward_at_goal_position=20,
    )

    agent = usf.agents.feature_goal_weight_agent.FeatureGoalWeightAgent(
        env=env, discount_factor=1.0, batch_size=1
    )

    agent.target_net = stub_feature_goal_network(features_size=3 * 3).to(agent.device)

    batch_args = {
        "agent_position_batch": None,
        "agent_position_features_batch": None,
        "goal_batch": torch.tensor([[1, 0]]).to(agent.device),
        "goal_weights_batch": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(
            agent.device
        ),
        "action_batch": None,
        "reward_batch": torch.tensor([0]).to(agent.device),
        "next_agent_position_batch": None,
        "next_agent_position_features_batch": torch.tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0]]
        ).to(agent.device),
        "terminated_batch": torch.tensor([False]).to(agent.device),
    }
    # The stub will take the features, repeat them 4 times, multiply the entire thirdf
    # entry by 5 and then multiply each of the four rows by the weights vector
    # which is all ones and the sum
    # the end result will be [1,1,5,1]
    # Since the max q value is 5 for action 3, we expect the q target to be
    # 0.0 + 1.0 * 5.0 = 5.0
    expected_q = 5.0

    target_q, action, sf, w, reward_phi_batch = agent._build_q_target(batch_args)

    # verify action chosen is the third one
    assert action == 2
    # verify correct q
    assert target_q == expected_q


def test_build_psi_target():
    env = usf.envs.GridWorld(
        rows=3,
        columns=3,
        penalization=0,
        reward_at_goal_position=20,
    )

    agent = usf.agents.feature_goal_weight_agent.FeatureGoalWeightAgent(
        env=env, discount_factor=1.0, batch_size=1
    )

    agent.target_net = stub_feature_goal_network(features_size=3 * 3).to(agent.device)

    batch_args = {
        "agent_position_batch": None,
        "agent_position_features_batch": None,
        "goal_batch": torch.tensor([[1, 0]]).to(agent.device),
        "goal_weights_batch": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(
            agent.device
        ),
        "action_batch": None,
        "reward_batch": torch.tensor([0]).to(agent.device),
        "next_agent_position_batch": None,
        "next_agent_position_features_batch": torch.tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0]]
        ).to(agent.device),
        "terminated_batch": torch.tensor([False]).to(agent.device),
    }

    action = torch.tensor([2]).to(agent.device)
    sf = torch.tensor(
        [
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ]
    ).to(agent.device)

    reward_phi_batch = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0]]).to(agent.device)

    target_psi = agent._build_psi_target(batch_args, action, sf, reward_phi_batch)

    expected_psi = torch.tensor([[6, 0, 0, 0, 0, 0, 0, 0, 0]]).to(agent.device)
    print(target_psi)

    cmp = expected_psi == target_psi

    assert cmp.all()


def test_usf_loss():
    pass


if __name__ == "__main__":
    sn = stub_feature_goal_network()

    my_features = torch.ones(9)

    print(sn(my_features, None, None))
