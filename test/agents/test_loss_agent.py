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

def build_env_agent_and_batch_args():
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
    agent.policy_net = stub_feature_goal_network(features_size=3 * 3).to(agent.device)

    batch_args = {
        "agent_position_batch": torch.tensor([[0,0]]),
        "agent_position_features_batch": torch.tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0]]
        ).to(agent.device),
        "goal_batch": torch.tensor([[2, 2]]).to(agent.device),
        # Note goal weights dont match with goal position because if it did,
        # the product sf*w would be zero and the max would not be possible to take 
        # Since I want to activate all the features, I just put all ones
        "goal_weights_batch": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(
            agent.device
        ),
        "action_batch": torch.tensor([1]).unsqueeze(1).to(agent.device),
        "reward_batch": torch.tensor([0]).to(agent.device),
        "next_agent_position_batch": torch.tensor([[1,0]]),
        "next_agent_position_features_batch": torch.tensor(
            [[0, 0, 0, 1, 0, 0, 0, 0, 0]]
        ).to(agent.device),
        "terminated_batch": torch.tensor([False]).to(agent.device),
    }
    return env, agent, batch_args

def test_build_q_target():

    env, agent, batch_args = build_env_agent_and_batch_args()

    # The stub will take the features, repeat them 4 times, multiply the entire thirdf
    # entry by 5 and then multiply each of the four rows by the weights vector
    # which is all ones and the sum
    # the end result will be [1,1,5,1]
    # Since the max q value is 5 for action 2(counting from zero),
    # we expect the q target to be 0.0 + 1.0 * 5.0 = 5.0
    expected_target_q = 5.0

    target_q, max_action, sf, w, reward_phi_batch = agent._build_q_target(batch_args)

    # verify action chosen is the third one
    assert max_action == 2
    # verify correct q
    assert target_q == expected_target_q

def test_build_psi_target():
    env, agent, batch_args = build_env_agent_and_batch_args()

    #### Here we simulate the output of get_q_target

    max_action = torch.tensor([2]).to(agent.device)
    sf = torch.tensor(
        [
            [
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 5, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0],
            ]
        ]
    ).to(agent.device)

    reward_phi_batch = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 0, 0]]).to(agent.device)

    #################################################################################

    target_psi = agent._build_psi_target(batch_args, max_action, sf, reward_phi_batch)

    # The expected psi target is calculated as reward_phi_batch + 1.0*max_a SF
    # Since the 
    expected_target_psi = torch.tensor([[0, 0, 0, 6, 0, 0, 0, 0, 0]]).to(agent.device)

    cmp = expected_target_psi == target_psi

    assert cmp.all()

def test_build_q_predicted():
    env, agent, batch_args = build_env_agent_and_batch_args()

    predicted_q, *_ = agent._build_q_predicted(batch_args)

    expected_predicted_q = 1.0
    
    assert predicted_q == expected_predicted_q

def test_build_psi_predicted():
    env, agent, batch_args = build_env_agent_and_batch_args()


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

    predicted_psi = agent._build_psi_predicted(batch_args, sf)

    expected_predicted_psi = torch.tensor( [[1, 0, 0, 0, 0, 0, 0, 0, 0]]).to(agent.device)

    cmp = expected_predicted_psi == predicted_psi

    assert cmp.all()


def test_get_td_error_for_usf():
    pass


if __name__ == "__main__":
    sn = stub_feature_goal_network()

    my_features = torch.ones(9)

    print(sn(my_features, None, None))
