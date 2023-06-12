import universal_successor_features as usf
import torch
from test.agents.stub_classes import StubFeatureGoalNetwork


def build_env_agent_and_batch_args():
    env = usf.envs.GridWorld(
        rows=3,
        columns=3,
        penalization=0,
        reward_at_goal_position=20,
        n_goals=1
    )

    agent = usf.agents.feature_goal_weight_agent.FeatureGoalWeightAgent(
        env=env,
        discount_factor=1.0,
        batch_size=1,
        loss_weight_psi=1,
        loss_weight_q=1,
        loss_weight_r=0,
    )

    # replace the policy and target net with the stub network
    agent.target_net = StubFeatureGoalNetwork(features_size=3 * 3).to(agent.device)
    agent.policy_net = StubFeatureGoalNetwork(features_size=3 * 3).to(agent.device)

    # simulate a batch (size 1) of experience
    batch_args = {
        "agent_position_batch": torch.tensor([[0, 0]]),
        "agent_position_features_batch": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0]]).to(
            agent.device
        ),
        "goal_batch": torch.tensor([[2, 2]]).to(agent.device),
        # Note goal weights dont match with goal position because if it did,
        # the product sf*w would be zero and the max would not be possible to take
        # Since I want to activate all the features, I just put all ones
        "goal_weights_batch": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(
            agent.device
        ),
        "action_batch": torch.tensor([1]).unsqueeze(1).to(agent.device),
        "reward_batch": torch.tensor([0]).to(agent.device),
        "next_agent_position_batch": torch.tensor([[1, 0]]),
        "next_agent_position_features_batch": torch.tensor(
            [[0, 0, 0, 1, 0, 0, 0, 0, 0]]
        ).to(agent.device),
        "terminated_batch": torch.tensor([False]).to(agent.device),
    }
    return env, agent, batch_args


def test_build_q_target():
    env, agent, batch_args = build_env_agent_and_batch_args()

    # The stub will take next_agent_position_features, repeat them 4 times,
    # multiply the entire third entry by 5 and then multiply each of the four
    # rows by the weights vector which is all ones and then sum
    # the end result will be [1,1,5,1]
    # Since the max q value is 5 for action 2(counting from zero),
    # we expect the q target to be 0.0 + 1.0 * 5.0 = 5.0
    expected_target_q = 5.0

    target_q, max_action, sf, w, reward_phi_batch = agent._build_q_target(batch_args)

    assert (w == batch_args["goal_weights_batch"]).all()
    assert (reward_phi_batch == batch_args["next_agent_position_features_batch"]).all()

    # verify action chosen is the third one
    assert max_action == 2
    # verify correct q
    assert target_q == expected_target_q


def test_build_psi_target():
    env, agent, batch_args = build_env_agent_and_batch_args()

    #### Here we simulate the output of build_q_target

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
    # Since the max is the third action, we will pick the third entry of SF and
    # then add reward phi_batch to it
    expected_target_psi = torch.tensor([[0, 0, 0, 6, 0, 0, 0, 0, 0]]).to(agent.device)

    cmp = expected_target_psi == target_psi

    assert cmp.all()


def test_build_q_predicted():
    env, agent, batch_args = build_env_agent_and_batch_args()

    predicted_q, *_ = agent._build_q_predicted(batch_args)
    # the stub network will take agent_position_features, repeat it, and multiply
    # the third entry by 5. Then we multiply all the entries by all ones and sum.
    # the result is [1,1,5,1]
    # Since the action we took is 1, we will need to select the second entry
    # which is one

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

    expected_predicted_psi = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0]]).to(
        agent.device
    )

    cmp = expected_predicted_psi == predicted_psi

    assert cmp.all()


def test_get_td_error_for_usf():
    env, agent, batch_args = build_env_agent_and_batch_args()

    predicted_q, predicted_psi, predicted_r = agent._build_predicted_batch(batch_args)
    target_q, target_psi, target_r = agent._build_target_batch(batch_args)

    td_error = agent._get_td_error_for_usf(
        target_q, target_psi, target_r, predicted_q, predicted_psi, predicted_r
    )
    # from tests above
    # expected_predicted_q = 1.0
    # expected_predicted_psi = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0]])
    # expected_target_q = 5.0
    # expected_target_psi = torch.tensor([[0, 0, 0, 6, 0, 0, 0, 0, 0]])
    # expected td_error = (5-1)^2 + mean((-1,0,0,6,0,0,0,0,0))^2 = 16 + (1 + 36)/9

    expected_td_error = torch.tensor([16 + 37/9]).to(agent.device)

    assert torch.allclose(expected_td_error, td_error, rtol=0, atol=0.05)



if __name__ == "__main__":
    sn = StubFeatureGoalNetwork()

    my_features = torch.ones(9)

    print(sn(my_features, None, None))
