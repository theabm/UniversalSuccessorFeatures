import torch
import numpy as np
import universal_successor_features as usf

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


# Utilities needed for testing the learning of agents
def compute_q_function(agent, env, list_of_goal_positions, use_pos, use_weight):
    size = env.rows * env.columns
    pos = []
    features = []
    rbf_position = []
    for i in range(env.rows):
        for j in range(env.columns):
            pos.append([i, j])
            features.append(env._get_one_hot_vector_at(np.array([[i, j]])))
            rbf_position.append(env._get_rbf_vector_at(np.array([[i, j]])))

    # shape size*len(list_of_goal_positions), pos_size
    pos_tensor = (
        torch.tensor(pos)
        .tile((len(list_of_goal_positions), 1))
        .to(torch.float)
        .to(agent.device)
    )
    # size (grid_size, 1, size_features) -> (grid_size, size_features)
    # -> (2* grid_size, size_features)
    features_tensor = np.array(features)
    features_tensor = (
        torch.tensor(features_tensor)
        .squeeze()
        .tile((len(list_of_goal_positions), 1))
        .to(torch.float)
        .to(agent.device)
    )

    rbf_position_tensor = np.array(rbf_position)
    rbf_position_tensor = (
        torch.tensor(rbf_position_tensor)
        .squeeze()
        .tile((len(list_of_goal_positions), 1))
        .to(torch.float)
        .to(agent.device)
    )
    assert rbf_position_tensor.shape == (2 * size, env.rbf_size)

    rbf_goal_positions_tensor = [
        env._get_rbf_vector_at(goal_position)
        for goal_position in list_of_goal_positions
    ]

    # shape len(list_goals), 1, rbf_size
    rbf_goal_positions_tensor = np.array(rbf_goal_positions_tensor)

    rbf_goal_positions_tensor = (
        torch.tensor(rbf_goal_positions_tensor)
        .tile((1, size, 1))
        .reshape(
            len(list_of_goal_positions) * size, rbf_goal_positions_tensor.shape[-1]
        )
        .to(torch.float)
        .to(agent.device)
    )
    assert rbf_goal_positions_tensor.shape == (2 * size, env.rbf_size)

    # shape len(list_of_goal_positions), 1, goal_size
    goal_tensor = np.array(list_of_goal_positions)

    # shape len(list_of_goal_positions)*size, goal_size
    goal_tensor = (
        torch.tensor(goal_tensor)
        .tile((1, size, 1))
        .reshape(len(list_of_goal_positions) * size, goal_tensor.shape[-1])
        .to(torch.float)
        .to(agent.device)
    )

    # shape len(list_of_goal_positions), 1, goal_size
    weight_tensor = np.array(
        [env._get_goal_weights_at(goal) for goal in list_of_goal_positions]
    )

    # shape len(list_of_goal_positions)*size, goal_size
    weight_tensor = (
        torch.tensor(weight_tensor)
        .tile((1, size, 1))
        .reshape(len(list_of_goal_positions) * size, weight_tensor.shape[-1])
        .to(torch.float)
        .to(agent.device)
    )

    batch_dict = {
        "agent_position_batch": pos_tensor,
        "agent_position_rbf_batch": rbf_position_tensor,
        "features_batch": features_tensor,
        "goal_position_batch": goal_tensor,
        "goal_position_rbf_batch": rbf_goal_positions_tensor,
        "goal_weights_batch": weight_tensor,
    }

    args = agent._build_predicted_args(batch_dict)
    q, *_ = agent.policy_net(**args)

    for i, goal in enumerate(list_of_goal_positions):
        idx = env.rows * goal[0][0] + goal[0][1] + i * size
        q[idx] = 0
    return q


def test_training(
    agent, env, n_steps, q_ground_truth, step_function, use_pos, use_weight
):
    start_position = np.array([[0, 0]])

    goal_list = [np.array([[2, 2]]), np.array([[2, 0]])]

    step = 0
    episode = 0

    while step < n_steps:
        terminated = False
        truncated = False

        goal_position = env.sample_a_goal_position_from_list(goal_list=goal_list)
        obs, _ = env.reset(
            start_agent_position=start_position, goal_position=goal_position
        )
        agent.start_episode(episode=episode)

        while not terminated and not truncated and step < n_steps:
            next_obs, reward, terminated, truncated, transition = step_function(
                obs, agent, env, [obs["goal_position"]], True
            )
            agent.train(
                transition=transition,
                list_of_goal_positions_for_augmentation=goal_list,
            )

            obs = next_obs
            step += 1

        agent.end_episode()
        episode += 1

    q_predicted = compute_q_function(agent, env, goal_list, use_pos, use_weight)
    q_gt = q_ground_truth.to(agent.device)
    print("Predicted:\n", q_predicted)
    print("Ground Truth:\n", q_gt)

    cmp = torch.allclose(q_predicted, q_gt, rtol=0, atol=0.05)
    return cmp
