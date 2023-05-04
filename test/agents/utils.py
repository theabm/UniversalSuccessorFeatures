import torch
import numpy as np


# Utilities needed for testing the learning of agents
def compute_q_function(agent, env, list_of_goal_positions, use_pos, use_weight):
    size = env.rows * env.columns
    pos = []
    features = []
    for i in range(env.rows):
        for j in range(env.columns):
            pos.append([i, j])
            features.append(
                env._get_agent_position_features_at(np.array([[i, j]]))
            )

    # shape size*len(list_of_goal_positions), pos_size
    pos_tensor = (
        torch.tensor(pos)
        .tile((len(list_of_goal_positions), 1))
        .to(torch.float)
        .to(agent.device)
    )
    # size (nsize, 1, size_features) -> (n_size, size_features)
    # -> (2_nsize, size_features)
    features_tensor = (
        torch.tensor(features)
        .squeeze()
        .tile((len(list_of_goal_positions), 1))
        .to(torch.float)
        .to(agent.device)
    )

    # shape len(list_of_goal_positions), 1, goal_size
    goal_tensor = torch.tensor(list_of_goal_positions)

    # shape len(list_of_goal_positions)*size, goal_size
    goal_tensor = (
        goal_tensor.tile((1, size, 1))
        .reshape(len(list_of_goal_positions) * size, goal_tensor.shape[-1])
        .to(torch.float)
        .to(agent.device)
    )

    if use_pos and not use_weight:
        # shape (list_of_goal_positions)*size, goal_size(pos_size)
        q, *_ = agent.policy_net(
            agent_position=pos_tensor,
            policy_goal_position=goal_tensor,
            env_goal_position=goal_tensor,
        )

    # zero out the entries at the goal position index since the Q function is not
    # defined at these points (if you look above the ground truth is set to zero
    # for these entries)
    for i, goal in enumerate(list_of_goal_positions):
        idx = env.rows * goal[0][0] + goal[0][1] + i * size
        q[idx] = 0
    return q


def test_training(agent, env, n_steps, q_ground_truth, step_function, use_pos, use_weight):
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
                obs, agent, env
            )
            agent.train(transition=transition)

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


def step_state_goal_agent(obs, agent, my_env):
    action = agent.choose_action(
        agent_position=obs["agent_position"],
        list_of_goal_positions=[obs["goal_position"]],
        env_goal_position=obs["goal_position"],
        training=True,
    )

    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (
        obs["agent_position"],
        obs["goal_position"],
        action,
        reward,
        next_obs["agent_position"],
        terminated,
        truncated,
    )

    return next_obs, reward, terminated, truncated, transition


def step_feature_goal_agent(obs, agent, my_env):
    action = agent.choose_action(
        agent_position_features=obs["agent_position_features"],
        list_of_goal_positions=[obs["goal_position"]],
        env_goal_position=obs["goal_position"],
        training=True,
    )

    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (
        obs["agent_position_features"],
        obs["goal_position"],
        action,
        reward,
        next_obs["agent_position_features"],
        terminated,
        truncated,
    )

    return next_obs, reward, terminated, truncated, transition
