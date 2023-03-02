import universalSuccessorFeatures.envs as envs
import universalSuccessorFeatures.agents as a
import universalSuccessorFeatures.networks as nn
import numpy as np
import pytest
import exputils as eu
import torch



#Ground truth values for the following configuration (discount = 0.5)
# o o o
# o o o 
# o o g         #UP  #DOWN #RIGHT #LEFT
q_gt_g1_s1 = [0.0625,0.125,0.125,0.0625]
q_gt_g1_s2 = [0.125,0.250,0.250,0.0625]
q_gt_g1_s3 = [0.250,0.500,0.250,0.125]
q_gt_g1_s4 = [0.0625,0.250,0.250,0.125]
q_gt_g1_s5 = [0.125,0.500,0.500,0.125]
q_gt_g1_s6 = [0.250,1.000,0.500,0.250]
q_gt_g1_s7 = [0.125,0.250,0.500,0.250]
q_gt_g1_s8 = [0.250,0.500,1.000,0.250]
# q_gt_g1_s9 = [0.500,1.000,1.000,0.500] #corresponding Q(goal state, actions)

q_gt_g1_array = np.array([q_gt_g1_s1, q_gt_g1_s2, q_gt_g1_s3, q_gt_g1_s4, q_gt_g1_s5, q_gt_g1_s6, q_gt_g1_s7, q_gt_g1_s8])

#Ground truth values for the following configuration (discount = 0.5)
# o o o
# o o o 
# g o o
q_gt_g2_s1 = [0.250,0.500,0.125,0.250]
q_gt_g2_s2 = [0.125,0.250,0.0625,0.250]
q_gt_g2_s3 = [0.0625,0.125,0.0625,0.125]
q_gt_g2_s4 = [0.250,1.000,0.250,0.500]
q_gt_g2_s5 = [0.125,0.500,0.125,0.500]
q_gt_g2_s6 = [0.0625,0.250,0.125,0.250] 
# q_gt_g2_s7 = [0.500,1.000,0.500,1.000]
q_gt_g2_s8 = [0.250,0.500,0.250,1.000]
q_gt_g2_s9 = [0.125,0.250,0.250,0.500]

q_gt_g2_array = np.array([q_gt_g2_s1, q_gt_g2_s2, q_gt_g2_s3, q_gt_g2_s4, q_gt_g2_s5, q_gt_g2_s6, q_gt_g2_s8, q_gt_g2_s9])

@pytest.mark.parametrize(
        "network",
        [
            (nn.StateGoalPaperDQN),
            (nn.StateGoalUSF),
            (nn.StateGoalAugmentedDQN)
        ]
)
def test_training(network, discount_factor = 0.5, num_episodes=50, seed=0):

    if seed is not None:
        eu.misc.seed(seed)

    my_env = envs.GridWorld(rows = 3, columns = 3, penalization = 0, reward_at_goal_position = 1)

    agent = a.StateGoalAgent(
        epsilon = {"value" : 1.0}, train_for_n_iterations = 2, discount_factor = discount_factor, network = {"cls":network}
        )
    device = agent.device

    start_position = np.array([0,0])
    
    goal_1_position = np.array([2,2])
    goal_2_position = np.array([2,0])
    goal_list = [goal_1_position,goal_2_position]

    step = 0

    for episode in range(num_episodes):

        goal_position = my_env.sample_a_goal_position_from_list(goal_list=goal_list)
        obs, _ = my_env.reset(start_agent_position = start_position, goal_position=goal_position)
        agent.start_episode(episode=episode)

        while True:
            
            action = agent.choose_action(agent_position=obs["agent_position"], goal_position=obs["goal_position"], training=True)
            
            next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

            transition = (obs["agent_position"], obs["goal_position"], action, reward, next_obs["agent_position"], terminated, truncated)

            agent.train(transition=transition, step = step)

            if terminated or truncated:
                agent.end_episode()
                break

            obs = next_obs
            step += 1
    
    goal_1_position = torch.tensor(goal_1_position).to(torch.float).unsqueeze(0).to(device)
    goal_2_position = torch.tensor(goal_2_position).to(torch.float).unsqueeze(0).to(device)
            
    q_pred_g1_array = []
    q_pred_g2_array = []
    for i in range(my_env.rows):
        for j in range(my_env.columns):
            agent_position = torch.tensor([i,j]).to(torch.float).unsqueeze(0).to(device)
            idx = i*my_env.rows + j
            if idx == 8:
                q_pred_g2_array.append(agent.policy_net(agent_position=agent_position, goal_position=goal_2_position).cpu().squeeze().detach().numpy())
                continue
            elif idx == 6:
                q_pred_g1_array.append(agent.policy_net(agent_position=agent_position, goal_position=goal_1_position).cpu().squeeze().detach().numpy())
                continue
            else:
                q_pred_g1_array.append(agent.policy_net(agent_position=agent_position, goal_position=goal_1_position).cpu().squeeze().detach().numpy())
                q_pred_g2_array.append(agent.policy_net(agent_position=agent_position, goal_position=goal_2_position).cpu().squeeze().detach().numpy())


    q_pred_g1_array = np.array(q_pred_g1_array)
    q_pred_g2_array = np.array(q_pred_g2_array)
    cmp1 = np.allclose(q_pred_g1_array, q_gt_g1_array, rtol = 0, atol = 0.05)
    cmp2 = np.allclose(q_pred_g2_array, q_gt_g2_array, rtol = 0, atol = 0.05)

    assert cmp1 and cmp2
            
