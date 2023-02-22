import universalSuccessorFeatures.envs as envs
import universalSuccessorFeatures.agents as mdqa
import universalSuccessorFeatures.networks.multigoalDQN as mdqn
import numpy as np
from collections import namedtuple
import pytest
import exputils as eu


my_test_env = envs.GridWorld(rows = 3, columns = 3, penalization = 0, reward_at_goal_state = 1)

start_position = (0,0)

goal_1_position = (2,2)
goal_2_position = (2,0)
goal_list = [goal_1_position,goal_2_position]

discount_factor = 0.5

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

q_gt_g1_list = np.array([q_gt_g1_s1, q_gt_g1_s2, q_gt_g1_s3, q_gt_g1_s4, q_gt_g1_s5, q_gt_g1_s6, q_gt_g1_s7, q_gt_g1_s8])

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

q_gt_g2_list = np.array([q_gt_g2_s1, q_gt_g2_s2, q_gt_g2_s3, q_gt_g2_s4, q_gt_g2_s5, q_gt_g2_s6, q_gt_g2_s8, q_gt_g2_s9])

Transition = namedtuple("Transition", ("state", "goal", "action", "reward", "next_state", "terminated", "truncated"))

@pytest.mark.parametrize(
    "network, use_goals, use_features",
    [
        (mdqn.StateGoalPaperDQN, True, False),
        (mdqn.StateGoalAugmentedDQN, True, False),
        # (mdqn.FeatureGoalPaperDQN, False, True),
        # (mdqn.FeatureGoalAugumentedDQN, False, True),

    ]
)
def test_training(network, use_goals, use_features, seed = 0, num_episodes=50):
    if seed is not None:
        eu.misc.seed(seed)


    agent = mdqa.MultigoalDQNAgentBase(
        epsilon = 1.0, train_for_n_iterations = 2, discount_factor = discount_factor, network = {"cls": network}
        ) 

    reward_per_episode = 0
    step = 0

    for episode in range(num_episodes):

        goal_position = my_test_env.sample_a_goal_position(goal_list=goal_list)

        agent_state, _ = my_test_env.reset(start_position=start_position,goal=goal_position)
        agent_state_features = my_test_env.get_current_state_features()

        goal_position = my_test_env.get_current_goal_position_in_matrix()

        agent.start_episode(episode=episode)

        while True:
            
            action = agent.choose_action(s = agent_state, g = goal_position, phi_s = agent_state_features, purpose = "training")
            
            agent_next_state, reward, terminated, truncated, _ = my_test_env.step(action=action)
            agent_next_state_features = my_test_env.get_current_state_features()

            if use_features:
                t = Transition(agent_state_features, goal_position, action, reward, agent_next_state_features, terminated, truncated)
            elif use_goals:
                t = Transition(agent_state, goal_position, action, reward, agent_next_state, terminated, truncated)

            agent.train(t, step = 0)


            agent_state = agent_next_state
            agent_state_features = agent_next_state_features
            step += 1
            
            agent.end_episode()
            if terminated or truncated:
                break
        
    global goal_1_position
    goal_1_position = np.array(goal_1_position)
    global goal_2_position
    goal_2_position = np.array(goal_2_position)

    q_pred_g1_list = []
    q_pred_g2_list = []
    for i in range(my_test_env.rows):
        for j in range(my_test_env.columns):
            state = np.array([i,j])
            state_features = my_test_env.get_features_for_state(state = state)
            if i*my_test_env.rows + j == 8:
                q_pred_g2_list.append(agent.policy_net(**agent._make_compatible_with_nn(s = state, g = goal_2_position, phi_s = state_features)).squeeze().cpu().detach().numpy())
                continue
            elif i*my_test_env.rows + j == 6:
                q_pred_g1_list.append(agent.policy_net(**agent._make_compatible_with_nn(s = state, g = goal_1_position, phi_s = state_features)).squeeze().cpu().detach().numpy())
                continue
            else:
                q_pred_g1_list.append(agent.policy_net(**agent._make_compatible_with_nn(s = state, g = goal_1_position, phi_s = state_features)).squeeze().cpu().detach().numpy())
                q_pred_g2_list.append(agent.policy_net(**agent._make_compatible_with_nn(s = state, g = goal_2_position, phi_s = state_features)).squeeze().cpu().detach().numpy())

    q_pred_g1_list = np.array(q_pred_g1_list)
    q_pred_g2_list = np.array(q_pred_g2_list)

    print("\n", q_gt_g1_list,"\n", q_pred_g1_list) 
    print("\n", q_gt_g2_list,"\n", q_pred_g2_list) 
    cmp1 = np.allclose(q_pred_g1_list, q_gt_g1_list, rtol = 0, atol = 0.05)
    cmp2 = np.allclose(q_pred_g2_list, q_gt_g2_list, rtol = 0, atol = 0.05)

    assert cmp1 and cmp2
            