import universalSuccessorFeatures.envs as envs
import universalSuccessorFeatures.agents as mdqa
import universalSuccessorFeatures.networks.multigoalDQN as mdqn
import numpy as np
from collections import namedtuple

my_test_env = envs.GridWorld(lenght_x = 3, length_y = 3, penalization = 0, reward_at_goal_state = 1)

start_position = (0,0)

goal_1_position = (2,2)
goal_2_position = (2,0)
goal_list = [goal_1_position,goal_2_position]

discount_factor = 0.5

#Ground truth values for the following configuration (discount = 0.5)
# o o o
# o o o 
# o o g
q_gt_g1_s1 = [0.0625,0.125,0.125,0.0625]
q_gt_g1_s2 = [0.125,0.250,0.250,0.0625]
q_gt_g1_s3 = [0.250,0.500,0.250,0.125]
q_gt_g1_s4 = [0.0625,0.250,0.250,0.125]
q_gt_g1_s5 = [0.125,0.500,0.500,0.125]
q_gt_g1_s6 = [0.250,1.000,0.500,0.250]
q_gt_g1_s7 = [0.125,0.250,0.500,0.250]
q_gt_g1_s8 = [0.250,0.500,1.000,0.250]
q_gt_g1_s9 = [0.500,1.000,1.000,0.500]

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
q_gt_g2_s7 = [0.500,1.000,0.500,1.000]
q_gt_g2_s8 = [0.250,0.500,0.250,1.000]
q_gt_g2_s9 = [0.125,0.250,0.250,0.500]

Transition = namedtuple("Transition", ("state","goal","action","reward","next_state","terminated","truncated"))
def test_training(episodes = 20):
    step = 0
    agent = mdqa.MultigoalDQNAgent(discount_factor = discount_factor, network = {"cls": mdqn.StateGoalPaperDQN}) 
    for ep in episodes:

        total_reward = 0

        goal_position = my_test_env.sample_a_goal_position(goal_list=goal_list)

        agent_state, _ = my_test_env.reset(start_position=start_position,goal=goal_position)
        goal_position = my_test_env.get_current_goal_position_in_matrix()

        agent.start_episode(episode=ep)

        while True:
            
            action = agent.choose_action(s = agent_state, g = goal_position, purpose = "training")
            agent_next_state, reward, terminated, truncated, _ = my_test_env.step(action=action)
            total_reward+=reward
            
            t = Transition(agent_state, goal_position, action, reward, agent_next_state, terminated, truncated)
            agent.train(t, step = 0)


            agent_state = agent_next_state
            step += 1
            

            if terminated or truncated:
                break
    ## MISSING FUNCTION TO COMPARE BOTH Q VALUES
