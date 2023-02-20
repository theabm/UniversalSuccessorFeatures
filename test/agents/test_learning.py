import universalSuccessorFeatures.envs as envs
import numpy as np


my_test_env = envs.GridWorld(lenght_x = 3, length_y = 3)

start_position = np.array([0,0])

goal_1_position = np.array([2,2])
goal_2_position = np.array([2,0])
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


def test_training(episodes = 20):
