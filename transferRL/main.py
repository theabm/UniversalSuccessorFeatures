import torch
import numpy as np
import agents.multigoalDQN as dqn
import envs.gridWorld as env
import memory.experienceReplayMemory as mem

dqn_agent = dqn.MultigoalDQN()
my_env = env.GridWorld()


def train_agent(agent, my_env, episodes):

    step = 0
    
    for episode in range(episodes):
        
        agent_state, info = my_env.reset()
        goal_state = my_env.get_current_goal_coordinates()

        agent.start_episode(episode = episode)
        
        while True:

            state = np.concatenate((agent_state, goal_state))

            action = agent.choose_action(state, purpose = "training")
            
            agent_next_state, reward, terminated, truncated, _ = my_env.step(action)

            transition = goal_state, agent_state, action, reward, agent_next_state, terminated, truncated

            agent.train(transition, step)

            break


train_agent(dqn_agent, my_env, 10)
    

