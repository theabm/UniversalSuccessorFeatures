import numpy as np
import agents.multigoalDQN as dqn
import envs.gridWorld as env
import exputils.data.logging as log

dqn_agent = dqn.MultigoalDQN()
my_env = env.GridWorld()


def train_agent(agent, my_env, episodes):

    step = 0
    
    for episode in range(episodes):
        
        total_reward = 0
        agent_state, _ = my_env.reset()
        goal_state = my_env.get_current_goal_coordinates()

        agent.start_episode(episode = episode)
        
        while True:
        
            state = np.concatenate((agent_state, goal_state))

            action = agent.choose_action(state, purpose = "inference")
            
            agent_next_state, reward, terminated, truncated, _ = my_env.step(action)

            total_reward += reward

            transition = goal_state, agent_state, action, reward, agent_next_state, terminated, truncated

            agent.train(transition, step)

            agent_state = agent_next_state

            if terminated or truncated:
                break

            log.add_value("reward_per_step", total_reward)
        log.add_value("reward_per_episode", total_reward, tb_global_step=episode)



train_agent(dqn_agent, my_env, 10)
    

