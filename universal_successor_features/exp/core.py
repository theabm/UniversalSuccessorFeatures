import numpy as np
import exputils as eu
import exputils.data.logging as log

def run_rl_training(config = None, **kwargs):

    # default config
    default_config = eu.AttrDict(
        seed = None, 
        env=eu.AttrDict(
            cls=None,
            nmax_steps=np.inf,
        ),
        agent = eu.AttrDict(
            cls = None,
            network = eu.AttrDict(cls = None),
        ),
        update_agent = None,
        training = None,
        step_function = None,
        n_max_steps = np.inf,
        log_name_episode = 'episode',
        log_name_step = 'step',
        log_name_step_per_episode = 'step_per_episode',
        log_name_episode_per_step = 'episode_per_step',
        log_name_reward_per_episode = 'reward_per_episode',
        log_name_reward_per_step = 'reward_per_step',
        log_name_total_reward = 'total_reward',
        log_name_done_rate = 'done_rate'
    )
    
    config = eu.combine_dicts(kwargs, config, default_config)

    if config.seed is not None:
        eu.misc.seed(config.seed)

    # build instances
    my_env = eu.misc.create_object_from_config(
        config.env,
    )
    
    agent = eu.misc.create_object_from_config(
        config.agent,
        env = my_env
    )

    step = 0
    total_reward = 0
    episode = 0
    successful_episodes = 0

    while step < config.n_max_steps:
        terminated = False
        truncated = False
        
        reward_per_episode = 0
        step_per_episode = 0

        log.add_value(config.log_name_episode, episode)

        agent.start_episode(episode = episode)

        goal_position = my_env.sample_a_goal_position(training = config.training)

        obs, _ = my_env.reset(goal_position = goal_position)
        
        while not terminated and not truncated and step < config.n_max_steps:

            next_obs, reward, terminated, truncated, transition = config.step_function(obs, agent, my_env)

            if config.update_agent:
                agent.train(transition = transition, step = step)

            if terminated:
                successful_episodes+=1

            reward_per_episode += reward
            total_reward += reward
            done_rate = successful_episodes/(episode+1)

            log.add_value(config.log_name_step, step)
            log.add_value(config.log_name_reward_per_step, reward)
            log.add_value(config.log_name_episode_per_step, episode)
            log.add_value(config.log_name_done_rate, done_rate)

            obs = next_obs

            step += 1
            step_per_episode += 1
            
        agent.end_episode()
        if episode > 0 and episode%500 == 0:
            agent.save(episode, step)
            
        log.add_value(config.log_name_step_per_episode, step_per_episode)
        log.add_value(config.log_name_reward_per_episode, reward_per_episode)

        episode += 1
    
    log.save()

def step_feature_goal_agent(obs, agent, my_env):
    action = agent.choose_action(agent_position_features=obs["agent_position_features"], goal_position=obs["goal_position"], training=True)

    next_obs, reward, terminated, truncated, _ = my_env.step(action=action) 
    
    transition = (obs["agent_position_features"], obs["goal_position"], action, reward, next_obs["agent_position_features"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition 

def step_feature_goal_weight_agent(obs, agent, my_env):
    action = agent.choose_action(agent_position_features=obs["agent_position_features"], goal_position=obs["goal_position"], goal_weights = obs["goal_weights"], training=True)
            
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (obs["agent_position_features"], obs["goal_position"], obs["goal_weights"], action, reward, next_obs["agent_position_features"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition

def step_state_goal_agent(obs, agent, my_env):
    action = agent.choose_action(agent_position=obs["agent_position"], goal_position=obs["goal_position"], training=True)
            
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (obs["agent_position"], obs["goal_position"], action, reward, next_obs["agent_position"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition

def step_state_goal_weight_agent(obs, agent, my_env):
    action = agent.choose_action(agent_position=obs["agent_position"], goal_position=obs["goal_position"], goal_weights = obs["goal_weights"], training=True)
            
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (obs["agent_position"], obs["goal_position"], obs["goal_weights"], action, reward, next_obs["agent_position"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition

    
