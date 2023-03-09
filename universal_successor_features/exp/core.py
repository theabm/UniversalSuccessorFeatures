import numpy as np
import exputils as eu
import exputils.data.logging as log

def run_rl_training_feature_goal_agent(step_function, config = None, **kwargs):

    # default config
    default_config = eu.AttrDict(
        seed = None, 
        env = eu.AttrDict(cls = None),
        agent = eu.AttrDict(cls = None),
        goal_list = None,
        n_max_steps = np.inf,
        n_max_episodes = np.inf,
        n_max_steps_per_episode = np.inf,
        log_name_episode = 'episode',
        log_name_step = 'step',
        log_name_step_per_episode = 'step_per_episode',
        log_name_episode_per_step = 'episode_per_step',
        log_name_reward_per_episode = 'reward_per_episode',
        log_name_reward_per_step = 'reward_per_step',
        log_name_total_reward = 'total_reward',
    )
    
    config = eu.combine_dicts(kwargs, config, default_config)

    if config.seed is not None:
        eu.misc.seed(config.seed)

    # build instances
    my_env = eu.misc.create_object_from_config(
        config.env,
        nmax_steps = config.n_max_steps_per_episode
    )
    
    agent = eu.misc.create_object_from_config(
        config.agent,
        env = my_env
    )

    step = 0
    total_reward = 0
    episode = 0

    while episode < config.n_max_episodes and step < config.n_max_steps:

        reward_per_episode = 0
        step_per_episode = 0

        log.add_value(config.log_name_episode, episode)

        agent.start_episode(episode = episode)

        goal_position = my_env.sample_a_goal_position_from_list(goal_list = config.goal_list)

        obs, _ = my_env.reset(goal_position = goal_position)
        
        while not terminated and not truncated and step < config.n_max_steps:

            next_obs, reward, terminated, truncated, transition = step_function(obs = obs)

            if config.update_agent:
                agent.train(transition = transition, step = step)

            reward_per_episode += reward
            total_reward += reward

            log.add_value(config.log_name_step, step)
            log.add_value(config.log_name_reward_per_step, reward)
            log.add_value(config.log_name_episode_per_step, episode)

            obs = next_obs

            step += 1
            step_per_episode += 1

        agent.end_episode()

        log.add_value(config.log_name_step_per_episode, step_per_episode)
        log.add_value(config.log_name_reward_per_episode, reward_per_episode)

        episode += 1

def step_feature_goal_agent(obs):
    global agent
    action = agent.choose_action(agent_position_features=obs["agent_position_features"], goal_position=obs["goal_position"], training=True)

    global my_env
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action) 
    
    transition = (obs["agent_position_features"], obs["goal_position"], action, reward, next_obs["agent_position_features"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition 

def step_feature_goal_weight_agent(obs):
    global agent
    action = agent.choose_action(agent_position_features=obs["agent_position_features"], goal_position=obs["goal_position"], goal_weights = obs["goal_weights"], training=True)
            
    global my_env
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (obs["agent_position_features"], obs["goal_position"], obs["goal_weights"], action, reward, next_obs["agent_position_features"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition

def step_state_goal_agent(obs):
    global agent
    action = agent.choose_action(agent_position=obs["agent_position"], goal_position=obs["goal_position"], training=True)
            
    global my_env
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (obs["agent_position"], obs["goal_position"], action, reward, next_obs["agent_position"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition

def step_state_goal_weight_agent(obs):
    global agent
    action = agent.choose_action(agent_position=obs["agent_position"], goal_position=obs["goal_position"], goal_weights = obs["goal_weights"], training=True)
            
    global my_env
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (obs["agent_position"], obs["goal_position"], obs["goal_weights"], action, reward, next_obs["agent_position"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition

    
