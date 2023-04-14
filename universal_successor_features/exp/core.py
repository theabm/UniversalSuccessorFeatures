import numpy as np
import exputils as eu
import exputils.data.logging as log
import copy

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
        step_function = None,
        n_max_steps_train = np.inf,
        n_max_steps_total = np.inf,
        log_name_step = 'step',

        log_name_episode = 'episode',
        log_name_step_per_episode = 'step_per_episode',
        log_name_episode_per_step = 'episode_per_step',
        log_name_reward_per_episode = 'reward_per_episode',
        log_name_reward_per_step = 'reward_per_step',
        log_name_total_reward = 'total_reward',
        log_name_done_rate = 'done_rate',

        log_name_episode_eval = 'episode_eval',
        log_name_step_per_episode_eval = 'step_per_episode_eval',
        log_name_episode_per_step_eval = 'episode_per_step_eval',
        log_name_reward_per_episode_eval = 'reward_per_episode_eval',
        log_name_reward_per_step_eval = 'reward_per_step_eval',
        log_name_total_reward_eval = 'total_reward_eval',
        log_name_done_rate_eval = 'done_rate_eval',
    )
    
    config = eu.combine_dicts(kwargs, config, default_config)

    if config.seed is not None:
        eu.misc.seed(config.seed)

    # build instances
    my_env = eu.misc.create_object_from_config(
        config.env,
    )
    test_env = copy.deepcopy(my_env) # Copy of environment for testing
    
    agent = eu.misc.create_object_from_config(
        config.agent,
        env = my_env
    )

    step = 0
    total_reward = 0
    episode = 0

    while step <= config.n_max_steps_train:
        terminated = False
        truncated = False
        
        reward_per_episode = 0
        step_per_episode = 0

        log.add_value(config.log_name_episode, episode)
        log.add_value(config.log_name_episode_eval, episode)

        agent.start_episode(episode = episode)

        goal_position = my_env.sample_source_goal()

        obs, _ = my_env.reset(goal_position = goal_position)
        
        while not terminated and not truncated and step <= config.n_max_steps_train:

            next_obs, reward, terminated, truncated, transition = config.step_function(obs, agent, my_env, training = True)

            agent.train(transition = transition)

            reward_per_episode += reward
            total_reward += reward

            log.add_value(config.log_name_step, step)
            log.add_value(config.log_name_reward_per_step, reward)
            log.add_value(config.log_name_total_reward, total_reward)
            log.add_value(config.log_name_episode_per_step, episode)

            log.add_value(config.log_name_reward_per_step_eval, reward)
            log.add_value(config.log_name_total_reward_eval, total_reward)
            log.add_value(config.log_name_episode_per_step_eval, episode)

            if step%100 == 0:
                done_rate = evaluate_agent(agent, test_env, config.step_function, test_env.goal_list_source_tasks)
                done_rate_eval = evaluate_agent(agent, test_env, config.step_function, test_env.goal_list_evaluation_tasks)

            log.add_value(config.log_name_done_rate, done_rate)
            log.add_value(config.log_name_done_rate_eval, done_rate_eval)

            obs = next_obs

            step += 1
            step_per_episode += 1

        agent.end_episode()
            
        log.add_value(config.log_name_step_per_episode, step_per_episode)
        log.add_value(config.log_name_reward_per_episode, reward_per_episode)

        log.add_value(config.log_name_step_per_episode_eval, step_per_episode)
        log.add_value(config.log_name_reward_per_episode_eval, reward_per_episode)

        episode += 1


##########################################################################################

################# BEGIN TESTING ON TARGET SET OF GOALS ###################################

##########################################################################################

    done_rate = evaluate_agent(agent, test_env, config.step_function, test_env.goal_list_target_tasks)

    # Copy state of agent for evaluation on tertiary set of goals later
    agent_for_eval=copy.deepcopy(agent)
    episode_eval = episode
    step_eval = step
    total_reward_eval = total_reward

    # Begin of Eval Phase

    agent.prepare_for_eval_phase()
    
    while step <= config.n_max_steps_total:
        terminated = False
        truncated = False
        
        reward_per_episode = 0
        step_per_episode = 0

        log.add_value(config.log_name_episode, episode)

        agent.start_episode(episode = episode)

        goal_position = my_env.sample_target_goal()

        obs, _ = my_env.reset(goal_position = goal_position)
        
        while not terminated and not truncated and step <= config.n_max_steps_total:

            next_obs, reward, terminated, truncated, transition = config.step_function(obs, agent, my_env, training = True)

            if config.update_agent:
                agent.train_during_eval_phase(transition = transition, p_pick_new_memory_buffer = 0.5)

            reward_per_episode += reward
            total_reward += reward

            log.add_value(config.log_name_step, step)
            log.add_value(config.log_name_reward_per_step, reward)
            log.add_value(config.log_name_total_reward, total_reward)
            log.add_value(config.log_name_episode_per_step, episode)

            if step%100 == 0:
                done_rate = evaluate_agent(agent, test_env, config.step_function, test_env.goal_list_target_tasks)

            log.add_value(config.log_name_done_rate, done_rate)

            obs = next_obs

            step += 1
            step_per_episode += 1

        agent.end_episode()
            
        log.add_value(config.log_name_step_per_episode, step_per_episode)
        log.add_value(config.log_name_reward_per_episode, reward_per_episode)

        episode += 1

##########################################################################################

################# BEGIN TESTING ON TERTIARY SET OF GOALS #################################

##########################################################################################

    agent_for_eval.prepare_for_eval_phase()
    episode = episode_eval
    step = step_eval

    total_reward = total_reward_eval

    while step <= config.n_max_steps_total:
        terminated = False
        truncated = False
        
        reward_per_episode = 0
        step_per_episode = 0

        log.add_value(config.log_name_episode_eval, episode)

        agent_for_eval.start_episode(episode = episode)

        goal_position = my_env.sample_eval_goal()

        obs, _ = my_env.reset(goal_position = goal_position)
        
        while not terminated and not truncated and step <= config.n_max_steps_total:

            next_obs, reward, terminated, truncated, transition = config.step_function(obs, agent_for_eval, my_env, training = True)

            if config.update_agent:
                agent_for_eval.train_during_eval_phase(transition = transition, p_pick_new_memory_buffer = 0.5)

            reward_per_episode += reward
            total_reward += reward

            log.add_value(config.log_name_reward_per_step_eval, reward)
            log.add_value(config.log_name_total_reward_eval, total_reward)
            log.add_value(config.log_name_episode_per_step_eval, episode)

            if step%100 == 0:
                done_rate_eval = evaluate_agent(agent_for_eval, test_env, config.step_function, test_env.goal_list_evaluation_tasks)

            log.add_value(config.log_name_done_rate_eval, done_rate_eval)

            obs = next_obs

            step += 1
            step_per_episode += 1

        agent.end_episode()
            
        log.add_value(config.log_name_step_per_episode_eval, step_per_episode)
        log.add_value(config.log_name_reward_per_episode_eval, reward_per_episode)

        episode += 1

    log.save()

def evaluate_agent(agent, test_env, step_fn, goal_list):
    num_goals = len(goal_list)
    completed_goals = 0
    for goal in goal_list:
        terminated = False
        truncated = False
        obs, _ = test_env.reset(goal_position = goal)
        while not terminated and not truncated:

            next_obs, reward, terminated, truncated, _ = step_fn(obs, agent, test_env, training = False)
            obs = next_obs

        if terminated:
            completed_goals+=1
        
    done_rate = completed_goals/num_goals

    return done_rate


def step_feature_goal_agent(obs, agent, my_env, training):
    action = agent.choose_action(agent_position_features=obs["agent_position_features"], list_of_goal_positions=[obs["goal_position"]], training=training)

    next_obs, reward, terminated, truncated, _ = my_env.step(action=action) 
    
    transition = (obs["agent_position_features"], obs["goal_position"], action, reward, next_obs["agent_position_features"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition 

def step_feature_goal_weight_agent(obs, agent, my_env, training):
    action = agent.choose_action(agent_position_features=obs["agent_position_features"], list_of_goal_positions=[obs["goal_position"]], goal_weights = obs["goal_weights"], training=training)
            
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (obs["agent_position_features"], obs["goal_position"], obs["goal_weights"], action, reward, next_obs["agent_position_features"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition

def step_state_goal_agent(obs, agent, my_env, training):
    action = agent.choose_action(agent_position=obs["agent_position"], goal_position=obs["goal_position"], training=training)
            
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (obs["agent_position"], obs["goal_position"], action, reward, next_obs["agent_position"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition

def step_state_goal_weight_agent(obs, agent, my_env, training):
    action = agent.choose_action(agent_position=obs["agent_position"], goal_position=obs["goal_position"], goal_weights = obs["goal_weights"], training=training)
            
    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (obs["agent_position"], obs["goal_position"], obs["goal_weights"], action, reward, next_obs["agent_position"], terminated, truncated)

    return next_obs, reward, terminated, truncated, transition

    
