import numpy as np
import exputils as eu
import exputils.data.logging as log
import copy
import universal_successor_features.agents as a
import torch
import universal_successor_features.envs as envs


def run_rl_first_phase(config=None, **kwargs):
    # default config
    default_config = eu.AttrDict(
        seed=None,
        env=eu.AttrDict(
            cls=None,
            nmax_steps=np.inf,
        ),
        agent=eu.AttrDict(
            cls=None,
            network=eu.AttrDict(cls=None),
        ),
        n_steps=np.inf,
        log_name_step="step",
        log_name_episode="episode",
        log_name_step_per_episode="step_per_episode",
        log_name_episode_per_step="episode_per_step",
        log_name_reward_per_episode="reward_per_episode",
        log_name_reward_per_step="reward_per_step",
        log_name_total_reward="total_reward",
        log_name_done_rate="done_rate",
        log_name_done_rate_eval="done_rate_eval",
    )

    config = eu.combine_dicts(kwargs, config, default_config)

    if config.seed is not None:
        eu.misc.seed(config.seed)

    # build instances
    my_env = eu.misc.create_object_from_config(
        config.env,
    )

    # Copy of environment for testing since I dont want to change its state
    test_env = copy.deepcopy(my_env)

    agent = eu.misc.create_object_from_config(config.agent, env=my_env)

    step = 0
    total_reward = 0
    episode = 0

    while step <= config.n_steps:
        terminated = False
        truncated = False

        reward_per_episode = 0
        step_per_episode = 0

        log.add_value(config.log_name_episode, episode)

        episode += 1
        agent.start_episode(episode=episode)

        goal_position = my_env.sample_source_goal()

        obs, _ = my_env.reset(goal_position=goal_position)

        while not terminated and not truncated and step <= config.n_steps:
            # In the first part of the training I do not use GPI so goals_so_far is
            # the single goal I am working with. If I was using GPI it would be a list
            # of goals I have seen so far.
            next_obs, reward, terminated, truncated, transition = general_step_function(
                obs, agent, my_env, goals_so_far=[goal_position], training=True
            )

            agent.train(transition=transition)

            reward_per_episode += reward
            total_reward += reward

            log.add_value(config.log_name_step, step)
            log.add_value(config.log_name_reward_per_step, reward)
            log.add_value(config.log_name_total_reward, total_reward)
            log.add_value(config.log_name_episode_per_step, episode)

            if step % 100 == 0:
                done_rate = evaluate_agent(
                    agent,
                    test_env,
                    general_step_function,
                    test_env.goal_list_source_tasks,
                    use_gpi=False,
                )
                done_rate_eval = evaluate_agent(
                    agent,
                    test_env,
                    general_step_function,
                    test_env.goal_list_evaluation_tasks,
                    use_gpi=False,
                )

            log.add_value(config.log_name_done_rate, done_rate)
            log.add_value(config.log_name_done_rate_eval, done_rate_eval)

            obs = next_obs

            step += 1
            step_per_episode += 1

        agent.end_episode()

        log.add_value(config.log_name_step_per_episode, step_per_episode)
        log.add_value(config.log_name_reward_per_episode, reward_per_episode)

    agent.save(episode=episode, step=step, total_reward=total_reward)
    my_env.save()
    log.save()


def run_rl_second_phase(config=None, **kwargs):
    default_config = eu.AttrDict(
        seed=None,
        env=eu.AttrDict(
            cls=None,
        ),
        agent=eu.AttrDict(
            cls=None,
        ),
        env_checkpoint_path=None,
        agent_checkpoint_path=None,
        n_steps=np.inf,
        use_gpi_eval=False,
        use_gpi_train=False,
        use_target_tasks=True,
        log_name_step="step",
        log_name_episode="episode",
        log_name_step_per_episode="step_per_episode",
        log_name_episode_per_step="episode_per_step",
        log_name_reward_per_episode="reward_per_episode",
        log_name_reward_per_step="reward_per_step",
        log_name_total_reward="total_reward",
        log_name_done_rate_eval="done_rate_eval",
        log_name_done_rate_source="done_rate_source",
        log_name_done_rate_combined="done_rate_combined",
    )

    config = eu.combine_dicts(kwargs, config, default_config)

    if config.seed is not None:
        eu.misc.seed(config.seed)

    # build instances
    if config.env_checkpoint_path is None:
        my_env = eu.misc.create_object_from_config(
            config.env,
        )
    else:
        my_env = config.env.cls.load_from_checkpoint(config.env_checkpoint_path)

    # Assert that source goals that agent learned are the same goals that the
    # environment has
    saved_source_goals = torch.load(config.agent_checkpoint_path)["env_goals_source"]
    assert all(
        [
            (goal1 == goal2).all()
            for goal1, goal2 in zip(my_env.goal_list_source_tasks, saved_source_goals)
        ]
    ), "The agent did not learn the goals of this environment"

    # Copy of environment for testing since I dont want to change its state
    test_env = copy.deepcopy(my_env)

    # Instantiate agent from saved checkpoint with same config
    agent = config.agent.cls.load_from_checkpoint(my_env, config.agent_checkpoint_path)

    if config.use_target_tasks:
        goal_sampler = my_env.sample_target_goal
        goal_list_for_eval = my_env.goal_list_target_tasks
    else:
        goal_sampler = my_env.sample_eval_goal
        goal_list_for_eval = my_env.goal_list_evaluation_tasks

    step = 0
    total_reward = agent.total_reward
    episode = agent.current_episode

    # Begin of Eval Phase
    agent.prepare_for_eval_phase()

    while step <= config.n_steps:
        terminated = False
        truncated = False

        reward_per_episode = 0
        step_per_episode = 0

        log.add_value(config.log_name_episode, episode)

        episode += 1
        agent.start_episode(episode=episode)

        goal_position = goal_sampler()
        goals_so_far = [goal_position]

        if config.use_gpi_train:
            goals_so_far += my_env.goal_list_source_tasks 

        obs, _ = my_env.reset(goal_position=goal_position)

        while not terminated and not truncated and step <= config.n_steps:
            next_obs, reward, terminated, truncated, transition = general_step_function(
                obs, agent, my_env, goals_so_far, training=True
            )

            agent.train_during_eval_phase(
                transition=transition, p_pick_new_memory_buffer=0.5
            )

            reward_per_episode += reward
            total_reward += reward

            log.add_value(config.log_name_step, step)
            log.add_value(config.log_name_reward_per_step, reward)
            log.add_value(config.log_name_total_reward, total_reward)
            log.add_value(config.log_name_episode_per_step, episode)

            if step % 100 == 0:
                done_rate_eval = evaluate_agent(
                    agent,
                    test_env,
                    general_step_function,
                    goal_list_for_eval,
                    use_gpi=config.use_gpi_eval,
                )
                done_rate_source = evaluate_agent(
                    agent,
                    test_env,
                    general_step_function,
                    test_env.goal_list_source_tasks,
                    use_gpi=False,
                )
            log.add_value(config.log_name_done_rate_eval, done_rate_eval)
            log.add_value(config.log_name_done_rate_source, done_rate_source)
            log.add_value(
                config.log_name_done_rate_combined,
                (done_rate_eval + done_rate_source) / 2,
            )

            obs = next_obs

            step += 1
            step_per_episode += 1

        agent.end_episode()

        log.add_value(config.log_name_step_per_episode, step_per_episode)
        log.add_value(config.log_name_reward_per_episode, reward_per_episode)

    # Only for debugging purposes, can delete later
    done_rate_eval = evaluate_agent(
                    agent,
                    test_env,
                    general_step_function,
                    goal_list_for_eval,
                    use_gpi=config.use_gpi_eval,
                )
    log.save()
    agent.save(episode=episode, step=step, total_reward=total_reward)

def evaluate_agent(agent, test_env, step_fn, goal_list_for_eval, use_gpi):
    num_goals = len(goal_list_for_eval)
    completed_goals = 0

    for goal in goal_list_for_eval:
        goals_so_far = [goal]
        if use_gpi:
            goals_so_far += test_env.goal_list_source_tasks

        terminated = False
        truncated = False
        obs, _ = test_env.reset(goal_position=goal)
        while not terminated and not truncated:
            next_obs, reward, terminated, truncated, _ = step_fn(
                obs, agent, test_env, goals_so_far=goals_so_far, training=False
            )
            obs = next_obs

        if terminated:
            completed_goals += 1

    done_rate = completed_goals / num_goals

    return done_rate

def general_step_function(obs, agent, my_env, goals_so_far, training):
    action = agent.choose_action(
        obs=obs,
        list_of_goal_positions=goals_so_far,
        training=training,
    )

    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = (
        obs["agent_position"],
        obs["agent_position_features"],
        obs["goal_position"],
        obs["goal_weights"],
        action,
        reward,
        next_obs["agent_position"],
        next_obs["agent_position_features"],
        terminated,
        truncated,
    )

    return next_obs, reward, terminated, truncated, transition

if __name__ == "__main__":
# Only for debugging, can delete later
    config = eu.AttrDict(
        # random seed for the repetition
        seed = 3487 + 0,
        env=eu.AttrDict(
            cls=envs.RoomGridWorld,
            penalization=0.0,
            reward_at_goal_position=20.0,
            nmax_steps=31,
        ),
        agent=eu.AttrDict(
            cls = a.FeatureGoalWeightAgent,
        ),

        agent_checkpoint_path = "/home/andres/inria/projects/universalSuccessorFeatures/experiments/second_phase020n/FeatureGoalWeightAgent_checkpoint.pt",
        env_checkpoint_path = "/home/andres/inria/projects/universalSuccessorFeatures/experiments/second_phase020n/env_config.cfg",
        # agent_checkpoint_path="/scratch/pictor/abermeom/projects/universalSuccessorFeatures/experiments/first_phase020n/experiments/"
        # + "experiment_000002" +"/repetition_{:06d}/".format(0) + "FeatureGoalWeightAgent" 
        # + "_checkpoint.pt",
        # env_checkpoint_path = "/scratch/pictor/abermeom/projects/universalSuccessorFeatures/experiments/first_phase020n/experiments/"
        # + "experiment_000002" +"/repetition_{:06d}/".format(0) + "env_config.cfg",


        n_steps=12000,
        use_gpi_eval = True,
        use_target_tasks = True,
    )

    run_rl_second_phase(config=config)
