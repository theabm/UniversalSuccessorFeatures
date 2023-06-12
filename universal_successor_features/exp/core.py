import numpy as np
import exputils as eu
import exputils.data.logging as log
import copy
import universal_successor_features as usf
from collections import namedtuple

PartialTransition = namedtuple(
    "Experiences",
    (
        "agent_position",
        "agent_position_features",
        "goal",
        "goal_weights",
        "action",
        "reward",
        "next_agent_position",
        "next_agent_position_features",
        "terminated",
        "truncated",
    ),
)


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
        n_steps=48000,
        augment_data=False,
        log_name_step="step",
        log_name_episode="episode",
        log_name_step_per_episode="step_per_episode",
        log_name_episode_per_step="episode_per_step",
        log_name_reward_per_episode="reward_per_episode",
        log_name_reward_per_step="reward_per_step",
        log_name_total_reward="total_reward",
        log_name_done_rate_primary="done_rate_primary",
        log_name_done_rate_tertiary="done_rate_tertiary",
        log_name_agent="agent",
        log_name_env="env",
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

        # if I want to have data augmentation while I train, I augment over all
        # the goals possible.
        # Otherwise I simply keep my current goal.
        if config.augment_data:
            list_of_goal_positions_for_augmentation = my_env.goal_list_source_tasks
        else:
            list_of_goal_positions_for_augmentation = [goal_position]

        obs, _ = my_env.reset(goal_position=goal_position)

        while not terminated and not truncated and step <= config.n_steps:
            # In the first part of the training I do not use GPI so
            # goals_for_gpi is the single goal I am working with. If I was
            # using GPI it would be a list of goals I have seen so far.
            next_obs, reward, terminated, truncated, transition = general_step_function(
                obs, agent, my_env, goals_for_gpi=[goal_position], training=True
            )

            agent.train(
                transition=transition,
                list_of_goal_positions_for_augmentation=list_of_goal_positions_for_augmentation,
            )

            reward_per_episode += reward
            total_reward += reward

            log.add_value(config.log_name_step, step)
            log.add_value(config.log_name_reward_per_step, reward)
            log.add_value(config.log_name_total_reward, total_reward)
            log.add_value(config.log_name_episode_per_step, episode)

            if step % 100 == 0:
                done_rate_primary = evaluate_agent(
                    agent,
                    test_env,
                    general_step_function,
                    test_env.goal_list_source_tasks,
                    use_gpi=False,
                )
                done_rate_tertiary = evaluate_agent(
                    agent,
                    test_env,
                    general_step_function,
                    test_env.goal_list_evaluation_tasks,
                    use_gpi=False,
                )

            log.add_value(config.log_name_done_rate_primary, done_rate_primary)
            log.add_value(config.log_name_done_rate_tertiary, done_rate_tertiary)

            obs = next_obs

            step += 1
            step_per_episode += 1

        agent.end_episode()

        log.add_value(config.log_name_step_per_episode, step_per_episode)
        log.add_value(config.log_name_reward_per_episode, reward_per_episode)

    log.add_single_object(config.log_name_agent, agent)
    log.add_single_object(config.log_name_env, my_env)
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
        log_directory=None,
        n_steps=np.inf,
        use_gpi_eval=False,
        use_gpi_train=False,
        use_target_tasks=True,
        augment_data=False,
        log_name_step="step",
        log_name_episode="episode",
        log_name_step_per_episode="step_per_episode",
        log_name_episode_per_step="episode_per_step",
        log_name_reward_per_episode="reward_per_episode",
        log_name_reward_per_step="reward_per_step",
        log_name_total_reward="total_reward",
        log_name_done_rate_source="done_rate_source",
        log_name_done_rate_eval="done_rate_eval",
        log_name_done_rate_combined="done_rate_combined",
        log_name_agent="agent",
        log_name_env="env",
    )

    config = eu.combine_dicts(kwargs, config, default_config)

    if config.seed is not None:
        eu.misc.seed(config.seed)

    log.load(directory=config.log_directory, load_objects=True)

    # build instances
    my_env = log.get_item("env")
    agent = log.get_item("agent")

    agent_saved_source_goals = agent.env.goal_list_source_tasks
    agent_saved_target_goals = agent.env.goal_list_target_tasks
    agent_saved_evaluation_goals = agent.env.goal_list_evaluation_tasks

    assert all(
        [
            (goal1 == goal2).all()
            for goal1, goal2 in zip(
                my_env.goal_list_source_tasks, agent_saved_source_goals
            )
        ]
    ), "The agent did not learn the goals of this environment"

    assert all(
        [
            (goal1 == goal2).all()
            for goal1, goal2 in zip(
                my_env.goal_list_target_tasks, agent_saved_target_goals
            )
        ]
    ), "The agent did not learn the goals of this environment"

    assert all(
        [
            (goal1 == goal2).all()
            for goal1, goal2 in zip(
                my_env.goal_list_evaluation_tasks, agent_saved_evaluation_goals
            )
        ]
    ), "The agent did not learn the goals of this environment"

    # Copy of environment for testing since I dont want to change its state
    test_env = copy.deepcopy(my_env)

    if config.use_target_tasks:
        goal_sampler = my_env.sample_target_goal
        goal_list_for_eval = my_env.goal_list_target_tasks
    else:
        goal_sampler = my_env.sample_eval_goal
        goal_list_for_eval = my_env.goal_list_evaluation_tasks

    step = 0
    total_reward = log.get_item(config.log_name_total_reward)[-1]
    episode = log.get_item(config.log_name_episode)[-1]

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

        # goals for gpi is only for action selection or evaluation
        # it determines over which goals the gpi procedure should be used.
        # if done during action selection, this will change the experiences I
        # store in my memory buffer
        goals_for_gpi = [goal_position]

        # On the other hand, list_of_goal_positions_for_augmentation is for
        # training directly
        # It specifies the goals that I will use to augment my data.
        if config.augment_data:
            list_of_goal_positions_for_augmentation = (
                goal_list_for_eval + my_env.goal_list_source_tasks
            )
        else:
            list_of_goal_positions_for_augmentation = [goal_position]

        if config.use_gpi_train:
            goals_for_gpi += my_env.goal_list_source_tasks

        obs, _ = my_env.reset(goal_position=goal_position)

        while not terminated and not truncated and step <= config.n_steps:
            next_obs, reward, terminated, truncated, transition = general_step_function(
                obs, agent, my_env, goals_for_gpi, training=True
            )

            agent.train_during_eval_phase(
                transition=transition,
                p_pick_new_memory_buffer=0.5,
                list_of_goal_positions_for_augmentation=list_of_goal_positions_for_augmentation,
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
    # agent.save(episode=episode, step=step, total_reward=total_reward)
    # my_env.save()
    log.add_single_object("agent", agent)
    log.add_single_object("env", my_env)
    log.save()


def evaluate_agent(agent, test_env, step_fn, goal_list_for_eval, use_gpi):
    num_goals = len(goal_list_for_eval)
    completed_goals = 0

    for goal in goal_list_for_eval:
        goals_for_gpi = [goal]
        if use_gpi:
            goals_for_gpi += test_env.goal_list_source_tasks

        terminated = False
        truncated = False
        obs, _ = test_env.reset(goal_position=goal)
        while not terminated and not truncated:
            next_obs, reward, terminated, truncated, _ = step_fn(
                obs, agent, test_env, goals_for_gpi=goals_for_gpi, training=False
            )
            obs = next_obs

        if terminated:
            completed_goals += 1

    done_rate = completed_goals / num_goals

    return done_rate


def general_step_function(obs, agent, my_env, goals_for_gpi, training):
    """Choose an action and take a step in the environment.

    Creates the transition and returns it among the various arguments.
    """
    action = agent.choose_action(
        obs=obs,
        list_of_goal_positions=goals_for_gpi,
        training=training,
    )

    next_obs, reward, terminated, truncated, _ = my_env.step(action=action)

    transition = PartialTransition(
        obs["agent_position"],
        obs["agent_position_features"],
        # obs["goal_position"],
        # obs["goal_weights"],
        action,
        # reward,
        next_obs["agent_position"],
        next_obs["agent_position_features"],
        # terminated,
        # truncated,
    )

    return next_obs, reward, terminated, truncated, transition


if __name__ == "__main__":
    # Only for debugging, can delete later
    config = eu.AttrDict(
        # random seed for the repetition
        seed=3487 + 0,
        env=eu.AttrDict(
            cls=usf.envs.RoomGridWorld,
            penalization=0.0,
            reward_at_goal_position=20.0,
            nmax_steps=31,
        ),
        agent=eu.AttrDict(
            cls=usf.agents.FeatureGoalAgent,
            network=eu.AttrDict(
                cls=usf.networks.FeatureGoalUSF,
            ),
            loss_weight_q=1.0,
            loss_weight_psi=0.01,
            loss_weight_phi=0.0,
            discount_factor=0.90,
            batch_size=32,
            learning_rate=5e-4,
            epsilon=eu.AttrDict(value=0.25),
            memory=eu.AttrDict(
                cls=usf.memory.ExperienceReplayMemory,
                alpha=None,
                beta0=None,
            ),
        ),
        n_steps=48000,
    )

    run_rl_first_phase(config=config)
