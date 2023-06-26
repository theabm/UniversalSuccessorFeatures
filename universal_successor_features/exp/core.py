import numpy as np
import exputils as eu
import exputils.data.logging as log
import copy
import universal_successor_features as usf
from collections import namedtuple
import warnings

FullTransition = namedtuple(
    "FullTransition",
    (
        "agent_position",
        "agent_position_rbf",
        "features",
        "goal_position",
        "goal_position_rbf",
        "goal_weights",
        "action",
        "reward",
        "next_agent_position",
        "next_agent_position_rbf",
        "next_features",
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
        log_directory=None,
        log_name_step="step",
        log_name_episode="episode",
        log_name_step_per_episode="step_per_episode",
        log_name_episode_per_step="episode_per_step",
        log_name_reward_per_episode="reward_per_episode",
        log_name_reward_per_step="reward_per_step",
        log_name_total_reward="total_reward",
        log_name_done_rate_primary="done_rate_primary",
        log_name_done_rate_tertiary="done_rate_tertiary",
        log_name_agent="agent.pt",
        log_name_env="env",
    )

    config = eu.combine_dicts(kwargs, config, default_config)

    if config.seed is not None:
        eu.misc.seed(config.seed)

    if config.log_directory:

        warnings.warn("Using pretrained agent... loading env")

        # if config.log_directory is specified, then I will train my agent
        # using another pretrained agent as behavioral policy, i.e. to
        # select actions.
        # Since this agent was pretrained on an environment, we need to
        # use that same environment for the training of the current agent.

        # log.load(directory=config.log_directory, load_objects=True)
        # my_env = log.get_item(config.log_name_env)

        my_env = usf.envs.GridWorld.load_from_checkpoint(
            config.log_directory + config.log_name_env
        )

        warnings.warn("Using pretrained agent... loading agent")

        # if config.log_directory is specified, we will use a pretrained
        # agent as a behavioral policy.

        # note that the pretrained agent is supposed to be an "optimal" agent
        # that has already learned to solve the task and will select optimal
        # actions from the space.

        # This is useful to understand whether our current agent can actually
        # learn with the current architecture present.

        # In our case, since we want to understand whether the agent can learn
        # the true psi function (without no q loss), we first train an agent
        # only with q loss (since we have seen that this is easy to do) and
        # then use this agent as a behavioral policy that selects the actions
        # the agent will observe and subsequently, train on.

        # behavioral_agent = log.get_item(config.log_name_agent)

        optimal_agent = usf.agents.BaseAgent.load_from_checkpoint(
            my_env, config.log_directory + config.log_name_agent
        )

        # After having obtained the agent, we reset the log so that
        # we dont have the data from the previous agent as well.

        # log.reset()

        # we assert that the saved goals of the agent and the environment 
        # are the same (should be guaranteed but better safe than sorry)

        assert all(
            [
                (goal1 == goal2).all()
                for goal1, goal2 in zip(
                    my_env.goal_list_source_tasks, optimal_agent._env_primary_goals
                )
            ]
        ), "The agent did not learn the goals of this environment"

        assert all(
            [
                (goal1 == goal2).all()
                for goal1, goal2 in zip(
                    my_env.goal_list_target_tasks, optimal_agent._env_secondary_goals
                )
            ]
        ), "The agent did not learn the goals of this environment"

        assert all(
            [
                (goal1 == goal2).all()
                for goal1, goal2 in zip(
                    my_env.goal_list_evaluation_tasks,
                    optimal_agent._env_tertiary_goals,
                )
            ]
        ), "The agent did not learn the goals of this environment"

        # we set the training parameter for the general step function. This
        # will get fed to agent.choose_action(). When we are using a pretrained
        # agent, we dont want any exploration and want to get the optimal
        # actions each time.
        # Update: setting training = False and actually running the exp.
        # results in an agent that is not able to learn. This is because the
        # agent only sees a specific subset of the state,action space.
        # Since the agent initially has random values for every (s,a)
        # combination, this means that only for a subset of them, it learns
        # the true value. However, the rest of (s,a) pairs are never updated
        # and remain arbitrarily high/low. These values can lead the agent
        # to select bad actions.
        # a first solution is setting training = True
        # this means the optimal agent will still do some exploration.
        # Update: unsurprisingly, it doesnt work. so now we switch to using 
        # the psi of the optimal agent as target but letting the agent still 
        # do its own stuff.
        config.agent.optimal_target_net = optimal_agent.target_net

    else:
        warnings.warn("Using new agent... creating env")
        # otherwise, I create an environment from scratch.
        my_env = eu.misc.create_object_from_config(
            config.env,
        )

    # Copy of environment for evaluation since I dont want to change its
    # internal state.
    test_env = copy.deepcopy(my_env)

    # the agent that will be trained during the episodes of the environment
    agent = eu.misc.create_object_from_config(config.agent, env=my_env)

    list_of_goal_positions_for_augmentation = my_env.goal_list_source_tasks

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
            # In the first part of the training I do not use GPI so
            # goals_for_gpi is the single goal I am working with. If I was
            # using GPI it would be a list of goals I have seen so far.
            next_obs, reward, terminated, truncated, transition = general_step_function(
                obs,
                agent,
                my_env,
                goals_for_gpi=[goal_position],
                training=True,
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

    # log.add_single_object(config.log_name_agent, agent)
    # log.add_single_object(config.log_name_env, my_env)
    log.save()

    my_env.save(config.log_name_env)
    agent.save(config.log_name_agent, episode, step, total_reward)


def run_rl_second_phase(config=None, **kwargs):
    default_config = eu.AttrDict(
        seed=None,
        log_directory=None,
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
        log_name_done_rate_source="done_rate_source",
        log_name_done_rate_eval="done_rate_eval",
        log_name_done_rate_combined="done_rate_combined",
        log_name_agent="agent.pt",
        log_name_env="env",
    )

    config = eu.combine_dicts(kwargs, config, default_config)

    if config.seed is not None:
        eu.misc.seed(config.seed)

    # log.load(directory=config.log_directory, load_objects=True)

    # build instances
    # my_env = log.get_item("env")
    # agent = log.get_item("agent")

    my_env = usf.envs.GridWorld.load_from_checkpoint(
        config.log_directory + config.log_name_env
    )
    agent = usf.agents.BaseAgent.load_from_checkpoint(
        my_env, config.log_directory + config.log_name_agent
    )

    agent_saved_source_goals = agent._env_primary_goals
    agent_saved_target_goals = agent._env_secondary_goals
    agent_saved_evaluation_goals = agent._env_tertiary_goals

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

    list_of_goal_positions_for_augmentation = (
        my_env.goal_list_source_tasks + goal_list_for_eval
    )

    step = 0
    total_reward = 0
    episode = 0

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

        # On the other hand, list_of_goal_positions_for_augmentation is for
        # training directly
        # It specifies the goals that I will use to augment my data.
        # it only applies to feature goal weight agent. Since it decreases the 
        # performance, it is not used.
        goals_for_gpi = [goal_position]

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
    # log.add_single_object("agent", agent)
    # log.add_single_object("env", my_env)
    log.save()

    my_env.save(config.log_name_env)
    agent.save(config.log_name_agent, episode, step, total_reward)

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

    # Note: In the normal case, this works fine.
    # When I do data augmentation however, I will still use this transition,
    # but I will only work with the entries that do not change across goal
    # i.e. pos, feaures, action
    transition = FullTransition(
        obs["agent_position"],
        obs["agent_position_rbf"],
        obs["features"],
        obs["goal_position"],
        obs["goal_position_rbf"],
        obs["goal_weights"],
        action,
        reward,
        next_obs["agent_position"],
        next_obs["agent_position_rbf"],
        next_obs["features"],
        terminated,
        truncated,
    )

    return next_obs, reward, terminated, truncated, transition


if __name__ == "__main__":
    pass
