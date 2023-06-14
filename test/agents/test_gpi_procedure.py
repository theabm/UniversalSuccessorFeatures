import universal_successor_features.envs as envs
import universal_successor_features.memory as mem
import universal_successor_features.agents as a
import universal_successor_features.networks as nn
import universal_successor_features.exp as exp
import pytest
import exputils as eu
import numpy as np

import test.agents.utils as u  # for pytest

# import utils as u  # for python


# QUESTION: If I modify nmax_steps for the environment to be 31,
#           I need 1000 steps to learn the Q function well.
#           On the other hand, if (like in the other tests)
#           I leave nmax_steps at 1e6, the Q function is learnt in 500 steps.
# Answer: with 1e6 steps, the agent is able to find the reward and learn from 
# this information sooner. 

@pytest.mark.parametrize(
    "agent_type, network, memory, n_steps",
    [
        (a.FeatureGoalAgent, nn.FeatureGoalUSF, mem.ExperienceReplayMemory, 1000),
        (
            a.FeatureGoalWeightAgent,
            nn.FeatureGoalWeightUSF,
            mem.ExperienceReplayMemory,
            1000,
        ),
    ],
)
def test_gpi(agent_type, network, memory, n_steps, seed=0):
    if seed is not None:
        eu.misc.seed(seed)

    my_env = envs.GridWorld(
        rows=3,
        columns=3,
        penalization=0,
        reward_at_goal_position=1,
        nmax_steps=31,
        n_goals = 1
    )

    agent = agent_type(
        env=my_env,
        epsilon={"value": 1.0},
        train_for_n_iterations=2,
        discount_factor=0.5,
        network={"cls": network},
        memory=eu.AttrDict(cls=memory, alpha=0.5, beta0=0.5, schedule_length=n_steps),
    )

    if isinstance(agent, a.FeatureGoalAgent):
        use_pos = False
        use_weight = False
    elif isinstance(agent, a.FeatureGoalWeightAgent):
        use_pos = False
        use_weight = True
    elif isinstance(agent, a.StateGoalAgent):
        use_pos = True
        use_weight = False
    elif isinstance(agent, a.StateGoalWeightAgent):
        use_pos = True
        use_weight = True

    cmp = u.test_training(
        agent,
        my_env,
        n_steps,
        u.q_ground_truth,
        exp.general_step_function,
        use_pos=use_pos,
        use_weight=use_weight,
    )

    assert cmp

    # After having learned policy for 2 goals, I would like to see if GPI enables me
    # to solve a third goal

    # Learned:
    # o o o # o o o
    # o o o # o o o
    # o o g # o o g

    # New Goal: Any of the goals in the grid that arent the two previous goals
    # For example
    # o o o
    # o o g
    # o o o

    start_position = np.array([[0, 0]])

    goals_so_far = [np.array([[2, 2]]), np.array([[2, 0]])]
    goal_positions_3 = [
        np.array([[0, 1]]),
        np.array([[0, 2]]),
        np.array([[1, 0]]),
        # np.array([[1, 1]]),
        np.array([[1, 2]]),
        # np.array([[2, 1]]),
    ]
    for goal_position_3 in goal_positions_3:
        step = 0
        episode = 0

        terminated = False
        truncated = False

        obs, _ = my_env.reset(
            start_agent_position=start_position, goal_position=goal_position_3
        )
        agent.start_episode(episode=episode)

        while not terminated and not truncated:
            (
                next_obs,
                reward,
                terminated,
                truncated,
                transition,
            ) = exp.general_step_function(obs, agent, my_env, goals_so_far, False)
            my_env.render()

            obs = next_obs
            step += 1

        agent.end_episode()
        print(f"Goal: {goal_position_3}\tSteps: {step}")

        assert terminated
