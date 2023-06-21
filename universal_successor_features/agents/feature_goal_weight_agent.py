import torch
import exputils as eu
import universal_successor_features.networks as nn
import universal_successor_features.envs as envs
from universal_successor_features.agents.base_agent import (
    BaseAgent,
    FullTransition,
    Experiences,
)
import numpy as np


class FeatureGoalWeightAgent(BaseAgent):
    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            network=eu.AttrDict(
                cls=nn.FeatureGoalWeightUSF,
                use_gdtuo=False,
                optimizer=torch.optim.Adam,
            ),
            augment_data=False,
        )
        return cnf

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(
            kwargs,
            config,
            FeatureGoalWeightAgent.default_config(),
        )
        super().__init__(env=env, config=self.config)

        # override the sample experiences and update memory functions
        # I do this since I dont want this to happen automatically.
        if self.config.augment_data:
            self._sample_experiences = self._sample_experiences_augmented
            self._update_memory = self._update_memory_augmented

    def _build_arguments_from_obs(self, obs, goal_position):
        return {
            "features": torch.tensor(obs["features"]).to(torch.float).to(self.device),
            "policy_goal_position": torch.tensor(goal_position)
            .to(torch.float)
            .to(self.device),
            "env_goal_weights": torch.tensor(obs["goal_weights"])
            .to(torch.float)
            .to(self.device),
        }

    @staticmethod
    def _build_target_args(batch_args):
        return {
            "features": batch_args["next_features_batch"],
            "policy_goal_position": batch_args["goal_position_batch"],
            "env_goal_weights": batch_args["goal_weights_batch"],
        }

    @staticmethod
    def _build_predicted_args(batch_args):
        return {
            "features": batch_args["features_batch"],
            "policy_goal_position": batch_args["goal_position_batch"],
            "env_goal_weights": batch_args["goal_weights_batch"],
        }

    def _update_memory_augmented(
        self, td_error, list_of_goal_positions_for_augmentation
    ):
        td_error = td_error.reshape(
            (self.batch_size, len(list_of_goal_positions_for_augmentation))
        ).mean(dim=1)

        assert td_error.shape == (self.batch_size,)

        super()._update_memory(td_error, list_of_goal_positions_for_augmentation)

    def _sample_experiences_augmented(self, list_of_goal_positions_for_augmentation):
        # list of goal positions holds all the goals over which I want
        # to augment my training..

        assert type(list_of_goal_positions_for_augmentation) == list

        len_list_goals = len(list_of_goal_positions_for_augmentation)

        # this is the real batch size I need to work with.
        self._augmented_batch_size = len_list_goals * self.batch_size

        # first I sample batch_size experiences.
        # these experiences are named tuples of the following structure:
        # (agent_position, agent_position_features, action, next_agent_position,
        # next_agent_position_features)

        # Here I must always use the true batch size I have determined
        experiences, weights = self.memory.sample(self.batch_size)

        assert type(experiences) == list
        assert len(experiences) == self.batch_size

        augmented_experiences = self._augment_experiences(
            experiences, list_of_goal_positions_for_augmentation
        )

        augmented_weights = self._augment_weights(weights, len_list_goals)

        return Experiences(*zip(*augmented_experiences)), torch.tensor(
            augmented_weights
        )

    def _augment_experiences(
        self, experiences, list_of_goal_positions_for_augmentation
    ):
        # Then, from these, we will construct a full transition tuple for each
        # of the goals by doing the following:
        # goal - given by goal list
        # goal_weight - calculated by environment
        # reward - goal_weight*next_agent_position_features
        # terminated - true if position == goal
        # truncated - we dont need for training.

        augmented_experiences = []

        for experience in experiences:
            for goal_position in list_of_goal_positions_for_augmentation:
                goal_weights = self.env._get_goal_weights_at(goal_position)
                goal_position_rbf = self.env._get_rbf_vector_at(goal_position)
                assert goal_weights.shape == (1, self.features_size)

                reward = int(np.sum(experience.next_features * goal_weights))
                assert type(reward) == int

                terminated = (
                    True
                    if (goal_position == experience.next_agent_position).all()
                    else False
                )
                assert type(terminated) == bool

                new_experience = FullTransition(
                    experience.agent_position,
                    experience.agent_position_rbf,
                    experience.features,
                    goal_position,
                    goal_position_rbf,
                    goal_weights,
                    experience.action,
                    reward,
                    experience.next_agent_position,
                    experience.next_agent_position_rbf,
                    experience.next_features,
                    terminated,
                    experience.truncated,
                )

                augmented_experiences.append(new_experience)

        assert len(augmented_experiences) == self._augmented_batch_size

        return augmented_experiences

    def _augment_weights(self, weights, len_list_goals):
        assert weights.shape == (self.batch_size,)

        # We take the weights, reshape to (batch,1) so that we can tile in the
        # first dimension, then tile to obtain (batch, 12) then we reshape to
        # batch*12.

        augmented_weights = weights.reshape((self.batch_size, 1))

        augmented_weights = np.tile(augmented_weights, (1, len_list_goals))

        assert augmented_weights.shape == (self.batch_size, len_list_goals)

        augmented_weights = augmented_weights.reshape((self._augmented_batch_size,))

        assert augmented_weights.shape == (self._augmented_batch_size,)

        return augmented_weights


if __name__ == "__main__":
    env = envs.GridWorld()
    agent = FeatureGoalWeightAgent(env)
