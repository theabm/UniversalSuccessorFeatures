import torch
import exputils as eu
import universal_successor_features.networks as nn
import universal_successor_features.envs as envs
from universal_successor_features.agents.base_agent import BaseAgent


class FeatureGoalWeightAgent(BaseAgent):
    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            network=eu.AttrDict(
                cls=nn.FeatureGoalWeightUSF,
                use_gdtuo=False,
                optimizer=torch.optim.Adam,
            ),
        )
        return cnf

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(
            kwargs,
            config,
            FeatureGoalWeightAgent.default_config(),
        )
        super().__init__(env=env, config=self.config)

    def _build_arguments_from_obs(self, obs):
        return {
            "agent_position_features": torch.tensor(obs["agent_position_features"])
            .to(torch.float)
            .to(self.device),
            "env_goal_weights": torch.tensor(obs["goal_weights"])
            .to(torch.float)
            .to(self.device),
        }

    @staticmethod
    def _build_target_args(batch_args):
        return {
            "agent_position_features": batch_args["next_agent_position_features_batch"],
            "policy_goal_position": batch_args["goal_batch"],
            "env_goal_weights": batch_args["goal_weights_batch"],
        }

    @staticmethod
    def _build_predicted_args(batch_args):
        return {
            "agent_position_features": batch_args["agent_position_features_batch"],
            "policy_goal_position": batch_args["goal_batch"],
            "env_goal_weights": batch_args["goal_weights_batch"],
        }

if __name__ == "__main__":
    env = envs.GridWorld()
    agent = FeatureGoalWeightAgent(env)
