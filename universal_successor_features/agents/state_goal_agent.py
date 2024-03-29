import torch
import exputils as eu
import universal_successor_features.networks as nn
import universal_successor_features.envs as envs
from universal_successor_features.agents.base_agent import BaseAgent


class StateGoalAgent(BaseAgent):
    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            network=eu.AttrDict(
                cls=nn.StateGoalPaperDQN,
                use_gdtuo=False,
                optimizer=torch.optim.Adam,
            ),
        )
        return cnf

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(
            kwargs,
            config,
            StateGoalAgent.default_config(),
        )
        super().__init__(env=env, config=self.config)

    def _build_arguments_from_obs(self, obs, goal_position):
        return {
            "agent_position": torch.tensor(obs["agent_position"])
            .to(torch.float)
            .to(self.device),
            "policy_goal_position": torch.tensor(goal_position)
            .to(torch.float)
            .to(self.device),
            "env_goal_position": torch.tensor(obs["goal_position"])
            .to(torch.float)
            .to(self.device),
        }

    @staticmethod
    def _build_target_args(batch_args):
        return {
            "agent_position": batch_args["next_agent_position_batch"],
            "policy_goal_position": batch_args["goal_position_batch"],
            "env_goal_position": batch_args["goal_position_batch"],
        }

    @staticmethod
    def _build_predicted_args(batch_args):
        return {
            "agent_position": batch_args["agent_position_batch"],
            "policy_goal_position": batch_args["goal_position_batch"],
            "env_goal_position": batch_args["goal_position_batch"],
        }


if __name__ == "__main__":
    env = envs.GridWorld()
    agent = StateGoalAgent(env)
