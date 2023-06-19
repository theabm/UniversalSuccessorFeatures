import torch
import exputils as eu
import universal_successor_features.networks as nn
import universal_successor_features.envs as envs
from universal_successor_features.agents.base_agent import BaseAgent


class StateGoalRBFAgent(BaseAgent):
    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            network=eu.AttrDict(
                cls=nn.StateGoalRBFUSF,
                use_gdtuo=False,
                optimizer=torch.optim.Adam,
            ),
        )
        return cnf

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(
            kwargs,
            config,
            StateGoalRBFAgent.default_config(),
        )
        super().__init__(env=env, config=self.config)

    def _build_arguments_from_obs(self, obs, goal_position):
        goal_position_rbf = self.env._get_rbf_vector_at(goal_position)
        return {
            "agent_position_rbf": torch.tensor(obs["agent_position_rbf"])
            .to(torch.float)
            .to(self.device),
            "policy_goal_position_rbf": torch.tensor(goal_position_rbf)
            .to(torch.float)
            .to(self.device),
            "env_goal_position_rbf": torch.tensor(obs["goal_position_rbf"])
            .to(torch.float)
            .to(self.device),
        }

    @staticmethod
    def _build_target_args(batch_args):
        return {
            "agent_position_rbf": batch_args["next_agent_position_rbf_batch"],
            "policy_goal_position_rbf": batch_args["goal_position_rbf_batch"],
            "env_goal_position_rbf": batch_args["goal_position_rbf_batch"],
        }

    @staticmethod
    def _build_predicted_args(batch_args):
        return {
            "agent_position_rbf": batch_args["agent_position_rbf_batch"],
            "policy_goal_position_rbf": batch_args["goal_position_rbf_batch"],
            "env_goal_position_rbf": batch_args["goal_position_rbf_batch"],
        }


if __name__ == "__main__":
    env = envs.GridWorld()
    agent = StateGoalRBFAgent(env)
