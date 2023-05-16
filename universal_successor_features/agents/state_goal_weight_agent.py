import torch
import exputils as eu
import universal_successor_features.networks as nn
import universal_successor_features.envs as envs
from universal_successor_features.agents.base_agent import BaseAgent

class StateGoalWeightAgent(BaseAgent):
    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            network=eu.AttrDict(
                cls=nn.StateGoalWeightUSF,
                optimizer=torch.optim.Adam,
            ),
        )
        return cnf

    def __init__(self, env, config=None, **kwargs):
        self.config = eu.combine_dicts(
            kwargs,
            config,
            StateGoalWeightAgent.default_config(),
        )
        super().__init__(env=env, config=self.config)

    def _build_arguments_from_obs(self, obs):
        return {
            "agent_position": torch.tensor(obs["agent_position"])
            .to(torch.float)
            .to(self.device),
            "env_goal_weights": torch.tensor(obs["goal_weights"])
            .to(torch.float)
            .to(self.device),
        }

    def _build_target_batch(self, batch_args):
        terminated_batch = batch_args["terminated_batch"]
        reward_batch = batch_args["reward_batch"]

        if self.is_a_usf:
            with torch.no_grad():
                q, sf_s_g, w, reward_phi_batch = self.target_net(
                    agent_position=batch_args["next_agent_position_batch"],
                    policy_goal_position=batch_args["goal_batch"],
                    env_goal_weights=batch_args["goal_weights_batch"],
                )

                qm, action = torch.max(q, axis=1)
                # shape (batch_size,)
                target_q = reward_batch + self.discount_factor * torch.mul(
                    qm, ~terminated_batch
                )

                terminated_batch = terminated_batch.unsqueeze(1)
                # shape (batch_size,1,n)
                action = (
                    action.reshape(self.batch_size, 1, 1)
                    .tile(self.features_size)
                    .to(self.device)
                )
                # shape (batch, features_size)
                target_psi = reward_phi_batch + self.discount_factor * torch.mul(
                    sf_s_g.gather(1, action).squeeze(), ~terminated_batch
                )

            return target_q, target_psi, reward_batch

        else:
            with torch.no_grad():
                q, *_ = self.target_net(
                    agent_position=batch_args["next_agent_position_batch"],
                    policy_goal_position=batch_args["goal_batch"],
                    env_goal_weights=batch_args["goal_weights_batch"],
                )
                q, _ = torch.max(q, axis=1)
                target_q = reward_batch + self.discount_factor * torch.mul(
                    q, ~terminated_batch
                )

            return target_q

    def _build_predicted_batch(self, batch_args):
        action_batch = batch_args["action_batch"]

        if self.is_a_usf:
            q, sf_s_g, w, phi = self.policy_net(
                agent_position=batch_args["agent_position_batch"],
                policy_goal_position=batch_args["goal_batch"],
                env_goal_weights=batch_args["goal_weights_batch"],
            )
            # shape (batch_size,)
            predicted_q = q.gather(1, action_batch).squeeze()

            action_batch = action_batch.reshape(self.batch_size, 1, 1).tile(
                self.features_size
            )
            # shape (batch_size, features_size)
            predicted_psi = sf_s_g.gather(1, action_batch).squeeze()

            return predicted_q, predicted_psi, torch.sum(phi * w, dim=1)

        else:
            predicted_q, *_ = self.policy_net(
                agent_position=batch_args["agent_position_batch"],
                policy_goal_position=batch_args["goal_batch"],
                env_goal_weights=batch_args["goal_weights_batch"],
            )

            return predicted_q.gather(1, action_batch).squeeze()

if __name__ == "__main__":
    env = envs.GridWorld()
    agent = StateGoalWeightAgent(env)
