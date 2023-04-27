import torch
import exputils as eu


class FeatureGoalUSF(torch.nn.Module):
    @staticmethod
    def default_config():
        return eu.AttrDict(
            state_size=2,
            goal_size=2,
            features_size=100,
            num_actions=4,
        )

    def __init__(self, config=None, **kwargs) -> None:
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        self.num_actions = self.config.num_actions
        self.features_size = self.config.features_size

        self.layer_goal_weights = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.config.goal_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=self.config.features_size),
        )

        self.layer_goal = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.config.goal_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=self.config.features_size),
        )
        self.layer_concat = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=2 * self.config.features_size, out_features=256
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=256,
                out_features=self.config.num_actions * self.config.features_size,
            ),
        )

    def forward(self, agent_position_features, policy_goal_position, env_goal_position):
        phi_g = self.layer_goal(policy_goal_position)
        rep = torch.cat((agent_position_features, phi_g), dim=1)
        sf_s_g = self.layer_concat(rep)

        batch_size = sf_s_g.shape[0]
        sf_s_g = sf_s_g.reshape(batch_size, self.num_actions, self.features_size)

        env_goal_weights = self.layer_goal_weights(env_goal_position)

        q = torch.sum(torch.mul(sf_s_g, env_goal_weights.unsqueeze(1)), dim=2)

        return q, sf_s_g, env_goal_weights, agent_position_features


if __name__ == "__main__":
    my_dqn = FeatureGoalUSF()
    print(my_dqn)

    rand_features = torch.rand(10, 100)
    rand_goals = torch.rand(10, 2)

    q, *_ = my_dqn(rand_features, rand_goals, rand_goals)
    print(q.shape)

    # Emulating behavior of epsilon greedy call
    # for a single state, goal pair (not a batch)
    rand_feature = torch.rand(100).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)

    q, *_ = my_dqn(rand_feature, rand_goal, rand_goal)
    print(q.shape)
