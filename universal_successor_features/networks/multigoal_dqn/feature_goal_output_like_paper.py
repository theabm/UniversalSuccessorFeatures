import torch
import exputils as eu

class FeatureGoalPaperDQN(torch.nn.Module):

    @staticmethod
    def default_config():
        return eu.AttrDict(
                state_size = 2,
                goal_size = 2,
                features_size = 100,
                num_actions = 4,
                )

    def __init__(self, config = None, **kwargs) -> None:
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, self.default_config())
        self.is_a_usf = False

        self.goal_position_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.config.goal_size,
                    out_features=64
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=64,
                    out_features=self.config.features_size
                    ),
                )
        self.concatenation_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=2*self.config.features_size,
                    out_features=256
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=256,
                    out_features=self.config.num_actions
                    ),
                )
    def forward(self,
                agent_position_features,
                env_goal_position,
                **kwargs
                ):
        env_goal_representation = self.goal_position_layer(env_goal_position)
        joined_representation = torch.cat(
                (agent_position_features,env_goal_representation),
                dim=1
                )
        q = self.concatenation_layer(joined_representation)

        return q, None, None, None


if __name__ == '__main__':
    my_dqn = FeatureGoalPaperDQN()
    print(my_dqn)

    rand_features = torch.rand(10,100)
    rand_goals = torch.rand(10,2)

    q, *_ = my_dqn(rand_features, rand_goals)
    print(q.shape)

    # Emulating behavior of epsilon greedy call for a single state,
    # goal pair (not a batch)
    rand_feature = torch.rand(100).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)

    q, *_ = my_dqn(rand_feature, rand_goal)
    print(q.shape)
