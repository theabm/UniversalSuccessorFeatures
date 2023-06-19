import torch
import exputils as eu

class FeatureGoalWeightKUSF(torch.nn.Module):

    @staticmethod
    def default_config():
        return eu.AttrDict(
                state_size=2,
                goal_size=2,
                features_size=100,
                num_actions=4,
                )

    def __init__(self, config = None, **kwargs) -> None:
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, self.default_config())
        self.is_a_usf = True

        self.num_actions = self.config.num_actions
        self.features_size = self.config.features_size

        self.policy_goal_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.config.goal_size,
                    out_features=64
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=64,
                    out_features=64,
                    ),
                torch.nn.Linear(
                    in_features=64,
                    out_features=64,
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=64,
                    out_features=self.config.features_size
                    )
                )
        self.sf_1 = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=2*self.config.features_size,
                    out_features=512
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=512,
                    out_features=256,
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=256,
                    out_features=512,
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=512,
                    out_features=self.config.features_size
                    ),
                )
        self.sf_2 = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=2*self.config.features_size,
                    out_features=512
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=512,
                    out_features=256,
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=256,
                    out_features=512,
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=512,
                    out_features=self.config.features_size
                    ),
                )
        self.sf_3 = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=2*self.config.features_size,
                    out_features=512
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=512,
                    out_features=256,
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=256,
                    out_features=512,
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=512,
                    out_features=self.config.features_size
                    ),
                )
        self.sf_4 = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=2*self.config.features_size,
                    out_features=512
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=512,
                    out_features=256,
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=256,
                    out_features=512,
                    ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    in_features=512,
                    out_features=self.config.features_size
                    ),
                )

    def forward(self,
                features,
                policy_goal_position,
                env_goal_weights):

        goal_position_features = self.policy_goal_layer(policy_goal_position)

        joined_representation = torch.cat(
                (features,goal_position_features),
                dim=1
                )

        # successor features
        # shape will be (batch, 1, feature_size)
        sf_1 = self.sf_1(joined_representation).unsqueeze(1)
        sf_2 = self.sf_2(joined_representation).unsqueeze(1)
        sf_3 = self.sf_3(joined_representation).unsqueeze(1)
        sf_4 = self.sf_4(joined_representation).unsqueeze(1)
        
        sf = torch.cat((sf_1,sf_2,sf_3,sf_4), dim = 1)

        q = torch.sum(torch.mul(sf, env_goal_weights.unsqueeze(1)), dim=2)

        return q, sf, env_goal_weights, features

if __name__ == '__main__':
    my_dqn = FeatureGoalWeightKUSF()
    print(my_dqn)

    rand_features = torch.rand(10,100)
    rand_goals = torch.rand(10,2)
    rand_weights = torch.rand(10,100)

    q, *_ = my_dqn(rand_features, rand_goals, rand_weights)
    print(q.shape)

    # Emulating behavior of epsilon greedy call for a single state,
    # goal pair (not a batch)
    rand_feature = torch.rand(100).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)
    rand_weight = torch.rand(100).unsqueeze(0)

    q, *_ = my_dqn(rand_feature, rand_goal, rand_weight)
    print(q.shape)
