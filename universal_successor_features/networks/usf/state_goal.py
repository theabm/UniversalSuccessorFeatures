import torch
import exputils as eu

class StateGoalUSF(torch.nn.Module):
    
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
        self.is_a_usf = True

        self.num_actions = self.config.num_actions
        self.features_size = self.config.features_size

        self.env_goal_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.config.goal_size,
                out_features=64
                ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=64,
                out_features=64
                ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=64,
                out_features=self.config.features_size
                ),
        )

        self.agent_position_layer = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.config.state_size,
                out_features=81
                ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=81,
                out_features=self.config.features_size
                ),
        )

        self.policy_goal_layer = torch.nn.Sequential(
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
                out_features=self.config.num_actions*self.config.features_size
                ),
        )
    
    def forward(self,
                agent_position,
                policy_goal_position,
                env_goal_position
                ):
        features = self.agent_position_layer(agent_position)
        goal_position_features = self.policy_goal_layer(policy_goal_position)
        joined_representation = torch.cat(
                (features,goal_position_features),
                dim=1
                )

        # successor feature
        sf = self.concatenation_layer(joined_representation)

        batch_size = sf.shape[0]
        sf = sf.reshape(batch_size, self.num_actions, self.features_size)
        
        env_goal_weights = self.env_goal_layer(env_goal_position)

        # Output dot product between sf and env_goal_weights.
        # sf has shape (batch, num_actions, feature_size) while
        # env_goal_weights has shape (batch, feature_size)
        q = torch.sum(torch.mul(sf, env_goal_weights.unsqueeze(1)), dim=2)

        return q, sf, env_goal_weights, features


if __name__ == '__main__':
    my_dqn = StateGoalUSF()
    print(my_dqn)
    
    rand_states = torch.rand(10,2)
    rand_goals = torch.rand(10,2)

    q, *_ = my_dqn(rand_states, rand_goals, rand_goals)
    print(q.shape)
    
    # Emulating behavior of epsilon greedy call for a single state,
    # goal pair (not a batch)
    rand_state = torch.rand(2).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)

    q, *_ = my_dqn(rand_state, rand_goal, rand_goal)
    print(q.shape)
