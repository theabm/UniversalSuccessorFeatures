import torch
import exputils as eu
import copy

class StubFeatureGoalWeightNetwork(torch.nn.Module):
    @staticmethod
    def default_config():
        return eu.AttrDict(
            state_size=2,
            goal_size=2,
            features_size=None,
            num_actions=4,
        )

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, agent_position_features, policy_goal_position, env_goal_weights):
        # This network simply takes the agent features, repeats it 4 times in the
        # last dimension, and then multiplies the third element by 5.
        # Therefore, when calculating the max, it should always be the third action
        #
        # Expected shape (1,features_size)
        print("agent_position_features", agent_position_features)
        print("policy_goal_position", policy_goal_position)
        print("env_goal_weights", env_goal_weights)

        sf = copy.deepcopy(agent_position_features)

        sf = (sf).tile(4, 1)
        sf[2] *= 5
        sf = sf.unsqueeze(0)

        print("sf",sf)

        q = torch.sum(torch.mul(sf, (env_goal_weights).unsqueeze(1)), dim=2)
        print("q", q)

        return q, sf, env_goal_weights, agent_position_features
