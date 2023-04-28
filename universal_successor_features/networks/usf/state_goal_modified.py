import torch
import exputils as eu
import numpy as np

class StateGoalUSFModified(torch.nn.Module):
    
    @staticmethod
    def default_config():
        return eu.AttrDict(
           state_size = 2,
           goal_size = 2,
           features_size = 100,
           num_actions = 4,
        )
    
    def positions_to_feature(self, positions):
        positions = positions.to(torch.int)
        features = torch.zeros(positions.shape[0], self.features_size).to(positions.device)

        for i in range(positions.shape[0]):
            features[i][positions[i][0]*9 + positions[i][1]] = 1
        return features
        
    def __init__(self, config = None, **kwargs) -> None:
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
            torch.nn.Linear(in_features=2*self.config.features_size, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=self.config.num_actions*self.config.features_size),
        )
    
    def forward(self, agent_position, policy_goal_position, env_goal_position):
        agent_position_features = self.positions_to_feature(agent_position)
        goal_position_features = self.layer_goal(policy_goal_position)
        joined_representations = torch.cat((agent_position_features,goal_position_features),dim=1)
        sf = self.layer_concat(joined_representations)

        N = sf.shape[0]
        sf = sf.reshape(N, self.num_actions, self.features_size)
        
        env_goal_weigths = self.layer_goal_weights(env_goal_position)

        q = torch.sum(torch.mul(sf, env_goal_weigths.unsqueeze(1)), dim=2)

        return q, sf, env_goal_weigths, agent_position_features


if __name__ == '__main__':
    my_dqn = StateGoalUSFModified(features_size = 81)
    print(my_dqn)
    
    rand_states = torch.randint(low = 0, high = 9, size = (10,2))
    rand_goals = torch.rand(10,2)

    q, *_ = my_dqn(rand_states, rand_goals, rand_goals)
    print(q.shape)
    
    # Emulating behavior of epsilon greedy call for a single state, goal pair (not a batch)
    rand_state = torch.randint(low = 0, high = 9, size = (2,)).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)

    q, *_ = my_dqn(rand_state, rand_goal, rand_goal)
    print(q.shape)


    
