import torch
import exputils as eu

class FeatureGoalWeightUSF(torch.nn.Module):

    @staticmethod
    def default_config():
        return eu.AttrDict(
           state_size = 2,
           goal_size = 2,
           features_size = 100,
           num_actions = 4,
        )

    def __init__(self, config = None,  **kwargs) -> None:
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, self.default_config())
        
        self.num_actions = self.config.num_actions
        self.features_size = self.config.features_size

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
    def forward(self, phi_s, g, w, **kwargs):
        #phi_s is the feature state for s and it is assumed to be 100 dimensional
        #w is the reward weight vector which defines a task. It is the same size as the features
        g_rep = self.layer_goal(g)
        rep = torch.cat((phi_s,g_rep),dim=1)
        sf_s_g = self.layer_concat(rep)

        N = sf_s_g.shape[0]
        sf_s_g = sf_s_g.reshape(N, self.num_actions, self.features_size)

        Q_s_g = torch.matmul(sf_s_g, w.unsqueeze(2)).squeeze(dim=2)
        return Q_s_g


if __name__ == '__main__':
    my_dqn = FeatureGoalWeightUSF()
    print(my_dqn)
    
    rand_features = torch.rand(10,100)
    rand_goals = torch.rand(10,2)
    rand_weights = torch.rand(10,100)

    output = my_dqn(rand_features, rand_goals, rand_weights)
    print(output.shape)
    
    # Emulating behavior of epsilon greedy call for a single state, goal pair (not a batch)
    rand_feature = torch.rand(100).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)
    rand_weight = torch.rand(100).unsqueeze(0)

    output = my_dqn(rand_feature, rand_goal, rand_weight)
    print(output.shape)
