import torch

class FeatureGoalUSF(torch.nn.Module):
    def __init__(self, goal_size = 2, num_actions = 4, features_size = 100, **kwargs) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.features_size = features_size

        self.layer_goal_weights = torch.nn.Sequential(
            torch.nn.Linear(in_features=goal_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=features_size),
        )

        self.layer_goal = torch.nn.Sequential(
            torch.nn.Linear(in_features=goal_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=features_size),
        )
        self.layer_concat = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*features_size, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=num_actions*features_size),
        )
    def forward(self,phi_s,g):
        #phi_s is the feature state for s and it is assumed to be 100 dimensional

        g_rep = self.layer_goal(g)
        rep = torch.cat((phi_s,g_rep),dim=1)
        sf_s_g = self.layer_concat(rep)

        w = self.layer_goal_weights(g)

        N = sf_s_g.shape[0]
        sf_s_g = sf_s_g.reshape(N, self.num_actions, self.features_size)

        Q_s_g = torch.matmul(sf_s_g, w.unsqueeze(2)).squeeze(dim=2)
        return Q_s_g


if __name__ == '__main__':
    my_dqn = FeatureGoalUSF()
    print(my_dqn)
    
    rand_features = torch.rand(10,100)
    rand_goals = torch.rand(10,2)

    output = my_dqn(rand_features, rand_goals)
    print(output.shape)
    
    # Emulating behavior of epsilon greedy call for a single state, goal pair (not a batch)
    rand_feature = torch.rand(100).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)

    output = my_dqn(rand_feature, rand_goal)
    print(output.shape)
