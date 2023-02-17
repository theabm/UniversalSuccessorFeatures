import torch

class StateGoalUSF(torch.nn.Module):
    def __init__(self, state_size = 2, goal_size = 2, num_actions = 4, features_size = 100, **kwargs) -> None:
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

        self.layer_state = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=81),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=81, out_features=features_size),
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

    def forward(self,s,g):
        s_rep = self.layer_state(s)
        g_rep = self.layer_goal(g)
        rep = torch.cat((s_rep,g_rep),dim=1)
        sf_s_g = self.layer_concat(rep)
        
        w = self.layer_goal_weights(g)
        
        N = sf_s_g.shape[0]
        sf_s_g = sf_s_g.reshape(N, self.num_actions, self.features_size)

        Q_s_g = torch.matmul(sf_s_g, w.unsqueeze(2)).squeeze(dim=2)
        return Q_s_g


if __name__ == '__main__':
    my_dqn = StateGoalUSF()
    print(my_dqn)
    
    rand_states = torch.rand(10,2)
    rand_goals = torch.rand(10,2)

    output = my_dqn(rand_states, rand_goals)
    print(output.shape)
    
    # Emulating behavior of epsilon greedy call for a single state, goal pair (not a batch)
    rand_state = torch.rand(2).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)

    output = my_dqn(rand_state, rand_goal)
    print(output.shape)
