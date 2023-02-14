import torch

class MultigoalDQNNetwork(torch.nn.Module):

    def __init__(self, state_size = 2, goal_size = 2, num_actions = 4) -> None:
        super().__init__()
        self.layer_state = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=81),
            torch.nn.ReLU(),
        )   
        self.layer_goal = torch.nn.Sequential(
            torch.nn.Linear(in_features=goal_size, out_features=64),
            torch.nn.ReLU(),
        )
        self.layer_concat = torch.nn.Sequential(
            torch.nn.Linear(in_features=145, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=num_actions),
        )
    def forward(self,s,g):
        s_rep = self.layer_state(s)
        g_rep = self.layer_goal(g)
        rep = torch.cat((s_rep,g_rep),dim=1)
        q = self.layer_concat(rep)
        
        return q


if __name__ == '__main__':
    my_dqn = MultigoalDQNNetwork()
    print(my_dqn)