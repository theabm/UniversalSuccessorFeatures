import torch

class StateGoalPaperDQN(torch.nn.Module):

    def __init__(self, state_size = 2, goal_size = 2, num_actions = 4) -> None:
        super().__init__()
        self.state_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=state_size, out_features=81),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=81, out_features=100),
        )   
        self.goal_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=goal_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=100),
        )
        self.layer_concat = torch.nn.Sequential(
            torch.nn.Linear(in_features=200, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=num_actions),
        )
    def forward(self,s,g):
        s_rep = self.state_layer(s)
        g_rep = self.goal_layer(g)
        rep = torch.cat((s_rep,g_rep),dim=1)
        q = self.layer_concat(rep)
        
        return q


if __name__ == '__main__':
    my_dqn = StateGoalPaperDQN()
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