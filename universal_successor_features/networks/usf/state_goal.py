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

        self.num_actions = self.config.num_actions
        self.features_size = self.config.features_size

        self.layer_goal_weights = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.config.goal_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=self.config.features_size),
        )

        self.layer_state = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.config.state_size, out_features=81),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=81, out_features=self.config.features_size),
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
    
    def incomplete_forward(self, agent_position, goal_position):
        s_rep = self.layer_state(agent_position)
        g_rep = self.layer_goal(goal_position)
        rep = torch.cat((s_rep,g_rep),dim=1)
        sf_s_g = self.layer_concat(rep)
        
        w = self.layer_goal_weights(goal_position)

        N = sf_s_g.shape[0]
        sf_s_g = sf_s_g.reshape(N, self.num_actions, self.features_size)

        return sf_s_g, w, s_rep
    
    def complete_forward(self, sf_s_g, w):
        return torch.sum(torch.mul(sf_s_g, w.unsqueeze(1)), dim=2)
        # return torch.matmul(sf_s_g, w.unsqueeze(2)).squeeze(dim=2)

    def forward(self, agent_position, goal_position):
        sf_s_g, w, _ = self.incomplete_forward(agent_position=agent_position, goal_position=goal_position)

        return self.complete_forward(sf_s_g=sf_s_g, w=w)


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
