import torch
import exputils as eu

class UsfNN(torch.nn.Module):
    
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
    
    def forward(self, agent_position, goal_position):
        phi_s = self.layer_state(agent_position)
        g_rep = self.layer_goal(goal_position)
        rep = torch.cat((phi_s,g_rep),dim=1)
        sf_s_g = self.layer_concat(rep)
        
        w = self.layer_goal_weights(goal_position)

        N = sf_s_g.shape[0]
        sf_s_g = sf_s_g.reshape(N, self.num_actions, self.features_size)

        return sf_s_g, w, phi_s

if __name__ == '__main__':
    my_dqn = UsfNN(features_size = 2)
    print(my_dqn)
    
    rand_states = torch.rand(10,2)
    rand_goals = torch.rand(10,2)

    output = my_dqn(rand_states, rand_goals)
    q = torch.sum(torch.mul(output[0], output[1].unsqueeze(1)), dim=2)
    print(q.shape)
    
    # Emulating behavior of epsilon greedy call for a single state, goal pair (not a batch)
    rand_state = torch.rand(2).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)

    output = my_dqn(rand_state, rand_goal)
    print(output[0].shape, output[1].shape, output[2].shape)
    psi = output[0]
    print(psi)
    w = output[1]
    print(w)
    print(psi[0][0][0]*w[0][0]+psi[0][0][1]*w[0][1])
    print(psi[0][1][0]*w[0][0]+psi[0][1][1]*w[0][1])
    print(psi[0][2][0]*w[0][0]+psi[0][2][1]*w[0][1])
    print(psi[0][3][0]*w[0][0]+psi[0][3][1]*w[0][1])
    q = torch.sum(torch.mul(psi, w.unsqueeze(1)), dim=2)
    print(q)
    qm,am = torch.max(q,dim=1)
    print(qm, qm.shape)
    print(am, am.shape)
    
    
