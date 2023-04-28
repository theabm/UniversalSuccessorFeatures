import torch
import exputils as eu

class StateGoalPaperDQN(torch.nn.Module):
    
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

        self.state_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.config.state_size, out_features=81),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=81, out_features=self.config.features_size),
        )   
        self.goal_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.config.goal_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=self.config.features_size),
        )
        self.layer_concat = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*self.config.features_size, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=self.config.num_actions),
        )
    def forward(self, agent_position, env_goal_position, **kwargs):
        agent_position_representation = self.state_layer(agent_position)
        env_goal_position_representation = self.goal_layer(env_goal_position)
        joined_representation = torch.cat((agent_position_representation,env_goal_position_representation),dim=1)
        q = self.layer_concat(joined_representation)
        
        return q, None, None, None


if __name__ == '__main__':
    my_dqn = StateGoalPaperDQN()
    print(my_dqn)

    rand_states = torch.rand(10,2)
    rand_goals = torch.rand(10,2)

    q, *_ = my_dqn(rand_states, rand_goals)
    print(q.shape)
    
    # Emulating behavior of epsilon greedy call for a single state, goal pair (not a batch)
    rand_state = torch.rand(2).unsqueeze(0)
    rand_goal = torch.rand(2).unsqueeze(0)

    q,*_ = my_dqn(rand_state, rand_goal)
    print(q.shape)
