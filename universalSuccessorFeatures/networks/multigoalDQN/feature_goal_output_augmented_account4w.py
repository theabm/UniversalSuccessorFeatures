import torch
import exputils as eu


# This is augmented compared to what is described in the paper.
# In particular, it uses three additional hidden layers of size: 256, H2, H3, where H2 and H3 can be choosen in any of the combinations below.
# The rationale behind this is that, compared to USF architecture, the multigoalDQN has less parameters, thus, the comparison is not fair.
# Thus, we want to account for all the extra parameters in USF compared to multigoal DQN. 

# The calculations made are as follows: (4 is num actions)
# 1. The head of the USF architecture is made up of a linear layer 256 to 4 * 100. Additionally, we have linear layers for the reward weights of 64 - 64 - 100.
# In total, this is 112896 parameters
# 2. The multigoal DQN instead, as described in the paper, only has a final layer of 256 - 4. 
# To render the num of params equal, we add two additional hidden layers after 256 and need to solve:
# 256*H1 + H1*H2 + H2*4 = 112896.
# Below are some possible combinations that we tested

#              H2    H3
#Comb 1: 256 - 252 - 189 - 4
#Comb 2: 256 - 316 - 100 - 4
#Comb 3: 256 - 352 - 64  - 4 (default)
#
#
# Please note: Using this augmented network makes sense only if the flow of information in USF passes through the whole network (as described in the paper).
# If we were to keep separate networks for the weights and the SF, this would not make sense anymore because the weights w are part of a different network.

class FeatureGoalAugumentedDQN(torch.nn.Module):
    
    @staticmethod
    def default_config():
        return eu.AttrDict(
           state_size = 2,
           goal_size = 2,
           features_size = 100,
           num_actions = 4,
           H2_size = 352,
           H3_size = 64,
        )

    def __init__(self, config = None, **kwargs) -> None:
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, self.default_config())
        
        self.goal_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.config.goal_size, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=self.config.features_size),
        )
        self.layer_concat = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*self.config.features_size, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=self.config.H2_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.config.H2_size, out_features=self.config.H3_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.config.H3_size, out_features=self.config.num_actions),
        )
    def forward(self, phi_s, g, **kwargs):
        #phi_s is the feature state for s and it is assumed to be 100 dimensional
        g_rep = self.goal_layer(g)
        rep = torch.cat((phi_s,g_rep),dim=1)
        q = self.layer_concat(rep)
        
        return q


if __name__ == '__main__':
    my_dqn = FeatureGoalAugumentedDQN()
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