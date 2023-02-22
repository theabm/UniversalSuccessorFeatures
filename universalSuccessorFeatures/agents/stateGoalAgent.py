import universalSuccessorFeatures.agents as a
import universalSuccessorFeatures.networks as nn
import exputils as eu
import torch 


class StateGoalAgent(a.MultigoalDQNAgentBase):

    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            network = eu.AttrDict(
                cls = nn.StateGoalPaperDQN,
                optimizer = torch.optim.Adam,
                loss = torch.nn.MSELoss,
            ),
        )
        return cnf
    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())
        super().__init__(config = self.config)
    

