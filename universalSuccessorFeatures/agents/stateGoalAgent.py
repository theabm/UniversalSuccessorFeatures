import universalSuccessorFeatures.agents as a
import universalSuccessorFeatures.networks as nn
import exputils as eu


class StateGoalAgent(a.MultigoalDQNAgentBase):

    @staticmethod
    def default_config():
        cnf = eu.AttrDict(
            network = eu.AttrDict(
                cls = nn.StateGoalPaperDQN

            )
        )
    def __init__(self, config = None, **kwargs):
        self.config = None

