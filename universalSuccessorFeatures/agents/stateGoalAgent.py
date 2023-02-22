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
    
    def __build_target_batch(self, experiences, goal_batch):
        #Not sure I need to cast to torch.float each time since torch.tensor automatically handles this. But for now, this is more secure
        next_state_batch = self._build_tensor_from_batch_of_np_arrays(experiences.next_state_batch).to(torch.float).to(self.device)

        #reward and terminated batch are handled differently because they are a list of floats and bools respectively and not a list of np.arrays
        reward_batch = torch.tensor(experiences.reward_batch).to(torch.float).to(self.device)
        terminated_batch = torch.tensor(experiences.terminated_batch).to(self.device)

        target_batch = self._get_dql_target_batch(next_state_batch, goal_batch, reward_batch, terminated_batch)
    
        del next_state_batch
        del reward_batch
        del terminated_batch

        return target_batch

    def __build_predicted_batch(self, experiences, goal_batch):
        state_batch = self._build_tensor_from_batch_of_np_arrays(experiences.state_batch).to(torch.float).to(self.device)
        action_batch = torch.tensor(experiences.action_batch).unsqueeze(1).to(self.device)
        predicted_batch = self.policy_net(state_batch, goal_batch).gather(1, action_batch).squeeze()

        del state_batch
        del action_batch

        return predicted_batch


