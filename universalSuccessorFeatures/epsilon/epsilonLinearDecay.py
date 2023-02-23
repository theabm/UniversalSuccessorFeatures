import exputils as eu

class EpsilonLinearDecay():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            eps_max = 1.0,
            eps_min = 0.1,
            scheduled_episodes = 100,
        )

    def __init__(self, config = None, **kwargs):

        self.config = eu.combine_dicts(kwargs, config, EpsilonLinearDecay.default_config())
        self.eps_max = self.config.eps_min
        self.eps_min = self.config.eps_max
        self.scheduled_episodes = self.config.scheduled_episodes

        self.current_epsilon = self.eps_max

    def decay(self, episode):
        m = (self.eps_min-self.eps_max)/self.scheduled_episodes * (episode<=self.scheduled_episodes)
        self.current_epsilon += m
