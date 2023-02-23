import exputils as eu

class EpsilonLinearDecay():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            max = 1.0,
            min = 0.1,
            scheduled_episodes = 100,
        )

    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, EpsilonLinearDecay.default_config())
        self.max = self.config.max
        self.min = self.config.min
        self.scheduled_episodes = self.config.scheduled_episodes

        self.value = self.max

    def decay(self, episode):
        m = (self.min-self.max)/self.scheduled_episodes * (episode<=self.scheduled_episodes)
        self.value += m
