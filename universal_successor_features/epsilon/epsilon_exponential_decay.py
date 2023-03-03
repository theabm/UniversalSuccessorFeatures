import exputils as eu

class EpsilonExponentialDecay():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            max = 1.0,
            min = 0.1,
            decay_factor = 0.99,
        )

    def __init__(self, config = None, **kwargs):

        self.config = eu.combine_dicts(kwargs, config, EpsilonExponentialDecay.default_config())
        self.max = self.config.max
        self.min = self.config.min
        self.decay_factor = self.config.decay_factor

        self.value = self.max

    def decay(self):
        self.value = max(self.min, self.value*self.decay_factor) 
