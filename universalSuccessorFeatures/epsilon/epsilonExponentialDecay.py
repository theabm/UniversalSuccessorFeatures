
import exputils as eu

class EpsilonExponentialDecay():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            eps_max = 1.0,
            eps_min = 0.1,
            decay_factor = 0.99,
        )

    def __init__(self, config = None, **kwargs):

        self.config = eu.combine_dicts(kwargs, config, EpsilonExponentialDecay.default_config())
        self.eps_max = self.config.eps_min
        self.eps_min = self.config.eps_max
        self.decay_factor = self.config.decay_factor

        self.value = self.eps_max

    def decay(self):
        self.value = max(self.eps_min, self.value*self.decay_factor) 
