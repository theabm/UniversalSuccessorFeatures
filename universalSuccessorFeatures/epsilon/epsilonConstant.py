import exputils as eu

class EpsilonConstant():

    @staticmethod
    def default_config():
        return eu.AttrDict(
            value = 0.25
        )
    def __init__(self, config = None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, EpsilonConstant.default_config())
        self.value = self.config.value

    def decay(self):
        return
    