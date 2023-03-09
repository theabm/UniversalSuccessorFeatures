import numpy as np
import exputils as eu
import exputils.data.logging as log

def run_rl_training(config = None, **kwargs):

    # default config
    default_config = eu.AttrDict(
        agent = eu.AttrDict(cls = None),
        env = eu.AttrDict(cls = None),
        n_episodes = 100, 
        n_max_steps_per_episode = np.inf,
        log_to_tensor_board = True,
    )
    
    config = eu.combine_dicts(kwargs, config, default_config)

    eu.misc.seed(config)

    env = eu.misc.create_object_from_config(config.env)
    
    agent = eu.misc.create_object_from_config(config.agent, env)