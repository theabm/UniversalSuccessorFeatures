import exputils as eu
import numpy as np

def run_training(config = None, **kwargs):

    default_config = eu.AttrDict(
        seed = None, 
        env = None,
        agent = None,
        log_functions = [],
        save_log_automatically = False,
        log_agent_after_each_episode = False,
        log_name_episode = "episode",
        log_name_step_per_episode = "step_per_episode",
        log_name_reward_per_episode = "reward_per_episode",
        log_name_total_reward = "total_reward",
        log_name_step = "step",

    )

    if config == "get_default_config":
        return default_config
    
    config = eu.combine_dicts(kwargs, default_config, copy_mode="copy")

    if config.env is None:
        raise ValueError("Environment must be defined")
    
    if config.agent is None:
        raise ValueError("Agent must be defined")
