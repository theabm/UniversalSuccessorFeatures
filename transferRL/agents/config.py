import torch

#### DEVICE #################################################################

DEVICE = "gpu"

#############################################################################

#### ENVIRONMENT STRUCTURE ##################################################

OBS_SPACE_SIZE = 2
NUM_ACTIONS = 5
DISCOUNT_RATE = 0.9

#############################################################################

#### NETWORK STRUCTURE ######################################################

HIDDEN_LAYER_STRUCTURE = [64, 128, 64]
NETWORK_STRUCTURE = [OBS_SPACE_SIZE] + HIDDEN_LAYER_STRUCTURE + [NUM_ACTIONS]

#############################################################################

#### NETWORK HYPERPARAMETERS ################################################

LEARNING_RATE = 1e-3
OPTIMIZER = torch.optim.Adam
BATCH_SIZE = 64

#############################################################################

#### EPSILON GREEDY PARAMETERS ##############################################

EPSILON_MAX = 0.1
EPSILON_MIN = 0.01
SCHEDULE = "linear"
SCHEDULE_TIMESTEPS = 100

#############################################################################

#### EXPERIENCE REPLAY PARAMETERS ###########################################

MEMORY_BUFFER_SIZE = 1e6
TYPE = "experience replay"

#############################################################################

#### UPDATE METHOD ##########################################################

UPDATE_METHOD = "soft"
SOFT_UPDATE_PARAMETER = 1e-2            # measures the percentage of the policy neural network that is copied into the target neural network at each time step
HARD_UPDATE_FREQUENCY = 1e3             # measures number of timesteps before a hard update is performed

#############################################################################

#### DIRECTORY FOR SAVE/LOAD ################################################

SAVE_DIRECTORY = None
