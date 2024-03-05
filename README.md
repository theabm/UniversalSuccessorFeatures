# Repository for transfer RL for social robotics.

This repository contains the implementation of my thesis work 
titled: "A Study of Universal Successor Features with Generalized Policy 
Improvement".

The basis of the work consists in the combination of two frameworks for 
transfer deep reinforcement learning: 
- Generalized Policy Improvement - Barretto et al. 
- Universal Successor Features - Ma et al.

The goal of this thesis work was to combine these two frameworks to improve the 
rate with which an agent is able to acquire new skills by reusing previously 
acquired knowledge. 

The structure is as follows: 

## test 

Contains subdirectories that test all the corresponding components of the 
implementation. 
The unit tests are mainly focused on obtaining the correct dimensions, obtaining 
correct Q-values for certain simple cases, and making sure that the training 
procedure is convergent. 

## universal-successor-features

Contains the source code of this project. It is divided in subdirectories that 
implement a specific logic. 
- agents: implements a variety of agents. Some use the features, some use the 
position of the agent in a grid aka the state. Some make use of the weights of 
the task, others dont.
- env: implements both a simple grid world environment as well as a grid world 
with rooms as described in Ma et al. 
- epsilon: implements the different types of decay for the epsilon in epsilon 
greedy.
- exp: implements the experiment logic. 
- memory: implements the replay memory. Aka Experience Replay, Prioritized Experience Replay, 
and Combined Experience Replay
- networks: implements all the neural networks used internally by the agents. 
Some are augmented to account for the number of parameters. 
- plot utils: implements the logic to visualize the learned successor features 
using Dash. 

# Using
To use the environnment, I have included the requirements.txt file which lists 
all the necessary packages. All the experiments were run using the singulary 
container image in the directory. 

# Additional Comments 
During the writing of my thesis I ran some additional experiments and made some 
extra tweaks, so some things may be less documented and some tests may be break 
since I needed to get quick functionality. 


