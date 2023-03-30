import universal_successor_features.memory as mem
import numpy as np


def test_build_push():
    exp_repl = mem.ExperienceReplayMemory()
    exp_repl.push((1,2,3))
    assert len(exp_repl) == 1 and exp_repl[0][0] == 1 and exp_repl[0][1] == 2 and exp_repl[0][2] == 3

def test_sample(batch_size = 5):
    exp_repl = mem.ExperienceReplayMemory()
    for _ in range(batch_size*2):
        exp_repl.push(np.random.rand(4))
    
    samples, weights = exp_repl.sample(batch_size=batch_size)
    assert len(samples) == batch_size