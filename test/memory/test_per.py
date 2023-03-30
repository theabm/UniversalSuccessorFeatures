import universal_successor_features.memory as mem
import numpy as np
import torch


def test_build_push():
    exp_repl = mem.PrioritizedExperienceReplayMemory()
    exp_repl.push((1,2,3))
    assert len(exp_repl) == 1 and exp_repl[0][0] == 1 and exp_repl[0][1] == 2 and exp_repl[0][2] == 3

def test_sample(batch_size = 5):
    exp_repl = mem.PrioritizedExperienceReplayMemory()
    for _ in range(batch_size*2):
        exp_repl.push(np.random.rand(4))
    
    samples, weights = exp_repl.sample(batch_size=batch_size)

    assert len(samples) == batch_size and (weights <= 1.0).all

def test_sampling_is_correct_proportions():
    exp_repl = mem.PrioritizedExperienceReplayMemory()

    exp_repl.push("a")
    exp_repl.push("b")

    exp_repl.tree.add(0,1)
    exp_repl.tree.add(1,4)

    counter_1 = 0
    counter_4 = 0
    for _ in range(10000):
        val = np.random.uniform(0,exp_repl.tree.ptotal())
        p, i = exp_repl.tree.get(val)
        if p == 1.0:
            counter_1 +=1
        elif p == 4.0:
            counter_4 += 1
    assert np.isclose(counter_4/counter_1, 4, rtol=0, atol=0.5)

def test_updating_samples_work_as_expected():
    exp_repl = mem.PrioritizedExperienceReplayMemory(eps = 0)

    exp_repl.push("a")
    exp_repl.push("b")
    exp_repl.push("c")
    exp_repl.push("d")
    
    samples, weight = exp_repl.sample(2)

    new_td_error = [1,4]

    exp_repl.update_samples(torch.tensor(new_td_error))

    counter_1 = 0
    counter_4 = 0
    for _ in range(10000):
        val = np.random.uniform(0,exp_repl.tree.ptotal())
        p, i = exp_repl.tree.get(val)
        if p == 1.0:
            counter_1 +=1
        elif p == 4.0:
            counter_4 += 1
    exp_repl.push("e")
    exp_repl.push("f")
    _, weight = exp_repl.sample(2)
    assert np.isclose(counter_4/counter_1, 4, rtol=0, atol=0.5) and (weight <= 1).all()

#test_updating_samples_work_as_expected()