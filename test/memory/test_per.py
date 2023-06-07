import universal_successor_features.memory as mem
import numpy as np
import torch
import exputils as eu


def test_build_push():
    """
    Test that:
        1. Pushing a single transition results in the correct size
        2. It is stored in the memory.
        3. The value in the sum tree is as expected (max priority)
    """
    exp_repl = mem.PrioritizedExperienceReplayMemory()
    exp_repl.push((1, 2, 3))
    assert (
        len(exp_repl) == 1
        and exp_repl[0][0] == 1
        and exp_repl[0][1] == 2
        and exp_repl[0][2] == 3
    )
    assert exp_repl.tree[0] == exp_repl.max_priority


def test_sample(batch_size=5):
    """
    First I push twice the batch size elements in the tree.
    Then I sample batch_size elements.

    Test that:
        1. The number of samples is correct
        2. The weights are all <= 1.0
        3. The weights all have weights = to 1.0
        only in this case where we do not update the sample weights
    """
    num_transitions = batch_size*2
    exp_repl = mem.PrioritizedExperienceReplayMemory()
    for _ in range(num_transitions):
        exp_repl.push(np.random.rand(4))

    samples, weights = exp_repl.sample(batch_size=batch_size)

    assert len(samples) == batch_size and (weights <= 1.0).all()
    assert (weights == 1.0).all()


def test_sampling_is_correct_proportions_using_tree():
    """
    Insert two elements in the memory, manually modify their weights,
    sample them many times.
    """
    exp_repl = mem.PrioritizedExperienceReplayMemory()

    # I push two elements
    exp_repl.push("a")
    exp_repl.push("b")

    # I modify their weights manually
    # My implementation does not support adding elements
    # with associated weight. 
    exp_repl.tree.add(0, 1)
    exp_repl.tree.add(1, 4)

    # I expect that element "b" will be sampled four times as much as 
    # element "a"
    # By construction, any number from 0-1 will correspond to element a,
    # and every number from 1-4 will correspond to element b
    counter_1 = 0
    counter_4 = 0
    for _ in range(10000):
        # Sample an value from 0 - 4
        val = np.random.uniform(0, exp_repl.tree.ptotal())

        # Get that value
        p, i = exp_repl.tree.get(val)

        if p == 1.0:
            counter_1 += 1
        elif p == 4.0:
            counter_4 += 1

    assert np.isclose(counter_4 / counter_1, 4, rtol=0, atol=0.5)

def test_sampling_is_correct_proportions_using_memory():
    """
    Insert two elements in the memory, manually modify their weights,
    sample them many times.
    """
    eu.misc.seed(0)

    exp_repl = mem.PrioritizedExperienceReplayMemory()

    # I push two elements
    exp_repl.push("a")
    exp_repl.push("b")

    # I modify their weights manually
    # My implementation does not support adding elements
    # with associated weight. 
    exp_repl.tree.add(0, 1)
    exp_repl.tree.add(1, 4)

    # I expect that element "b" will be sampled four times as much as 
    # element "a"
    # By construction, any number from 0-1 will correspond to element a,
    # and every number from 1-4 will correspond to element b
    counter_a = 0
    counter_b = 0
    for _ in range(10000):
        # sample a single element
        item, _ = exp_repl.sample(1)

        if item[0] == "a":
            counter_a += 1
        elif item[0] == "b":
            counter_b += 1

    assert np.isclose(counter_b / counter_a, 4, rtol=0, atol=0.5)

def test_sampling_is_correct_proportions_using_memory_a0b0():
    """
    Insert two elements in the memory, manually modify their weights,
    sample them many times.
    Alpha = 0, Beta = 0.
    This means that even though I manually insert td error, I should 
    still sample uniformly
    """
    eu.misc.seed(0)

    exp_repl = mem.PrioritizedExperienceReplayMemory(
            alpha = 0,
            beta0 = 0,
            )

    # I push two elements
    exp_repl.push("a")
    exp_repl.push("b")
    exp_repl.push("c")
    exp_repl.push("d")

    samples, weight = exp_repl.sample(2)
    sample1 = samples[0]
    sample2 = samples[1]

    new_td_error = [1, 4]

    # This should make no difference since alpha and beta are 0
    exp_repl.update_samples(torch.tensor(new_td_error))

    # counter for sample 1 and sample 2
    counter_1 = 0
    counter_2 = 0
    for _ in range(10000):
        item, _ = exp_repl.sample(1)
        if item[0] == sample1:
            counter_1 += 1
        elif item[0] == sample2:
            counter_2 += 1
    exp_repl.push("e")
    exp_repl.push("f")
    _, weight = exp_repl.sample(2)
    assert (
        np.isclose(counter_2 / counter_1, 1, rtol=0, atol=0.5)
    )
    assert (weight <= 1).all()

def test_updating_samples_work_as_expected():
    """
    I push a few transitions, sample a batch_size = 2, update the weights 
    with a td-error manually, and then verify that the sampling is correct.
    """
    eu.misc.seed(0)

    exp_repl = mem.PrioritizedExperienceReplayMemory(eps=0)

    exp_repl.push("a")
    exp_repl.push("b")
    exp_repl.push("c")
    exp_repl.push("d")

    samples, weight = exp_repl.sample(2)
    sample1 = samples[0]
    sample2 = samples[1]

    new_td_error = [1, 4]

    exp_repl.update_samples(torch.tensor(new_td_error))

    # counter for sample 1 and sample 2
    counter_1 = 0
    counter_2 = 0
    for _ in range(10000):
        item, _ = exp_repl.sample(1)
        if item[0] == sample1:
            counter_1 += 1
        elif item[0] == sample2:
            counter_2 += 1
    exp_repl.push("e")
    exp_repl.push("f")
    _, weight = exp_repl.sample(2)
    assert (
        np.isclose(counter_2 / counter_1, 4, rtol=0, atol=0.5)
    )
    assert (weight <= 1).all()


