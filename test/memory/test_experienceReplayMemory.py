import universalSuccessorFeatures.memory as mem


def test_build_push():
    transition = {
        "state":[1,2],
        "goal": [1,4],
        "reward": [-5,8]
    }
    exp_repl = mem.ExperienceReplayMemory()
    exp_repl.push(**transition)
    assert len(exp_repl) == 1