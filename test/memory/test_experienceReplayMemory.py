import universalSuccessorFeatures.memory as mem
from collections import namedtuple


def test_build_push():
    Transition = namedtuple("Transition", ("state,goal,reward"))
    
    exp_repl = mem.ExperienceReplayMemory()
    exp_repl.push(Transition(1,2,3))
    assert len(exp_repl) == 1 and exp_repl[0].state == 1 and exp_repl[0].goal == 2 and exp_repl[0].reward == 3
