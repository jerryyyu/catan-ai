from copy import deepcopy

from catan_ai.belief import Particle
from catan_ai.engine import Resource


def test_particle_copy_is_equal_and_all_mutable_containers_are_detached():
    original = Particle.empty()
    original.hands[0][Resource.ORE] = 2
    original.development[1].append(('knight', 3))
    copied = deepcopy(original)
    assert copied == original
    copied.hands[0][Resource.ORE] = 0
    copied.development[1].clear()
    copied.deck.pop()
    assert original.hands[0][Resource.ORE] == 2
    assert original.development[1] == [('knight', 3)]
    assert len(original.deck) == 25
    repeated = deepcopy([original, original])
    assert repeated[0] is repeated[1]
    assert repeated[0] is not original
