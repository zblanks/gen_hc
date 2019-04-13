import numpy as np
from ..utils import create_rng


def test_create_rng():
    # If None is passed make sure that it gives a RNG and that it yields the
    # same array
    rng0 = np.random.RandomState(17)
    bytes0 = rng0.bytes(1)
    rng1 = create_rng(None)
    bytes1 = rng1.bytes(1)
    assert bytes0 == bytes1

    # Check if an integer is passed that it yields the same value
    rng1 = create_rng(17)
    bytes1 = rng1.bytes(1)
    assert bytes0 == bytes1

    # Check if RNG is passed that it yields the correct value
    rng1 = create_rng(np.random.RandomState(17))
    bytes1 = rng1.bytes(1)
    assert bytes0 == bytes1
