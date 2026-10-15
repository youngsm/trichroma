"""chroma.event.Photons.index: a hit's position in its event's input photons (-1 when unknown)."""

import numpy as np

from chroma.event import Photons


def _photons(n, index=None):
    return Photons(np.zeros((n, 3)), np.tile([0.0, 0.0, 1.0], (n, 1)), np.tile([1.0, 0.0, 0.0], (n, 1)),
                   np.full(n, 400.0), index=index)


def test_index_defaults_to_unknown():
    assert (_photons(3).index == -1).all()


def test_index_survives_slicing_and_concatenation():
    a, b = _photons(3, index=[0, 1, 2]), _photons(2, index=[7, 8])
    assert a[1:].index.tolist() == [1, 2]
    assert (a + b).index.tolist() == [0, 1, 2, 7, 8]
    assert Photons.join([a, b]).index.tolist() == [0, 1, 2, 7, 8]
