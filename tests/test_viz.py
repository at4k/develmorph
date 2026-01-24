import numpy as np
from develmorph.viz import make_figure


def test_make_figure():

    pos = np.zeros((2, 3))
    r = np.ones(2)

    fig, ax = make_figure(pos, r)

    assert fig is not None
    assert ax is not None
