import numpy as np
from develmorph.division import divide_cell


def test_division_increases_cell_number():

    pos = np.array([[0, 0, 0]])
    r = np.array([1.0])

    new_pos, new_r = divide_cell(pos, r, 0)

    assert len(new_pos) == 2
    assert len(new_r) == 2
