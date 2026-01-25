import numpy as np
from develmorph.project import project_overlap


def test_projection_removes_overlap():

    pos = np.array([[0, 0, 0], [0.1, 0, 0]])

    r = np.array([0.5, 0.5])

    new = project_overlap(pos, r)

    d = np.linalg.norm(new[0] - new[1])

    assert d >= r.sum() * 0.99
