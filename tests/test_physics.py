import numpy as np
from develmorph.physics import compute_repulsion


def test_repulsion_symmetry():

    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

    radii = np.array([0.5, 0.5])

    forces = compute_repulsion(pos, radii)

    # 作用反作用
    assert np.allclose(forces[0], -forces[1])


def test_no_overlap_no_force():

    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    radii = np.array([0.5, 0.5])

    forces = compute_repulsion(pos, radii)

    assert np.allclose(forces, 0.0)
