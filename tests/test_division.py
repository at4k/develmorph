import numpy as np

from develmorph.cell_body import create_cell_body
from develmorph.division import divide_cell_body


def test_division_produces_two_daughters():
    """Division creates exactly 2 daughter CellBodies."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions
    )

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )

    assert len(daughters) == 2


def test_division_conserves_volume():
    """Daughter volumes sum to parent volume."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions
    )
    parent_vol = parent.target_volume

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )

    daughter_vol = sum(d.target_volume for d in daughters)
    assert abs(daughter_vol - parent_vol) < 1e-10


def test_division_asymmetric_volume():
    """Asymmetric division distributes volume correctly."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions
    )

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]), volume_ratio=0.6
    )

    total = daughters[0].target_volume + daughters[1].target_volume
    ratio = daughters[0].target_volume / total
    assert abs(ratio - 0.6) < 1e-10


def test_division_daughters_along_axis():
    """Daughter centers are offset along the division axis."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions
    )

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )

    c1 = positions[daughters[0].center_idx]
    c2 = positions[daughters[1].center_idx]

    assert c1[0] > 0
    assert c2[0] < 0


def test_division_particle_count():
    """Division replaces 7 particles with 14."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions
    )
    assert len(positions) == 7

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )

    assert len(positions) == 14
