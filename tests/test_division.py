import numpy as np

from develmorph.cell_body import create_cell_body, TOPOLOGY_VOLUME_FACTOR
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


def test_divide_cell_body_with_area_threshold():
    """Daughter cells are created with subdivision when area_threshold is passed."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        area_threshold=0.3,
    )
    daughters, new_positions = divide_cell_body(
        parent, positions,
        axis=np.array([1.0, 0.0, 0.0]),
        area_threshold=0.3,
    )
    assert len(daughters) == 2
    for d in daughters:
        assert len(d.surface_indices) >= 6


def test_divide_cell_body_daughter_names():
    """Daughters receive the specified names."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        name="P0",
    )
    daughters, _ = divide_cell_body(
        parent,
        positions,
        axis=np.array([1.0, 0.0, 0.0]),
        daughter_names=("AB", "P1"),
    )
    assert daughters[0].name == "AB"
    assert daughters[1].name == "P1"


def test_divide_cell_body_default_daughter_names_are_none():
    """Without daughter_names, daughters have name=None."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    daughters, _ = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )
    assert daughters[0].name is None
    assert daughters[1].name is None


def test_divide_cell_body_inherits_parent_orientation():
    """Without daughter_orientations, daughters inherit parent orientation."""
    custom = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        orientation=custom,
    )
    daughters, _ = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )
    assert np.allclose(daughters[0].orientation, custom)
    assert np.allclose(daughters[1].orientation, custom)


def test_divide_cell_body_custom_daughter_orientations():
    """Daughters receive the specified orientations."""
    orient_a = np.eye(3)
    orient_b = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    daughters, _ = divide_cell_body(
        parent,
        positions,
        axis=np.array([1.0, 0.0, 0.0]),
        daughter_orientations=(orient_a, orient_b),
    )
    assert np.allclose(daughters[0].orientation, orient_a)
    assert np.allclose(daughters[1].orientation, orient_b)


def test_divide_ico_daughters_inherit_topology():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    axis = np.array([1.0, 0.0, 0.0])
    daughters, positions = divide_cell_body(cb, positions, axis=axis)
    assert daughters[0].topology == 'ico'
    assert daughters[1].topology == 'ico'


def test_divide_ico_daughters_correct_volume():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    parent_volume = cb.target_volume
    axis = np.array([1.0, 0.0, 0.0])
    daughters, positions = divide_cell_body(cb, positions, axis=axis)
    total_daughter_volume = daughters[0].target_volume + daughters[1].target_volume
    np.testing.assert_allclose(total_daughter_volume, parent_volume, rtol=0.01)
