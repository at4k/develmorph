import math
import numpy as np
import pytest
from develmorph.physics_cell import create_physics_cell
from develmorph.division import divide_physics_cell


def test_divide_produces_two_daughters():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    daughters, positions = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )
    assert len(daughters) == 2
    assert len(positions) == 2  # parent removed, 2 daughters added


def test_divide_conserves_volume():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    daughters, _ = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )
    V_parent = (4.0 / 3.0) * math.pi * parent.radius ** 3
    V_total = sum((4.0 / 3.0) * math.pi * d.radius ** 3 for d in daughters)
    assert abs(V_total - V_parent) < 1e-10


def test_divide_daughters_along_axis():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    axis = np.array([1.0, 0.0, 0.0])
    daughters, new_pos = divide_physics_cell(parent, positions, axis=axis)
    d1_pos = new_pos[daughters[0].center_idx]
    d2_pos = new_pos[daughters[1].center_idx]
    assert d1_pos[0] > 0.0  # first daughter along +axis
    assert d2_pos[0] < 0.0  # second daughter along -axis


def test_divide_daughter_indices_valid():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    daughters, new_pos = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )
    for d in daughters:
        assert 0 <= d.center_idx < len(new_pos)


def test_divide_volume_ratio():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    daughters, _ = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]), volume_ratio=0.6
    )
    r1, r2 = daughters[0].radius, daughters[1].radius
    frac = r1 ** 3 / (r1 ** 3 + r2 ** 3)
    assert abs(frac - 0.6) < 1e-10


def test_divide_daughter_names():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions,
                                            name="P0")
    daughters, _ = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]),
        daughter_names=("AB", "P1"),
    )
    assert daughters[0].name == "AB"
    assert daughters[1].name == "P1"


def test_divide_with_two_cells_surviving_position_preserved():
    """After dividing cell at index 0, the surviving cell's position is preserved."""
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([5.0, 0.0, 0.0]), 1.0, positions)
    c2_old_pos = positions[c2.center_idx].copy()

    daughters, new_pos = divide_physics_cell(
        c1, positions, axis=np.array([0.0, 1.0, 0.0])
    )
    # After deleting index 0 (c1), c2's position (was at index 1) is now at index 0
    np.testing.assert_array_almost_equal(new_pos[0], c2_old_pos)
