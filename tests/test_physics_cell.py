# tests/test_physics_cell.py
import numpy as np
import pytest
from develmorph.physics_cell import PhysicsCell, create_physics_cell


def test_create_physics_cell_empty_positions():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(
        center=np.array([1.0, 2.0, 3.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(positions) == 1
    np.testing.assert_array_almost_equal(positions[0], [1.0, 2.0, 3.0])
    assert cell.center_idx == 0
    assert cell.radius == 1.0


def test_create_physics_cell_appends_to_existing():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([5.0, 0.0, 0.0]), 1.0, positions)
    assert len(positions) == 2
    assert c1.center_idx == 0
    assert c2.center_idx == 1


def test_physics_cell_all_indices():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    assert cell.all_indices() == [0]


def test_physics_cell_default_orientation():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    np.testing.assert_array_almost_equal(cell.orientation, np.eye(3))


def test_physics_cell_custom_name_and_orientation():
    positions = np.zeros((0, 3))
    orient = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
    cell, positions = create_physics_cell(
        np.array([0.0, 0.0, 0.0]), 1.0, positions,
        name="P0", orientation=orient,
    )
    assert cell.name == "P0"
    np.testing.assert_array_almost_equal(cell.orientation, orient)
