import numpy as np
import pytest
from develmorph.physics_cell import create_physics_cell
from develmorph.solver import step_pc


def test_step_pc_returns_finite_positions():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, new_vel = step_pc(
        positions, velocities, inv_masses,
        dt=0.01,
        physics_cells=[cell],
        shell_params=(5.0, 5.0, 5.0),
        n_iter=10,
        damping=1.0,
    )
    assert np.all(np.isfinite(new_pos))


def test_step_pc_pushes_center_inside_shell():
    """Center outside effective ellipsoid (shell - radius) is pushed inside."""
    positions = np.zeros((0, 3))
    # Shell (2.0, 2.0, 2.0), r=0.5 -> effective (1.5, 1.5, 1.5)
    # Place center at x=1.8, outside effective shell
    cell, positions = create_physics_cell(np.array([1.8, 0.0, 0.0]), 0.5, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    for _ in range(20):
        positions, velocities = step_pc(
            positions, velocities, inv_masses,
            dt=0.01,
            physics_cells=[cell],
            shell_params=(2.0, 2.0, 2.0),
            n_iter=10,
            damping=1.0,
        )

    x, y, z = positions[cell.center_idx]
    a_eff, b_eff, c_eff = 1.5, 1.5, 1.5
    C = (x / a_eff) ** 2 + (y / b_eff) ** 2 + (z / c_eff) ** 2
    assert C <= 1.0 + 1e-6


def test_step_pc_two_cells_separated_after_settling():
    """Two overlapping cells are pushed apart by CollisionConstraint."""
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    for _ in range(50):
        positions, velocities = step_pc(
            positions, velocities, inv_masses,
            dt=0.01,
            physics_cells=[c1, c2],
            shell_params=(10.0, 10.0, 10.0),
            n_iter=10,
            damping=1.0,
        )

    dist = np.linalg.norm(positions[c1.center_idx] - positions[c2.center_idx])
    assert dist >= c1.radius + c2.radius - 1e-6


def test_step_pc_damping_one_zeroes_velocity():
    """damping=1.0 means velocity is fully zeroed after each step."""
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    velocities = np.ones_like(positions) * 5.0
    inv_masses = np.ones(len(positions))

    _, new_vel = step_pc(
        positions, velocities, inv_masses,
        dt=0.01,
        physics_cells=[cell],
        shell_params=(10.0, 10.0, 10.0),
        n_iter=1,
        damping=1.0,
    )
    assert np.allclose(new_vel, 0.0)


def test_step_pc_asymmetric_shell_confinement():
    """
    Narrow shell axis confines cell more than wide axis.
    Shell (3.0, 3.0, 1.5), r=1.0 -> effective (2.0, 2.0, 0.5).
    Center at (0, 0, 0.8): outside c-axis limit, inside a,b limits.
    """
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.8]), 1.0, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    for _ in range(30):
        positions, velocities = step_pc(
            positions, velocities, inv_masses,
            dt=0.01,
            physics_cells=[cell],
            shell_params=(3.0, 3.0, 1.5),
            n_iter=10,
            damping=1.0,
        )

    x, y, z = positions[cell.center_idx]
    assert abs(z) <= 0.5 + 1e-6
    assert abs(x) < 0.1
    assert abs(y) < 0.1
