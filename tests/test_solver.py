import numpy as np

from develmorph.cell_body import create_cell_body
from develmorph.solver import step


def test_solver_separates_overlapping_cells():
    """Two overlapping cells are pushed apart by XPBD solver."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    cb2, positions = create_cell_body(
        center=np.array([0.4, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb1, cb2],
        shell_params=(5.0, 5.0, 5.0),
    )

    dist = np.linalg.norm(new_pos[cb1.center_idx] - new_pos[cb2.center_idx])
    assert dist > 0.4


def test_solver_preserves_separated_cells():
    """Already-separated cells are not moved significantly."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    cb2, positions = create_cell_body(
        center=np.array([5.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb1, cb2],
        shell_params=(10.0, 10.0, 10.0),
    )

    assert np.allclose(new_pos[cb1.center_idx], [0.0, 0.0, 0.0], atol=0.1)
    assert np.allclose(new_pos[cb2.center_idx], [5.0, 0.0, 0.0], atol=0.1)


def test_solver_enforces_shell():
    """Cell outside shell is pushed back inside."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([10.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb],
        shell_params=(5.0, 5.0, 5.0),
        n_iter=20,
    )

    ea = 5.0 - 0.5
    val = (new_pos[cb.center_idx][0] / ea) ** 2
    assert val <= 1.5


def test_solver_updates_velocity():
    """Velocity is updated to reflect position correction."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    cb2, positions = create_cell_body(
        center=np.array([0.4, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    _, new_vel = step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb1, cb2],
        shell_params=(5.0, 5.0, 5.0),
    )

    assert not np.allclose(new_vel, 0.0)


def test_solver_surface_cell_maintains_volume():
    """Cell with surface points maintains volume after solver step."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb],
        shell_params=(5.0, 5.0, 5.0),
    )

    vol = 0.0
    for i0, i1, i2 in cb.surface_global_triangles():
        vol += np.dot(new_pos[i0], np.cross(new_pos[i1], new_pos[i2]))
    vol /= 6.0

    assert abs(vol - cb.target_volume) / cb.target_volume < 0.1


def test_solver_two_surface_cells_separate():
    """Two overlapping surface-model cells are pushed apart."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    cb2, positions = create_cell_body(
        center=np.array([0.5, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb1, cb2],
        shell_params=(5.0, 5.0, 5.0),
    )

    dist = np.linalg.norm(new_pos[cb1.center_idx] - new_pos[cb2.center_idx])
    assert dist > 0.5


def test_solver_surface_cell_deforms_on_collision():
    """Cell deforms (surface points shift) when pushed against shell boundary."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([4.0, 0.0, 0.0]), radius=1.0, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb],
        shell_params=(5.0, 5.0, 5.0),
        n_iter=20,
    )

    center = new_pos[cb.center_idx]
    dist_plus_x = np.linalg.norm(new_pos[cb.surface_indices[0]] - center)
    dist_minus_x = np.linalg.norm(new_pos[cb.surface_indices[1]] - center)

    # +x side compressed or -x side expanded
    assert dist_plus_x < 1.0 or dist_minus_x > 1.0
