import numpy as np

from develmorph.solver import step


def test_solver_separates_overlapping_cells():
    """Two overlapping cells are pushed apart by XPBD solver."""
    pos = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    inv_masses = np.array([1.0, 1.0])
    radii = np.array([0.5, 0.5])

    new_pos, new_vel = step(
        positions=pos,
        velocities=vel,
        inv_masses=inv_masses,
        radii=radii,
        dt=0.01,
        shell_params=(5.0, 5.0, 5.0),
    )

    dist = np.linalg.norm(new_pos[0] - new_pos[1])
    assert dist >= (radii[0] + radii[1]) * 0.95


def test_solver_preserves_separated_cells():
    """Already-separated cells are not moved significantly."""
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    inv_masses = np.array([1.0, 1.0])
    radii = np.array([0.5, 0.5])

    new_pos, _ = step(
        positions=pos,
        velocities=vel,
        inv_masses=inv_masses,
        radii=radii,
        dt=0.01,
        shell_params=(10.0, 10.0, 10.0),
    )

    assert np.allclose(new_pos, pos, atol=1e-6)


def test_solver_enforces_shell():
    """Cell outside shell is pushed back inside."""
    pos = np.array([[10.0, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    inv_masses = np.array([1.0])
    radii = np.array([0.5])

    new_pos, _ = step(
        positions=pos,
        velocities=vel,
        inv_masses=inv_masses,
        radii=radii,
        dt=0.01,
        shell_params=(5.0, 5.0, 5.0),
    )

    ea = 5.0 - 0.5
    val = (new_pos[0, 0] / ea) ** 2
    assert val <= 1.05


def test_solver_updates_velocity():
    """Velocity is updated to reflect position correction."""
    pos = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    inv_masses = np.array([1.0, 1.0])
    radii = np.array([0.5, 0.5])

    _, new_vel = step(
        positions=pos,
        velocities=vel,
        inv_masses=inv_masses,
        radii=radii,
        dt=0.01,
        shell_params=(5.0, 5.0, 5.0),
    )

    # Cells were pushed apart, so they should have nonzero velocity
    assert not np.allclose(new_vel, 0.0)
    # Velocity should be opposite for the two cells
    assert np.dot(new_vel[0], new_vel[1]) < 0
