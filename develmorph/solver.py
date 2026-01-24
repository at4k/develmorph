import numpy as np

from develmorph.physics import compute_repulsion


def step(
    positions,
    radii,
    dt=0.01,
    gamma=1.0,
    k_repulsion=10.0,
):
    """
    Advance system by one time step (overdamped dynamics).
    """

    # Forces
    forces = compute_repulsion(
        positions,
        radii,
        k=k_repulsion,
    )

    # Overdamped update
    velocities = forces / gamma
    new_positions = positions + dt * velocities

    return new_positions


def simulate(
    positions,
    radii,
    n_steps=1000,
    dt=0.01,
):
    """
    Run time evolution.
    """

    traj = [positions.copy()]

    x = positions.copy()

    for _ in range(n_steps):

        x = step(x, radii, dt=dt)
        traj.append(x.copy())

    return np.array(traj)
