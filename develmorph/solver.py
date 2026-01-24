import numpy as np

from develmorph.physics import compute_repulsion
from develmorph.project import project_overlap, project_shell


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
    x = positions + dt * velocities

    # Projection
    x = project_overlap(x, radii)
    x = project_shell(x)

    return x


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
