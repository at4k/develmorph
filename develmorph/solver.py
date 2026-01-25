import numpy as np

from develmorph.physics import compute_repulsion
from develmorph.project import project_overlap, project_shell
from develmorph.division import divide_cell


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
    division_steps=None,
):
    """
    Run time evolution.
    """

    if division_steps is None:
        division_steps = []

    traj = [positions.copy()]

    x = positions.copy()

    for t in range(n_steps):

        if t in division_steps:

            idx = np.argmax(radii)

            x, radii = divide_cell(x, radii, idx=idx)

        x = step(x, radii, dt=dt)

        traj.append((x.copy(), radii.copy()))

    return traj
