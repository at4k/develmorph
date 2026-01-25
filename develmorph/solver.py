import numpy as np
from develmorph.project import project_overlap, project_shell


def step(positions, velocities, radii, dt):
    """
    Advance physical state by one time step.
    """

    # Overdamped dynamics (temporary)
    velocities[:] = 0.0

    positions = positions + dt * velocities

    # Constraints
    positions = project_overlap(positions, radii)
    positions = project_shell(positions)

    return positions, velocities
