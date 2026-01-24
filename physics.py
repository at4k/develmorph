import numpy as np


def compute_repulsion(positions, radii, k=10.0):
    """
    Compute pairwise repulsive forces between cells.

    positions: (N,3)
    radii: (N,)
    returns: forces (N,3)
    """

    n = len(positions)
    forces = np.zeros_like(positions)

    for i in range(n):
        for j in range(i + 1, n):

            dx = positions[i] - positions[j]
            dist = np.linalg.norm(dx)

            if dist == 0:
                continue

            R = radii[i] + radii[j]

            # Overlap
            if dist < R:

                nvec = dx / dist
                fmag = k * (R - dist)

                f = fmag * nvec

                forces[i] += f
                forces[j] -= f

    return forces
