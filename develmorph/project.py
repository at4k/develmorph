import numpy as np


def project_overlap(positions, radii, n_iter=5):
    """
    Iteratively remove overlaps between cells.
    """

    x = positions.copy()
    n = len(x)

    for _ in range(n_iter):

        for i in range(n):
            for j in range(i + 1, n):

                dx = x[i] - x[j]
                dist = np.linalg.norm(dx)

                if dist < 1e-12:
                    continue

                R = radii[i] + radii[j]

                if dist < R:

                    nvec = dx / dist
                    delta = 0.5 * (R - dist)

                    x[i] += delta * nvec
                    x[j] -= delta * nvec

    return x


def project_shell(positions, a=5.0, b=5.0, c=5.0, radii=None):
    """
    Project cells inside an ellipsoidal shell.

    If radii is provided, ensures cell surface (not just center)
    stays inside the shell.
    """

    x = positions.copy()

    for i in range(len(x)):

        v = x[i]

        # Shrink effective shell by cell radius
        if radii is not None:
            r = radii[i]
            ea = max(a - r, 0.1)
            eb = max(b - r, 0.1)
            ec = max(c - r, 0.1)
        else:
            ea, eb, ec = a, b, c

        s = (v[0] ** 2) / (ea * ea) + (v[1] ** 2) / (eb * eb) + (v[2] ** 2) / (ec * ec)

        if s > 1.0:

            scale = 1.0 / np.sqrt(s)
            x[i] *= scale

    return x
