import numpy as np


def divide_cell(positions, radii, idx, axis=None):
    """
    Divide one cell into two.
    """

    x = positions[idx]
    r = radii[idx]

    if axis is None:
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
    else:
        v = axis / np.linalg.norm(axis)

    eps = 0.6 * r

    x1 = x + eps * v
    x2 = x - eps * v

    r_new = r * (0.5 ** (1 / 3))

    new_positions = []
    new_radii = []

    for i in range(len(positions)):

        if i == idx:
            new_positions.append(x1)
            new_positions.append(x2)

            new_radii.append(r_new)
            new_radii.append(r_new)
        else:
            new_positions.append(positions[i])
            new_radii.append(radii[i])

    return np.array(new_positions), np.array(new_radii)
