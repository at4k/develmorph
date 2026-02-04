import numpy as np


def divide_cell(positions, radii, idx, axis=None, volume_ratio=0.5):
    """
    Divide one cell into two with optional asymmetric volume distribution.

    Args:
        positions: Array of cell positions
        radii: Array of cell radii
        idx: Index of cell to divide
        axis: Division axis (unit vector). If None, random axis is used.
        volume_ratio: Fraction of parent volume going to first daughter (0.0-1.0).
                      Default 0.5 for symmetric division.
                      E.g., 0.6 means first daughter gets 60% of volume.

    Returns:
        new_positions: Updated positions array with daughter cells
        new_radii: Updated radii array with daughter cells
    """
    x = positions[idx]
    r = radii[idx]

    if axis is None:
        v = np.random.randn(3)
        v /= np.linalg.norm(v)
    else:
        v = axis / np.linalg.norm(axis)

    # Calculate daughter radii from volume ratio
    # V_parent = (4/3) * pi * r^3
    # V1 = volume_ratio * V_parent
    # V2 = (1 - volume_ratio) * V_parent
    # r1 = r * volume_ratio^(1/3)
    # r2 = r * (1 - volume_ratio)^(1/3)
    r1 = r * (volume_ratio ** (1 / 3))
    r2 = r * ((1 - volume_ratio) ** (1 / 3))

    # Position daughters along axis, offset by their radii
    # First daughter in positive direction, second in negative
    eps1 = 0.6 * r1
    eps2 = 0.6 * r2

    x1 = x + eps1 * v
    x2 = x - eps2 * v

    new_positions = []
    new_radii = []

    for i in range(len(positions)):
        if i == idx:
            new_positions.append(x1)
            new_positions.append(x2)

            new_radii.append(r1)
            new_radii.append(r2)
        else:
            new_positions.append(positions[i])
            new_radii.append(radii[i])

    return np.array(new_positions), np.array(new_radii)
