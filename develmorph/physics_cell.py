import numpy as np


class PhysicsCell:
    """
    A cell represented as a single center point and a radius scalar.

    Participates in XPBD as one particle (center_idx).
    Volume = (4/3) * pi * r^3, constant between divisions.
    Orientation tracks the cell's local frame for division axis resolution.
    """

    def __init__(self, center_idx, radius, name=None, orientation=None):
        self.center_idx = center_idx
        self.radius = float(radius)
        self.name = name
        self.orientation = orientation if orientation is not None else np.eye(3)

    def all_indices(self):
        """Return list of all particle indices belonging to this cell."""
        return [self.center_idx]


def create_physics_cell(center, radius, positions, name=None, orientation=None):
    """
    Create a PhysicsCell and append its center to the positions array.

    Args:
        center: (3,) center position
        radius: cell radius
        positions: existing (N, 3) positions array
        name: biological cell name (e.g. "P0", "AB")
        orientation: 3x3 orientation matrix; defaults to identity

    Returns:
        (cell, new_positions): PhysicsCell and updated positions array
    """
    center = np.asarray(center, dtype=float)
    idx = len(positions)
    new_row = center.reshape(1, 3)
    if len(positions) == 0:
        new_positions = new_row
    else:
        new_positions = np.vstack([positions, new_row])

    cell = PhysicsCell(
        center_idx=idx,
        radius=radius,
        name=name,
        orientation=orientation,
    )
    return cell, new_positions
