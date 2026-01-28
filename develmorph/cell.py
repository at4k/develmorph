import numpy as np
from typing import Optional


class Cell:
    """
    Cell with intrinsic orientation (local frame).

    Attributes:
        cell_id: Unique numeric identifier
        name: Biological name (e.g., "P0", "AB", "P1")
        position: 3D position vector
        orientation: 3x3 orthonormal matrix [e1, e2, e3]
                     e1 = major axis, e2 = minor axis, e3 = e1 x e2
        next_division_step: Step at which this cell will divide (optional)
    """

    def __init__(
        self,
        cell_id: int,
        name: str,
        position,
        orientation=None,
        next_division_step: Optional[int] = None,
    ):
        self.cell_id = cell_id
        self.name = name
        self.position = np.asarray(position, dtype=float)

        if orientation is None:
            self.orientation = np.eye(3)
        else:
            self.orientation = np.asarray(orientation, dtype=float)

        self.next_division_step = next_division_step

    def __repr__(self):
        return f"Cell({self.name}, id={self.cell_id}, pos={self.position})"
