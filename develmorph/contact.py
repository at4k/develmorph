# develmorph/contact.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContactRecord:
    """
    Analytical contact between two PhysicsCells.

    Computed from center-to-center distance and cell radii.
    Normal points from cell j toward cell i (repulsion direction for i).
    """

    i: int          # index in physics_cells list
    j: int          # index in physics_cells list
    overlap: float  # delta = r_i + r_j - dist (> 0 when in contact)
    area: float     # Hertz-like contact area = pi * delta * r_harmonic
    normal: np.ndarray  # unit vector from j to i


def compute_contacts(
    physics_cells: list,
    positions: np.ndarray,
) -> list[ContactRecord]:
    """
    Compute all pairwise contacts between PhysicsCells.

    Args:
        physics_cells: list of PhysicsCell
        positions: (N, 3) center positions

    Returns:
        list of ContactRecord for all overlapping pairs (i < j)
    """
    records = []
    n = len(physics_cells)
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = physics_cells[i], physics_cells[j]
            delta_vec = positions[ci.center_idx] - positions[cj.center_idx]
            dist = np.linalg.norm(delta_vec)
            overlap = ci.radius + cj.radius - dist
            if overlap <= 0.0:
                continue
            normal = delta_vec / dist if dist > 1e-12 else np.array([1.0, 0.0, 0.0])
            r_harm = 2.0 * ci.radius * cj.radius / (ci.radius + cj.radius)
            area = np.pi * overlap * r_harm
            records.append(ContactRecord(i=i, j=j, overlap=overlap, area=area, normal=normal))
    return records
