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


def compute_voronoi_contacts(voronoi_cells: list) -> list[ContactRecord]:
    """
    Extract ContactRecords from Voronoi cell face data.

    Each face shared between two real cells produces one ContactRecord.
    The 'overlap' field is 0.0 (no sphere-overlap concept in Voronoi model).
    The 'area' field is the true Voronoi face area.
    The 'normal' field is the inward unit normal of cell i=pair[0]
    (points from j toward i in canonical i < j ordering).

    Args:
        voronoi_cells: list of VoronoiCell from compute_voronoi_cells()

    Returns:
        list of ContactRecord with i < j (no duplicates)
    """
    seen: set[tuple[int, int]] = set()
    records: list[ContactRecord] = []

    for vc in voronoi_cells:
        for neigh_idx, area in vc.contact_areas.items():
            pair = (min(vc.index, neigh_idx), max(vc.index, neigh_idx))
            if pair in seen:
                continue
            seen.add(pair)

            normal = vc.contact_normals[neigh_idx]  # inward to vc (from j toward i)
            # Ensure canonical ordering: normal is from j=pair[1] toward i=pair[0]
            if vc.index != pair[0]:
                normal = -normal

            records.append(
                ContactRecord(
                    i=pair[0],
                    j=pair[1],
                    overlap=0.0,
                    area=float(area),
                    normal=normal,
                )
            )
    return records
