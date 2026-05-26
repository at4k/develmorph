# develmorph/power_diagram.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_ghost_points(
    a: float,
    b: float,
    c: float,
    n: int = 50,
    margin: float = 0.05,
) -> np.ndarray:
    """
    Generate n uniformly distributed points on the ellipsoid surface (a+margin, b+margin, c+margin).

    Uses Fibonacci spiral for near-uniform coverage including poles.

    Args:
        a, b, c: ellipsoid semi-axes (egg shell parameters)
        n: number of ghost points
        margin: extra margin beyond shell boundary so ghost points are outside

    Returns:
        (n, 3) array of 3D ghost point positions
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n)
    theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    phi = 2.0 * np.pi * i / golden
    xs = (a + margin) * np.sin(theta) * np.cos(phi)
    ys = (b + margin) * np.sin(theta) * np.sin(phi)
    zs = (c + margin) * np.cos(theta)
    return np.column_stack([xs, ys, zs])


@dataclass
class VoronoiCell:
    """
    Voronoi region for one PhysicsCell.

    Computed by compute_voronoi_cells() after each XPBD step.
    Contains the full geometric data needed for XPBD volume constraints
    and contact-area-weighted adhesion.

    Attributes:
        index: position in physics_cells list (not center_idx)
        vertices: (M, 3) world-space Voronoi vertices for this cell
        volume: cell volume computed from ConvexHull(vertices)
        face_areas: area of each Voronoi face (includes shell-boundary faces)
        face_normals: inward unit normal per face (toward cell center)
        face_neighbors: PhysicsCell index of neighbor cell per face, or None for shell faces
        contact_areas: {neighbor_index: area} for real-cell-to-real-cell faces only
        contact_normals: {neighbor_index: inward_unit_normal} for real-cell faces only
    """

    index: int
    vertices: np.ndarray
    volume: float
    face_areas: list[float]
    face_normals: list[np.ndarray]
    face_neighbors: list[int | None]
    contact_areas: dict[int, float]
    contact_normals: dict[int, np.ndarray]

    def volume_gradient(self) -> np.ndarray:
        """
        Gradient of Voronoi volume w.r.t. this cell's center position.

        Formula: dV/dx_i = (1/2) * sum(A_f * n_f for all faces f)
        where n_f is the inward unit normal and A_f is face area.

        Returns:
            (3,) gradient vector
        """
        if not self.face_areas:
            return np.zeros(3)
        grad = np.zeros(3)
        for area, normal in zip(self.face_areas, self.face_normals):
            grad += area * normal
        return grad * 0.5
