import math

import numpy as np

# Octahedron: 6 vertices along +/-x, +/-y, +/-z axes
OCTAHEDRON_DIRS = np.array(
    [
        [1.0, 0.0, 0.0],  # 0: +x
        [-1.0, 0.0, 0.0],  # 1: -x
        [0.0, 1.0, 0.0],  # 2: +y
        [0.0, -1.0, 0.0],  # 3: -y
        [0.0, 0.0, 1.0],  # 4: +z
        [0.0, 0.0, -1.0],  # 5: -z
    ]
)

# 8 triangular faces of the octahedron (indices into surface points 0-5)
# Oriented so normals point outward
OCTAHEDRON_TRIANGLES = [
    (0, 2, 4),  # +x, +y, +z
    (0, 4, 3),  # +x, +z, -y
    (0, 3, 5),  # +x, -y, -z
    (0, 5, 2),  # +x, -z, +y
    (1, 4, 2),  # -x, +z, +y
    (1, 3, 4),  # -x, -y, +z
    (1, 5, 3),  # -x, -z, -y
    (1, 2, 5),  # -x, +y, -z
]

# 12 edges of the octahedron (surface-surface)
OCTAHEDRON_EDGES = [
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),  # +x to equator
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),  # -x to equator
    (2, 4),
    (4, 3),
    (3, 5),
    (5, 2),  # equator ring
]


def compute_edges(triangles):
    """
    Extract unique edges from a triangle list.

    Each edge is a sorted tuple (i, j) where i < j.
    Returns a sorted list for deterministic ordering.
    """
    edge_set = set()
    for a, b, c in triangles:
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(b, c), max(b, c)))
        edge_set.add((min(a, c), max(a, c)))
    return sorted(edge_set)


def subdivide_mesh(directions, triangles, area_threshold, max_iterations=10):
    """
    Iteratively subdivide triangles whose area exceeds the threshold.

    For each oversized triangle, insert its centroid (projected to unit sphere)
    and replace the triangle with 3 sub-triangles.

    Args:
        directions: (N, 3) unit direction vectors on the sphere
        triangles: list of (i, j, k) index tuples
        area_threshold: maximum allowed triangle area
        max_iterations: safety limit on subdivision passes

    Returns:
        (new_directions, new_triangles): refined mesh as (np.ndarray, list)
    """
    dirs = list(directions)
    tris = list(triangles)

    for _ in range(max_iterations):
        new_tris = []
        any_subdivided = False
        for a, b, c in tris:
            v0 = np.asarray(dirs[a])
            v1 = np.asarray(dirs[b])
            v2 = np.asarray(dirs[c])
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            if area > area_threshold:
                centroid = (v0 + v1 + v2) / 3.0
                centroid = centroid / np.linalg.norm(centroid)
                ci = len(dirs)
                dirs.append(centroid)
                new_tris.append((a, b, ci))
                new_tris.append((b, c, ci))
                new_tris.append((c, a, ci))
                any_subdivided = True
            else:
                new_tris.append((a, b, c))
        tris = new_tris
        if not any_subdivided:
            break

    return np.array(dirs), tris


class CellBody:
    """
    A cell represented as center point + 6 octahedron surface points.

    All indices refer to positions in the flat particle array.
    """

    def __init__(
        self,
        center_idx,
        surface_indices,
        target_volume,
        center_surface_rest,
        surface_surface_rest,
        membrane_compliance=0.01,
    ):
        self.center_idx = center_idx
        self.surface_indices = list(surface_indices)
        self.triangles = OCTAHEDRON_TRIANGLES
        self.target_volume = target_volume
        self.membrane_compliance = membrane_compliance

        # Edges for distance constraints (with rest lengths)
        self.center_surface_edges = [(center_idx, si) for si in surface_indices]
        self.center_surface_rest = list(center_surface_rest)
        self.surface_surface_edges = [
            (surface_indices[a], surface_indices[b]) for a, b in OCTAHEDRON_EDGES
        ]
        self.surface_surface_rest = list(surface_surface_rest)

    def all_indices(self):
        """Return all particle indices belonging to this cell."""
        return [self.center_idx] + self.surface_indices

    def surface_global_triangles(self):
        """Return triangles with global particle indices."""
        return [
            (self.surface_indices[a], self.surface_indices[b], self.surface_indices[c])
            for a, b, c in self.triangles
        ]


def create_cell_body(center, radius, positions, membrane_compliance=0.01):
    """
    Create a CellBody and append its particles to the positions array.

    Args:
        center: (3,) center position
        radius: cell radius
        positions: existing (N, 3) positions array to append to
        membrane_compliance: membrane softness

    Returns:
        (cell_body, new_positions): CellBody and updated positions array
    """
    center = np.asarray(center, dtype=float)
    n = len(positions)

    # Center point
    center_idx = n
    new_points = [center]

    # Surface points
    surface_indices = []
    for i, d in enumerate(OCTAHEDRON_DIRS):
        surface_indices.append(n + 1 + i)
        new_points.append(center + radius * d)

    if len(positions) == 0:
        new_positions = np.array(new_points)
    else:
        new_positions = np.vstack([positions, new_points])

    # Compute target volume from actual mesh geometry
    target_volume = 0.0
    for a, b, c in OCTAHEDRON_TRIANGLES:
        p0 = new_positions[surface_indices[a]]
        p1 = new_positions[surface_indices[b]]
        p2 = new_positions[surface_indices[c]]
        target_volume += np.dot(p0, np.cross(p1, p2))
    target_volume /= 6.0

    # Compute rest lengths from initial geometry
    center_surface_rest = [
        np.linalg.norm(new_positions[si] - new_positions[center_idx])
        for si in surface_indices
    ]
    surface_surface_rest = [
        np.linalg.norm(
            new_positions[surface_indices[a]] - new_positions[surface_indices[b]]
        )
        for a, b in OCTAHEDRON_EDGES
    ]

    cell_body = CellBody(
        center_idx=center_idx,
        surface_indices=surface_indices,
        target_volume=target_volume,
        center_surface_rest=center_surface_rest,
        surface_surface_rest=surface_surface_rest,
        membrane_compliance=membrane_compliance,
    )

    return cell_body, new_positions


def build_particle_arrays(cell_bodies, positions):
    """
    Rebuild flat arrays from a list of CellBodies using current positions.

    Repacks particles contiguously and reassigns all indices.

    Args:
        cell_bodies: list of CellBody
        positions: current (M, 3) positions array

    Returns:
        (new_positions, new_inv_masses): repacked flat arrays
    """
    new_positions = []
    new_inv_masses = []
    idx = 0

    for cb in cell_bodies:
        # Collect current positions for this cell
        old_center = positions[cb.center_idx]
        old_surfaces = [positions[si] for si in cb.surface_indices]

        # Reassign indices
        cb.center_idx = idx
        new_positions.append(old_center)
        new_inv_masses.append(1.0)
        idx += 1

        new_surface_indices = []
        for sp in old_surfaces:
            new_surface_indices.append(idx)
            new_positions.append(sp)
            new_inv_masses.append(1.0)
            idx += 1

        cb.surface_indices = new_surface_indices

        # Rebuild edge lists with new indices
        cb.center_surface_edges = [(cb.center_idx, si) for si in cb.surface_indices]
        cb.surface_surface_edges = [
            (cb.surface_indices[a], cb.surface_indices[b]) for a, b in OCTAHEDRON_EDGES
        ]

    return np.array(new_positions), np.array(new_inv_masses)
