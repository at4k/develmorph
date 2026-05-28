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
        face_areas: area of each inter-cell Voronoi face (excludes shell-boundary faces;
            only ridges between real cells and between real/ghost cells from scipy Voronoi)
        face_normals: inward unit normal per face (toward cell center)
        face_neighbors: PhysicsCell index of neighbor cell per face, or None for ghost-cell faces
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

        Formula: dV/dx_i = (1/2) * sum_{j=real neighbors} A_ij * n̂_ij_inward
        where the sum is over REAL-CELL faces only (face_neighbor is not None).

        Ghost-cell faces are excluded: ghost points approximate the infinite
        exterior, and their Voronoi bisectors form a closed polyhedron together
        with the real-cell faces. Summing over all faces gives zero by the
        divergence theorem. Only the inter-cell (real-neighbor) faces carry the
        true volume-gradient signal.

        Note: this is the gradient of the UNclipped Voronoi volume measured by
        real-cell face areas. For cells well inside the shell this closely matches
        dV_clipped/dx_i; for boundary cells a small discrepancy remains.

        Returns:
            (3,) gradient vector
        """
        grad = np.zeros(3)
        for area, normal, neighbor in zip(self.face_areas, self.face_normals, self.face_neighbors):
            if neighbor is not None:
                grad += area * normal
        return grad * 0.5


def compute_voronoi_cells(
    physics_cells: list,
    positions: np.ndarray,
    shell_params: tuple[float, float, float],
    n_ghost: int = 50,
    ghost_margin: float = 5.0,
    n_shell_pts: int = 500,
    seed: int = 42,
) -> list[VoronoiCell]:
    """
    Compute Voronoi tessellation for all PhysicsCells within the egg shell.

    Uses standard Voronoi (scipy) with ghost points far outside the ellipsoid to
    prevent infinite Voronoi regions. Each cell's region is then clipped to the
    ellipsoidal shell boundary using halfspace intersection, so that volumes sum
    to approximately the shell volume.

    A small z-jitter is applied to real cell centers to avoid the near-planar
    degenerate case (Qhull "flat simplex" error).

    Face data (areas, normals, contacts) is populated by _fill_face_data().

    Args:
        physics_cells: list of PhysicsCell
        positions: (N, 3) center positions
        shell_params: (a, b, c) ellipsoid semi-axes
        n_ghost: number of ghost points far outside the shell surface
        ghost_margin: distance beyond shell where ghost ellipsoid is placed;
            large values (e.g. 5.0) ensure ghost Voronoi boundaries lie far
            outside the shell so real cells own the full interior up to the shell
        n_shell_pts: number of surface points used to build the convex-hull
            approximation of the ellipsoidal boundary for clipping
        seed: random seed for z-jitter reproducibility

    Returns:
        list[VoronoiCell] in same order as physics_cells
    """
    from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi

    a, b, c = shell_params
    n_real = len(physics_cells)

    centers = np.array([positions[pc.center_idx] for pc in physics_cells])

    # Ghost points placed far outside the shell prevent infinite Voronoi regions
    # without claiming interior volume from real cells.
    ghost_pts = make_ghost_points(a, b, c, n=n_ghost, margin=ghost_margin)

    # Build halfspace representation of the ellipsoidal shell boundary via a
    # high-resolution convex hull of surface points.
    shell_surface = make_ghost_points(a, b, c, n=n_shell_pts, margin=0.0)
    shell_hull = ConvexHull(shell_surface)
    shell_eqs = shell_hull.equations  # shape (F, 4): [A | b], A@x + b <= 0 inside

    # Tiny z-jitter on real cells prevents near-planar degenerate Voronoi
    rng = np.random.default_rng(seed=seed)
    z_jitter = rng.uniform(-1e-4, 1e-4, size=n_real)
    jittered = centers.copy()
    jittered[:, 2] += z_jitter

    all_pts = np.vstack([jittered, ghost_pts])
    vor = Voronoi(all_pts)

    result: list[VoronoiCell] = []
    for i in range(n_real):
        # Collect Voronoi halfspaces for cell i from the ridge structure.
        # For each ridge between cell i and neighbor j, the halfspace is the
        # half-plane closer to i than j: n @ x + b <= 0, n pointing away from i.
        cell_hs: list[np.ndarray] = []
        for pts_pair, verts_idx in zip(vor.ridge_points, vor.ridge_vertices):
            ci, cj = int(pts_pair[0]), int(pts_pair[1])
            if i not in (ci, cj):
                continue
            other = cj if ci == i else ci
            mid = (all_pts[ci] + all_pts[cj]) * 0.5
            normal = all_pts[other] - all_pts[i]
            norm_mag = float(np.linalg.norm(normal))
            if norm_mag < 1e-12:
                continue
            normal = normal / norm_mag
            offset = float(-normal @ mid)
            cell_hs.append(np.append(normal, offset))

        if len(cell_hs) < 4:
            # Not enough constraints: return zero-volume fallback
            result.append(
                VoronoiCell(
                    index=i,
                    vertices=np.zeros((0, 3)),
                    volume=0.0,
                    face_areas=[],
                    face_normals=[],
                    face_neighbors=[],
                    contact_areas={},
                    contact_normals={},
                )
            )
            continue

        # Clip Voronoi region to ellipsoidal shell by intersecting halfspaces
        all_hs = np.vstack([cell_hs, shell_eqs])
        feasible = jittered[i]

        try:
            hi = HalfspaceIntersection(all_hs, feasible)
            verts = hi.intersections  # (M, 3) clipped vertices
            hull = ConvexHull(verts)
            volume = float(hull.volume)
        except Exception:
            verts = np.zeros((0, 3))
            volume = 0.0

        result.append(
            VoronoiCell(
                index=i,
                vertices=verts,
                volume=volume,
                face_areas=[],
                face_normals=[],
                face_neighbors=[],
                contact_areas={},
                contact_normals={},
            )
        )

    # Fill face data (areas, normals, contacts) from ridge information
    _fill_face_data(result, vor, centers, n_real)

    return result


def _fill_face_data(
    cells: list[VoronoiCell],
    vor,
    centers: np.ndarray,
    n_real: int,
) -> None:
    """
    Populate face_areas, face_normals, face_neighbors, contact_areas, contact_normals
    for each VoronoiCell by iterating over scipy Voronoi ridges.

    Modifies cells in-place.
    """
    # Temporary accumulator: {cell_idx: [(area, inward_normal, neighbor_or_None)]}
    face_acc: dict[int, list] = {i: [] for i in range(n_real)}

    # Combined real + ghost points used by scipy Voronoi (same order as construction):
    # first n_real entries are real centers; remaining are ghost points. We need ghost
    # positions to compute correct face normals on real/ghost ridges, so reconstruct
    # them from vor.points (works because Voronoi keeps the input ordering).
    all_pts = vor.points

    for pts_pair, verts_idx in zip(vor.ridge_points, vor.ridge_vertices):
        ci, cj = int(pts_pair[0]), int(pts_pair[1])

        # Skip ghost-ghost ridges and infinite ridges
        if ci >= n_real and cj >= n_real:
            continue
        if -1 in verts_idx:
            continue

        face_verts = vor.vertices[list(verts_idx)]  # (M_face, 3)
        if len(face_verts) < 3:
            continue

        area = _polygon_area_3d(face_verts)
        if area < 1e-12:
            continue

        # Add face to each participating real cell. The Voronoi face between two
        # equal-weight sites is the perpendicular bisector, so the true inward
        # normal for cell `real_idx` points along (centers[real_idx] - other_pos).
        # Using the face centroid here is wrong: the centroid lies in the face
        # plane but not on the line between the two sites, so the resulting
        # vector has a spurious in-plane component (and the two cells' normals
        # are no longer antiparallel).
        for real_idx, other_idx in [(ci, cj), (cj, ci)]:
            if real_idx >= n_real:
                continue
            neighbor = other_idx if other_idx < n_real else None

            other_pos = all_pts[other_idx]
            inward = centers[real_idx] - other_pos
            norm = np.linalg.norm(inward)
            inward = inward / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])

            face_acc[real_idx].append((area, inward, neighbor))

    # Write accumulated data into each VoronoiCell
    for i, vc in enumerate(cells):
        faces = face_acc[i]
        vc.face_areas = [f[0] for f in faces]
        vc.face_normals = [f[1] for f in faces]
        vc.face_neighbors = [f[2] for f in faces]
        vc.contact_areas = {
            f[2]: f[0] for f in faces if f[2] is not None
        }
        vc.contact_normals = {
            f[2]: f[1] for f in faces if f[2] is not None
        }


def _polygon_area_3d(verts: np.ndarray) -> float:
    """Area of a planar polygon in 3D via triangle fan from first vertex."""
    area = 0.0
    v0 = verts[0]
    for k in range(1, len(verts) - 1):
        cross = np.cross(verts[k] - v0, verts[k + 1] - v0)
        area += 0.5 * float(np.linalg.norm(cross))
    return area
