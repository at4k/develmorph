# File: develmorph/viz.py
# Phase2-A: 3D visualization of developmental time series

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def make_figure(positions, radii):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = positions.T

    ax.scatter(xs, ys, zs)

    return fig, ax


def plot_cells(positions, radii):

    fig, ax = make_figure(positions, radii)
    plt.show()


def plot_cells_with_polarity(pos, r, p):

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=r * 500, c=p, cmap="coolwarm")

    plt.colorbar(sc)
    plt.show()


def visualize_lineage_3d(df: pd.DataFrame):
    """
    Visualize lineage-driven cell trajectories in 3D.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of LineageRecorder.to_dataframe().
        Required columns: step, cell_id, lineage_id, x, y, z
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for cell_id, group in df.groupby("cell_id"):
        ax.plot(
            group["x"],
            group["y"],
            group["z"],
            marker="o",
            label=f"cell {cell_id}",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Phase2-A Lineage-driven Cell Division")

    plt.legend()
    plt.show()


def visualize_cells_3d(
    cells, title="Cell Division", show_orientation=True, eggshell=None
):
    """
    Visualize cells in 3D with their orientations.

    Parameters
    ----------
    cells : dict[str, Cell] or list[Cell]
        Cells to visualize
    title : str
        Plot title
    show_orientation : bool
        If True, draw arrows showing e1 (red), e2 (green), e3 (blue)
    eggshell : tuple(a, b, c) or None
        If provided, draw ellipsoidal eggshell with semi-axes (a, b, c)
    """
    import numpy as np

    if isinstance(cells, dict):
        cells = list(cells.values())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection="3d")

    # Collect positions
    positions = np.array([c.position for c in cells])

    # Plot cells as spheres
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=200,
        alpha=0.6,
    )

    # Add labels and orientation arrows
    arrow_length = 0.3
    for cell in cells:
        pos = cell.position

        # Label
        ax.text(pos[0], pos[1], pos[2] + 0.15, cell.name, fontsize=10, ha="center")

        if show_orientation:
            # e1 (major axis) - red
            e1 = cell.orientation[:, 0]
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                e1[0],
                e1[1],
                e1[2],
                length=arrow_length,
                color="red",
                arrow_length_ratio=0.3,
            )

            # e2 (minor axis) - green
            e2 = cell.orientation[:, 1]
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                e2[0],
                e2[1],
                e2[2],
                length=arrow_length,
                color="green",
                arrow_length_ratio=0.3,
            )

            # e3 (orthogonal) - blue
            e3 = cell.orientation[:, 2]
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                e3[0],
                e3[1],
                e3[2],
                length=arrow_length,
                color="blue",
                arrow_length_ratio=0.3,
            )

    # Set equal aspect ratio
    if eggshell is not None:
        max_range = max(eggshell) + 0.2
    else:
        max_range = np.max(np.abs(positions)) + 0.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    # Draw eggshell
    if eggshell is not None:
        a, b, c = eggshell
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = a * np.outer(np.cos(u), np.sin(v))
        y = b * np.outer(np.sin(u), np.sin(v))
        z = c * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, linewidth=0.5)

    # Legend for orientation
    if show_orientation:
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="red", label="e1 (major)"),
            Line2D([0], [0], color="green", label="e2 (minor)"),
            Line2D([0], [0], color="blue", label="e3 (orthogonal)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.show()


def visualize_timelapse_napari(
    history, eggshell=None, surface_history=None, adhesion_history=None
):
    """
    Visualize cell division timelapse using napari.

    Parameters
    ----------
    history : list[dict[str, Cell]]
        List of cell states at each step
    eggshell : tuple(a, b, c) or None
        Ellipsoid semi-axes for eggshell surface
    surface_history : list[list[np.ndarray]] or None
        Per-step list of surface point arrays (N_cells x 6 x 3)
    adhesion_history : list[list[tuple[np.ndarray, np.ndarray]]] or None
        Per-step list of (center_a, center_b) pairs for adhesion lines
    """
    import napari
    import numpy as np

    viewer = napari.Viewer(title="C. elegans Embryo Development")

    # Prepare data for all timepoints
    all_points = []
    all_sizes = []
    all_features = {"step": [], "name": []}

    for step, cells in enumerate(history):
        cell_list = list(cells.values())
        for cell in cell_list:
            all_points.append(
                [step, cell.position[2], cell.position[1], cell.position[0]]
            )
            all_sizes.append(cell.radius * 2)
            all_features["step"].append(step)
            all_features["name"].append(cell.name)

    all_points = np.array(all_points)
    all_sizes = np.array(all_sizes)

    # Cell centers (small markers with labels)
    viewer.add_points(
        all_points,
        size=0.15,
        face_color="cyan",
        border_color="blue",
        border_width=0.02,
        features=all_features,
        text={
            "string": "{name}",
            "size": 10,
            "color": "white",
            "anchor": "center",
        },
        name="Cell Centers",
    )

    # Surface points (octahedron vertices)
    if surface_history is not None:
        surf_points = []
        for step, surfaces in enumerate(surface_history):
            for pts in surfaces:
                for pt in pts:
                    surf_points.append([step, pt[2], pt[1], pt[0]])
        if surf_points:
            surf_points = np.array(surf_points)
            viewer.add_points(
                surf_points,
                size=0.06,
                face_color="yellow",
                opacity=1.0,
                name="Surface Points",
            )

    # Adhesion lines (connecting centers of adhered cells)
    # Use add_vectors for 3D visibility (add_shapes lines are 2D only)
    if adhesion_history is not None:
        vec_origins = []
        vec_dirs = []
        for step, pairs in enumerate(adhesion_history):
            for ca, cb in pairs:
                origin = np.array([step, ca[2], ca[1], ca[0]])
                target = np.array([step, cb[2], cb[1], cb[0]])
                vec_origins.append(origin)
                vec_dirs.append(target - origin)
        if vec_origins:
            vec_data = np.stack([vec_origins, vec_dirs], axis=1)
            viewer.add_vectors(
                vec_data,
                edge_color="orange",
                edge_width=0.05,
                length=1.0,
                opacity=1.0,
                name="Adhesion",
            )

    # Add eggshell surface
    if eggshell is not None:
        a, b, c = eggshell
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 20)
        x = a * np.outer(np.cos(u), np.sin(v))
        y = b * np.outer(np.sin(u), np.sin(v))
        z = c * np.outer(np.ones_like(u), np.cos(v))

        shell_points = []
        for step in range(len(history)):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    shell_points.append([step, z[i, j], y[i, j], x[i, j]])

        shell_points = np.array(shell_points)
        viewer.add_points(
            shell_points,
            size=0.02,
            face_color="gray",
            opacity=1.0,
            name="Eggshell",
        )

    # Set camera
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (30, 30, 120)

    napari.run()


import numpy as np
from develmorph.cell_body import OCTAHEDRON_DIRS


class SurfaceShell:
    """
    Kinematic surface mesh for a PhysicsCell.

    Reconstructed each step from center + radius + contact records.
    Does NOT participate in XPBD. No drift possible.

    Points are octahedron vertices (6 pts at n_subdivisions=0) optionally
    flattened at contact planes proportional to overlap.
    """

    def __init__(self, center, radius, contacts):
        """
        Args:
            center: (3,) array, cell center position
            radius: float, cell radius
            contacts: list of ContactRecord involving this cell
        """
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.contacts = contacts

    def mesh_points(self, n_subdivisions: int = 0) -> np.ndarray:
        """
        Return (M, 3) surface mesh points.

        With no contacts, returns 6 points on the sphere surface (octahedron
        vertices). With contacts, flattens points on the contact-facing side
        proportional to overlap.

        The contact-facing side for cell i is in the -normal direction
        (normal = unit vector from j to i). Points facing toward j are pulled
        inward by flatten_amount * projection.

        Args:
            n_subdivisions: reserved for future subdivision (currently 6 pts always)

        Returns:
            (M, 3) array of world-space surface points
        """
        # Base sphere: 6 octahedron vertices in world space
        pts = self.center + self.radius * OCTAHEDRON_DIRS.astype(float)

        # Flatten at each contact plane (work in world space, no re-normalization)
        for record in self.contacts:
            # normal points from j toward i; contact face of cell i is in -normal direction
            n = record.normal  # unit vector from j to i
            flatten_amount = min(record.overlap * 0.5, self.radius * 0.3)
            for k in range(len(pts)):
                offset = pts[k] - self.center
                # proj > 0 when this point faces toward j (contact-facing side)
                proj = np.dot(offset / self.radius, -n)
                if proj > 0.0:
                    # Pull inward: move in +n direction (away from j = toward center)
                    pts[k] = pts[k] + flatten_amount * proj * n

        return pts


def reconstruct_surface_shells(
    physics_cells: list,
    positions: np.ndarray,
    contacts: list,
) -> list:
    """
    Build a SurfaceShell for each PhysicsCell from current physics state.

    Args:
        physics_cells: list of PhysicsCell
        positions: (N, 3) current center positions
        contacts: list of ContactRecord from compute_contacts()

    Returns:
        list of SurfaceShell, one per cell (same order as physics_cells)
    """
    shells = []
    for idx, pc in enumerate(physics_cells):
        cell_contacts = [r for r in contacts if r.i == idx or r.j == idx]
        shells.append(SurfaceShell(
            center=positions[pc.center_idx],
            radius=pc.radius,
            contacts=cell_contacts,
        ))
    return shells


class VoronoiMesh:
    """
    Kinematic surface mesh for a PhysicsCell based on Voronoi tessellation.

    Wraps a VoronoiCell and returns its vertices as the surface mesh.
    Not part of XPBD; reconstructed post-step. Zero drift by construction.
    """

    def __init__(self, voronoi_cell) -> None:
        """
        Args:
            voronoi_cell: VoronoiCell from compute_voronoi_cells()
        """
        self._cell = voronoi_cell

    def mesh_points(self) -> np.ndarray:
        """
        Return (M, 3) Voronoi cell vertices.

        Already bounded by shell ghost points; no additional clipping needed.
        """
        return self._cell.vertices


def reconstruct_voronoi_meshes(voronoi_cells: list) -> list:
    """
    Build a VoronoiMesh for each VoronoiCell.

    Args:
        voronoi_cells: list of VoronoiCell from compute_voronoi_cells()

    Returns:
        list of VoronoiMesh, one per cell (same order)
    """
    return [VoronoiMesh(vc) for vc in voronoi_cells]
