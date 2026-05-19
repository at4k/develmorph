"""
Demo: C. elegans 4-cell stage via XPBD simulation.

Runs lineage-driven division from 1-cell (P0) to 4-cell stage (ABa, ABp, EMS, P2),
then lets the XPBD solver settle cells into a physical arrangement.

Run with:
    uv run python scripts/demo_4cell_stage.py           # T-shape (sequential)
    uv run python scripts/demo_4cell_stage.py --diamond  # diamond (simultaneous)
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from develmorph.cell_body import create_cell_body
from develmorph.celllineage import CellLineage
from develmorph.engine import World

YAML_SEQUENTIAL = "develmorph/lineages/celegans_early.yaml"
YAML_SYNC       = "develmorph/lineages/celegans_4cell_sync.yaml"

CELL_COLORS = {
    "ABa": "royalblue",
    "ABp": "cornflowerblue",
    "EMS": "tomato",
    "P2":  "darkorange",
}


def run_simulation(diamond=False, n_settle_steps=300, record_every=5):
    if diamond:
        yaml_path = YAML_SYNC
        n_division_steps = 72  # step1: P0 div, step51: AB div, step71: P1 div, +1 settle
        n_iter_division = 100  # full iterations so cells settle between divisions
    else:
        yaml_path = YAML_SEQUENTIAL
        n_division_steps = 4   # step1..3: sequential divisions, step4: no-op
        n_iter_division = 100

    lineage = CellLineage(yaml_path)

    positions = np.zeros((0, 3))
    p0, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.4,
        positions=positions,
        name="P0",
        topology='ico',
    )

    # Eggshell dimensions from Kajita et al. 2003, Table 1:
    # Rx=29.77, Ry=20.27, Rz=16.83 um → biological ratio 1.77:1.21:1.00
    # (1.84, 1.26, 1.04): 1.04x biological ratio; cells nearly touching (gap~0.01)
    # while maintaining diamond configuration. Smaller shells break the arrangement.
    world = World(
        cell_bodies=[p0],
        positions=positions,
        lineage=lineage,
        shell_params=(1.84, 1.26, 1.04),
        dt=0.01,
        n_iter=n_iter_division,
        damping=1.0,
        shell_compliance=0.0,
        membrane_compliance=0.02,
        volume_compliance=0.1,
        contact_overlap=0.10,
        surface_contact_compliance=0.1,
        area_compliance=0.0,
    )

    frames = []  # list of (positions_copy, [(name, center_idx, surface_indices, target_volume)])

    def _record(label=None):
        snap = [
            (cb.name, cb.center_idx, list(cb.surface_indices), cb.target_volume, list(cb.triangles))
            for cb in world.cell_bodies
        ]
        frames.append((world.positions.copy(), snap, label or f"t={world.time:.2f}"))

    # Division phase
    for _ in range(n_division_steps):
        world.step()
        _record(f"division step {world.step_count}")

    print(f"After {n_division_steps} division steps: {[cb.name for cb in world.cell_bodies]}")

    # Settling phase
    world.n_iter = 100
    world.lineage = None
    for i in range(n_settle_steps):
        world.step()
        if i % record_every == 0:
            _record()

    return world, frames


def _draw_eggshell(ax, shell_params, alpha=0.08, color="silver"):
    a, b, c = shell_params
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = a * np.outer(np.cos(u), np.sin(v))
    ys = b * np.outer(np.sin(u), np.sin(v))
    zs = c * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=alpha, color=color)
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.15, linewidth=0.4, rstride=4, cstride=4)


def _draw_cell_sphere(ax, center, r, color):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.45, color=color)


def _draw_cell_mesh(ax, positions, cb, color):
    tris = cb.surface_global_triangles()
    verts = [[positions[i], positions[j], positions[k]] for i, j, k in tris]
    poly = Poly3DCollection(verts, alpha=0.55, facecolor=color, edgecolor="black", linewidth=0.3)
    ax.add_collection3d(poly)
    # Draw surface particles
    for si in cb.surface_indices:
        p = positions[si]
        ax.scatter(*p, color=color, s=18, depthshade=False, zorder=5)


def plot_4cell(world, mesh=False):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    _draw_eggshell(ax, world.shell_params)

    for cb in world.cell_bodies:
        center = world.positions[cb.center_idx]
        color = CELL_COLORS.get(cb.name, "gray")

        if mesh:
            _draw_cell_mesh(ax, world.positions, cb, color)
        else:
            r = (3.0 * cb.target_volume / 4.0) ** (1.0 / 3.0)
            _draw_cell_sphere(ax, center, r, color)

        ax.text(*center, cb.name, fontsize=9, ha="center", weight="bold")

    a, b, c = world.shell_params
    ax.set_xlim(-a * 1.1, a * 1.1)
    ax.set_ylim(-b * 1.1, b * 1.1)
    ax.set_zlim(-c * 1.1, c * 1.1)
    ax.set_box_aspect([a, b, c])
    ax.set_xlabel("X (A-P axis)")
    ax.set_ylabel("Y (V-D axis)")
    ax.set_zlabel("Z (L-R axis)")
    config = getattr(world, "_demo_config", "")
    ax.set_title(
        "C. elegans 4-cell stage (XPBD){}\nshell: {:.2f} x {:.2f} x {:.2f}".format(
            f" [{config}]" if config else "", *world.shell_params
        )
    )
    plt.tight_layout()
    plt.show()


def animate_4cell(world, frames):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.18)
    ax_slider = fig.add_axes([0.15, 0.06, 0.70, 0.03])
    slider = Slider(ax_slider, "Frame", 0, len(frames) - 1, valinit=0, valstep=1)
    a, b, c = world.shell_params

    def draw(fi):
        ax.cla()
        positions, snap, label = frames[fi]

        _draw_eggshell(ax, world.shell_params)

        for name, center_idx, surface_indices, target_volume, triangles in snap:
            color = CELL_COLORS.get(name, "gray")
            # Use outer surface triangles (sa, sb, sc) — no center particle.
            # This avoids visual artifacts when the center is offset from the
            # surface centroid due to contact-zone deformation.
            tris = [(surface_indices[a], surface_indices[b], surface_indices[c])
                    for a, b, c in triangles]
            verts = [[positions[i], positions[j], positions[k]] for i, j, k in tris]
            poly = Poly3DCollection(verts, alpha=0.55, facecolor=color, edgecolor="black", linewidth=0.3)
            ax.add_collection3d(poly)
            for si in surface_indices:
                ax.scatter(*positions[si], color=color, s=12, depthshade=False)
            # Label at surface centroid (not center particle) for accurate placement
            centroid = np.mean([positions[si] for si in surface_indices], axis=0)
            ax.text(*centroid, name, fontsize=9, ha="center", weight="bold")

        ax.set_xlim(-a * 1.1, a * 1.1)
        ax.set_ylim(-b * 1.1, b * 1.1)
        ax.set_zlim(-c * 1.1, c * 1.1)
        ax.set_box_aspect([a, b, c])
        ax.set_xlabel("X (A-P)")
        ax.set_ylabel("Y (V-D)")
        ax.set_zlabel("Z (L-R)")
        ax.set_title(f"C. elegans 4-cell stage  [{label}]  ({fi+1}/{len(frames)})")
        fig.canvas.draw_idle()

    slider.on_changed(lambda val: draw(int(val)))
    draw(0)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diamond", action="store_true", help="Run Exp2 (simultaneous division, diamond config)")
    parser.add_argument("--mesh", action="store_true", help="Draw actual octahedron surface particles instead of sphere approximation")
    parser.add_argument("--animate", action="store_true", help="Show time-series animation instead of final snapshot")
    args = parser.parse_args()

    world, frames = run_simulation(diamond=args.diamond)
    world._demo_config = "diamond" if args.diamond else "T-shape"

    if args.animate:
        _anim = animate_4cell(world, frames)
    else:
        plot_4cell(world, mesh=args.mesh)
