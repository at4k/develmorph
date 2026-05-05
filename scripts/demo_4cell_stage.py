"""
Demo: C. elegans 4-cell stage via XPBD simulation.

Runs lineage-driven division from 1-cell (P0) to 4-cell stage (ABa, ABp, EMS, P2),
then lets the XPBD solver settle cells into a physical arrangement.

Run with:
    uv run python scripts/demo_4cell_stage.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from develmorph.cell_body import create_cell_body
from develmorph.celllineage import CellLineage
from develmorph.engine import World

YAML_PATH = "develmorph/lineages/celegans_early.yaml"
CELL_COLORS = {
    "ABa": "royalblue",
    "ABp": "cornflowerblue",
    "EMS": "tomato",
    "P2":  "darkorange",
}


def run_simulation(n_division_steps=4, n_settle_steps=300):
    lineage = CellLineage(YAML_PATH)

    positions = np.zeros((0, 3))
    p0, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        name="P0",
    )

    # Eggshell dimensions from Kajita et al. 2003, Table 1:
    # Rx=29.77, Ry=20.27, Rz=16.83 um → biological ratio 1.77:1.21:1.00
    # Scaled to 1.3x (2.30, 1.57, 1.30) so spherical P0 (r=1.0) fits without divergence.
    # Biological P0 is ellipsoidal (~16um z in 16.83um egg); spherical model needs larger shell.
    # n_iter=100: large initial overlaps after division require many iterations to converge
    world = World(
        cell_bodies=[p0],
        positions=positions,
        lineage=lineage,
        shell_params=(2.30, 1.57, 1.30),
        dt=0.01,
        n_iter=100,
        damping=1.0,
        shell_compliance=0.001,
    )

    # Division phase: steps 0-3 (division at step_counts 1, 2, 3)
    for _ in range(n_division_steps):
        world.step()

    print(f"After {n_division_steps} steps: {[cb.name for cb in world.cell_bodies]}")

    # Settling phase: continue stepping (no more divisions)
    world.lineage = None  # Disable further divisions
    for _ in range(n_settle_steps):
        world.step()

    return world


def _draw_eggshell(ax, shell_params, alpha=0.08, color="silver"):
    a, b, c = shell_params
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = a * np.outer(np.cos(u), np.sin(v))
    ys = b * np.outer(np.sin(u), np.sin(v))
    zs = c * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=alpha, color=color)
    ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.15, linewidth=0.4, rstride=4, cstride=4)


def plot_4cell(world):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    _draw_eggshell(ax, world.shell_params)

    for cb in world.cell_bodies:
        center = world.positions[cb.center_idx]
        color = CELL_COLORS.get(cb.name, "gray")
        r = (3.0 * cb.target_volume / 4.0) ** (1.0 / 3.0)

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + r * np.outer(np.cos(u), np.sin(v))
        y = center[1] + r * np.outer(np.sin(u), np.sin(v))
        z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.5, color=color)
        ax.text(*center, cb.name, fontsize=9, ha="center", weight="bold")

    a, b, c = world.shell_params
    ax.set_xlim(-a * 1.1, a * 1.1)
    ax.set_ylim(-b * 1.1, b * 1.1)
    ax.set_zlim(-c * 1.1, c * 1.1)
    ax.set_xlabel("X (A-P axis)")
    ax.set_ylabel("Y (V-D axis)")
    ax.set_zlabel("Z (L-R axis)")
    ax.set_title("C. elegans 4-cell stage (XPBD)\nshell: {:.2f} x {:.2f} x {:.2f}".format(*world.shell_params))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    world = run_simulation()
    plot_4cell(world)
