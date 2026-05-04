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


def run_simulation(n_division_steps=4, n_settle_steps=200):
    lineage = CellLineage(YAML_PATH)

    positions = np.zeros((0, 3))
    p0, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        name="P0",
    )

    world = World(
        cell_bodies=[p0],
        positions=positions,
        lineage=lineage,
        shell_params=(4.0, 4.0, 4.0),
        dt=0.01,
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


def plot_4cell(world):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for cb in world.cell_bodies:
        center = world.positions[cb.center_idx]
        color = CELL_COLORS.get(cb.name, "gray")
        r = (3.0 * cb.target_volume / 4.0) ** (1.0 / 3.0)

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + r * np.outer(np.cos(u), np.sin(v))
        y = center[1] + r * np.outer(np.sin(u), np.sin(v))
        z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_surface(x, y, z, alpha=0.4, color=color)
        ax.text(*center, cb.name, fontsize=9, ha="center")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("X (A-P axis)")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("C. elegans 4-cell stage (XPBD)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    world = run_simulation()
    plot_4cell(world)
