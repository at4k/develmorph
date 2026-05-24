"""
TUI cell visualizer for WorldPC simulation.

Runs the C. elegans 4-cell division sequence and displays
cell center positions as a 2D ASCII plot after each key event.

Run with:
    uv run python scripts/tui_cells.py
"""

import math
import os
import time

import numpy as np

from develmorph.celllineage import CellLineage
from develmorph.engine import WorldPC
from develmorph.physics_cell import create_physics_cell

# --- Display parameters ---
SHELL = (1.84, 1.26, 1.04)
YAML = "develmorph/lineages/celegans_early.yaml"

# Grid size (characters)
W = 70   # width
H = 28   # height

# World-space range shown
X_RANGE = (-2.2, 2.2)
Y_RANGE = (-1.5, 1.5)

# Cell name display colors (ANSI)
CELL_COLORS = {
    "P0":  "\033[97m",   # white
    "AB":  "\033[96m",   # cyan
    "P1":  "\033[93m",   # yellow
    "ABa": "\033[92m",   # green
    "ABp": "\033[94m",   # blue
    "EMS": "\033[91m",   # red
    "P2":  "\033[95m",   # magenta
}
RESET = "\033[0m"
DIM   = "\033[2m"


def world_to_grid(x, y):
    """Convert world coordinates to grid (col, row)."""
    col = int((x - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0]) * (W - 1))
    # Invert y: world +y = grid top
    row = int((Y_RANGE[1] - y) / (Y_RANGE[1] - Y_RANGE[0]) * (H - 1))
    return col, row


def draw(physics_cells, positions, title="", subtitle=""):
    """Render cell positions to a character grid and print it."""
    grid = [[" "] * W for _ in range(H)]

    # Draw shell ellipse (dashed)
    a, b = SHELL[0], SHELL[1]
    for t in range(300):
        angle = 2 * math.pi * t / 300
        ex = a * math.cos(angle)
        ey = b * math.sin(angle)
        col, row = world_to_grid(ex, ey)
        if 0 <= col < W and 0 <= row < H:
            grid[row][col] = DIM + "·" + RESET

    # Draw axes (faint)
    col0, row0 = world_to_grid(0, 0)
    for c in range(W):
        if 0 <= row0 < H and grid[row0][c] == " ":
            grid[row0][c] = DIM + "─" + RESET
    for r in range(H):
        if 0 <= col0 < W and grid[r][col0] == " ":
            grid[r][col0] = DIM + "│" + RESET

    # Draw each cell
    for pc in physics_cells:
        cx = positions[pc.center_idx, 0]
        cy = positions[pc.center_idx, 1]
        r  = pc.radius
        color = CELL_COLORS.get(pc.name or "", "\033[37m")

        # Cell circle
        for t in range(180):
            angle = 2 * math.pi * t / 180
            px = cx + r * math.cos(angle)
            py = cy + r * math.sin(angle)
            col, row = world_to_grid(px, py)
            if 0 <= col < W and 0 <= row < H:
                grid[row][col] = color + "o" + RESET

        # Center mark + name
        col, row = world_to_grid(cx, cy)
        if 0 <= col < W and 0 <= row < H:
            grid[row][col] = color + "+" + RESET
        name = pc.name or "?"
        # Place label to the right of center
        for i, ch in enumerate(name):
            nc = col + 1 + i
            if 0 <= nc < W and 0 <= row < H:
                grid[row][nc] = color + ch + RESET

    # Print frame
    os.system("clear")
    print(f"\033[1m{title}\033[0m")
    if subtitle:
        print(f"  {DIM}{subtitle}{RESET}")
    print("┌" + "─" * W + "┐")
    for row_idx, row in enumerate(grid):
        # Y-axis label on right edge
        world_y = Y_RANGE[1] - row_idx / (H - 1) * (Y_RANGE[1] - Y_RANGE[0])
        label = f" {world_y:+.1f}" if row_idx % 5 == 0 else ""
        print("│" + "".join(row) + "│" + label)
    print("└" + "─" * W + "┘")

    # X-axis labels
    x_labels = "  "
    for i in range(0, W, 10):
        world_x = X_RANGE[0] + i / (W - 1) * (X_RANGE[1] - X_RANGE[0])
        x_labels += f"{world_x:+.1f}".ljust(10)
    print(x_labels[:W + 4])
    print(f"  {DIM}A-P →   (x={X_RANGE[0]:.1f}..{X_RANGE[1]:.1f},  y=D-V){RESET}")
    print()

    # Cell info table
    print(f"  {'Name':6s}  {'x':>7s}  {'y':>7s}  {'r':>5s}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*5}")
    for pc in physics_cells:
        cx = positions[pc.center_idx, 0]
        cy = positions[pc.center_idx, 1]
        color = CELL_COLORS.get(pc.name or "", "")
        print(f"  {color}{pc.name or '?':6s}{RESET}  {cx:+7.3f}  {cy:+7.3f}  {pc.radius:5.3f}")


def main():
    lineage = CellLineage(YAML)
    positions = np.zeros((0, 3))
    p0, positions = create_physics_cell(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.3,
        positions=positions,
        name="P0",
    )

    world = WorldPC(
        physics_cells=[p0],
        positions=positions,
        lineage=lineage,
        shell_params=SHELL,
        dt=0.01,
        n_iter=100,
        damping=1.0,
        shell_compliance=0.001,
        adhesion_compliance=0.1,
        attach_margin=0.3,
        detach_margin=0.8,
    )

    stages = [
        (0,  "Step 0: P0 placed in shell",           "settling..."),
        (1,  "Step 1: P0 → AB + P1",                 "first cleavage (A-P axis)"),
        (2,  "Step 2: AB → ABa + ABp",               "AB divides (D-V axis)"),
        (3,  "Step 3: P1 → EMS + P2",                "P1 divides (A-P axis)"),
    ]

    # Run step by step, pausing at each division
    for step_idx, title, subtitle in stages:
        world.step()
        draw(world.physics_cells, world.positions, title=title, subtitle=subtitle)
        input("  Press Enter to continue...")

    # Settling phase
    world.lineage = None
    world.n_iter = 100
    print("  Settling (300 steps)...", end="", flush=True)
    for i in range(300):
        world.step()
        if (i + 1) % 60 == 0:
            print(".", end="", flush=True)
    print(" done")
    time.sleep(0.3)

    # Final state
    names = {pc.name: pc for pc in world.physics_cells}
    ems_y = abs(world.positions[names["EMS"].center_idx, 1]) if "EMS" in names else float("nan")
    draw(
        world.physics_cells,
        world.positions,
        title=f"Final: 4-cell stage  |EMS_y| = {ems_y:.3f}",
        subtitle="Diamond formation complete" if ems_y > 0.5 else "T-shape (tune parameters)",
    )
    print(f"  Diamond score: |EMS_y| = {ems_y:.4f}  "
          f"{'✓ DIAMOND' if ems_y > 0.5 else '✗ T-shape'}")


if __name__ == "__main__":
    main()
