"""
Diamond formation validation for WorldPC (PhysicsCell model).

Runs the same division sequence as experiment_diamond.py but using the
center-only XPBD model with ShellConstraint(radius=pc.radius).

Success criterion: |EMS_y| > 0.5 (comparable to current CellBody model at 0.84).

Run with:
    uv run python scripts/demo_diamond_pc.py
"""

import numpy as np

from develmorph.celllineage import CellLineage
from develmorph.engine import WorldPC
from develmorph.physics_cell import create_physics_cell

SHELL = (1.84, 1.26, 1.04)
N_SETTLE = 300
YAML = "develmorph/lineages/celegans_early.yaml"


def run_diamond(label, n_iter_div, n_iter_settle, n_div_steps=4):
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
        n_iter=n_iter_div,
        damping=1.0,
        shell_compliance=0.001,
        adhesion_compliance=0.1,
        attach_margin=0.3,
        detach_margin=0.8,
    )

    for _ in range(n_div_steps):
        world.step()

    print(f"  [{label}] cells after divisions: "
          f"{[pc.name for pc in world.physics_cells]}")

    # Settling phase
    world.n_iter = n_iter_settle
    world.lineage = None
    for _ in range(N_SETTLE):
        world.step()

    return world


def score(world):
    pos = {pc.name: world.positions[pc.center_idx] for pc in world.physics_cells}
    ems_y = abs(pos["EMS"][1]) if "EMS" in pos else float("nan")
    return pos, ems_y


def main():
    print("=" * 60)
    print("Diamond Validation -- WorldPC (PhysicsCell model)")
    print(f"Shell: {SHELL}  Settle: {N_SETTLE} steps")
    print("=" * 60)

    experiments = [
        ("n_iter=100", 100, 100),
        ("n_iter=10",  10,  100),
    ]

    results = {}
    for label, n_div, n_set in experiments:
        print(f"\nRunning: {label}")
        world = run_diamond(label, n_div, n_set)
        pos, ems_y = score(world)
        results[label] = ems_y
        print(f"\n  --- {label} ---")
        for name in ["ABa", "ABp", "EMS", "P2"]:
            if name in pos:
                p = pos[name]
                print(f"    {name:4s}  x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f}")
        print(f"  |EMS_y| = {ems_y:.4f}  "
              f"{'=> DIAMOND tendency detected' if ems_y > 0.5 else '=> T-shape (tune needed)'}")

    print("\n" + "=" * 60)
    print("Summary: Diamond score (|EMS_y|)")
    print("  T-shape reference: ~0.0 | Diamond reference: >0.5")
    print("-" * 60)
    for label, ems_y in results.items():
        bar = "#" * int(ems_y * 20)
        status = "PASS" if ems_y > 0.5 else "FAIL"
        print(f"  {ems_y:.3f} |{bar:<20}| {label}  [{status}]")
    print("=" * 60)

    passed = all(v > 0.5 for v in results.values())
    if passed:
        print("\nPhase 1 gate: PASSED. Proceed to Phase 2.")
    else:
        print("\nPhase 1 gate: NEEDS TUNING.")
        print("Try: lower attach_margin (0.1-0.2) or lower adhesion_compliance (0.01)")


if __name__ == "__main__":
    main()
