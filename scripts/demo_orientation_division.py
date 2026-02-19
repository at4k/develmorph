"""
Phase2-B demonstration: Orientation-based cell division with eggshell.

Simulates C. elegans early embryo divisions using YAML-defined rules.
Cells are constrained inside an ellipsoidal eggshell and overlaps are resolved.
"""

import numpy as np

from develmorph.cell import Cell
from develmorph.celllineage import CellLineage
from develmorph.solver import step as xpbd_step
from develmorph.viz import visualize_cells_3d, visualize_timelapse_napari

# Eggshell parameters (ellipsoid semi-axes)
EGGSHELL_A = 2.0  # x (anterior-posterior)
EGGSHELL_B = 1.0  # y
EGGSHELL_C = 1.0  # z

# Initial cell radius (shrinks after each division)
INITIAL_RADIUS = 0.9


def divide_cell(parent: Cell, rule, lineage: CellLineage, next_id: int):
    """
    Divide parent cell into two daughters.
    Daughter radius shrinks to conserve volume.
    """
    axis = lineage.resolve_division_axis(rule, parent.orientation)

    # Separation distance based on parent radius
    dist = parent.radius * 0.6
    new_radius = parent.radius * Cell.DIVISION_RADIUS_FACTOR

    pos_a = parent.position + dist * axis
    pos_b = parent.position - dist * axis

    daughter_a = Cell(
        cell_id=next_id,
        name=rule.daughters[0],
        position=pos_a,
        radius=new_radius,
        orientation=parent.orientation.copy(),
    )
    daughter_b = Cell(
        cell_id=next_id + 1,
        name=rule.daughters[1],
        position=pos_b,
        radius=new_radius,
        orientation=parent.orientation.copy(),
    )

    return daughter_a, daughter_b


def apply_constraints(cells, n_iter=10):
    """
    Apply XPBD constraints to all cells.
    """
    cell_list = list(cells.values())
    positions = np.array([c.position for c in cell_list])
    radii = np.array([c.radius for c in cell_list])
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    positions, _ = xpbd_step(
        positions,
        velocities,
        inv_masses,
        radii,
        dt=0.01,
        shell_params=(EGGSHELL_A, EGGSHELL_B, EGGSHELL_C),
        n_iter=n_iter,
    )

    for i, cell in enumerate(cell_list):
        cell.position = positions[i]


def main():
    lineage = CellLineage("develmorph/lineages/celegans_early.yaml")

    # Initial cell: P0 at origin
    next_id = 0
    p0 = Cell(cell_id=next_id, name="P0", position=[0, 0, 0], radius=INITIAL_RADIUS)
    next_id += 1

    cells = {"P0": p0}

    # Record history for timelapse visualization
    history = []
    history.append(
        {
            name: Cell(
                c.cell_id, c.name, c.position.copy(), c.radius, c.orientation.copy()
            )
            for name, c in cells.items()
        }
    )

    print(f"Eggshell: a={EGGSHELL_A}, b={EGGSHELL_B}, c={EGGSHELL_C}")
    print(f"Initial radius: {INITIAL_RADIUS}")
    print(f"Division factor: {Cell.DIVISION_RADIUS_FACTOR:.3f}")
    print()
    print("Initial state:")
    print(f"  {p0}")
    print()

    step = 1
    while True:
        program = lineage.get_step_program(step)
        if program is None:
            break

        divisions_to_execute = [
            rule for rule in program.divisions if rule.parent in cells
        ]

        if not divisions_to_execute:
            break

        print(f"Step {step}:")

        for rule in divisions_to_execute:
            parent = cells[rule.parent]

            a, b = divide_cell(parent, rule, lineage, next_id)
            next_id += 2

            print(f"  {rule.parent} -> {a.name}, {b.name} (axis: {rule.axis})")

            del cells[rule.parent]
            cells[a.name] = a
            cells[b.name] = b

        # Apply physical constraints
        apply_constraints(cells)

        print(f"  After constraints: {len(cells)} cells")
        for name, cell in sorted(cells.items()):
            print(f"    {name}: r={cell.radius:.3f}, pos={cell.position.round(3)}")

        # Record state for timelapse
        history.append(
            {
                name: Cell(
                    c.cell_id, c.name, c.position.copy(), c.radius, c.orientation.copy()
                )
                for name, c in cells.items()
            }
        )

        print()
        step += 1

    print(f"Simulation ended at step {step}")
    print(f"Final cells ({len(cells)}):")
    for name, cell in sorted(cells.items()):
        print(f"  {cell}")

    # Visualize timelapse with napari
    visualize_timelapse_napari(
        history,
        eggshell=(EGGSHELL_A, EGGSHELL_B, EGGSHELL_C),
    )


if __name__ == "__main__":
    main()
