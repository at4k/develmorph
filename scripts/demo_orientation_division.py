"""
Phase2-B demonstration: Orientation-based cell division with eggshell.

Simulates C. elegans early embryo divisions using YAML-defined rules.
Cells are constrained inside an ellipsoidal eggshell and overlaps are resolved.
"""

import numpy as np
from develmorph.cell import Cell
from develmorph.celllineage import CellLineage
from develmorph.project import project_overlap, project_shell
from develmorph.viz import visualize_cells_3d


# Eggshell parameters (ellipsoid semi-axes)
EGGSHELL_A = 2.0  # x (anterior-posterior)
EGGSHELL_B = 1.0  # y
EGGSHELL_C = 1.0  # z

# Cell radius for overlap detection
CELL_RADIUS = 0.8


def divide_cell(parent: Cell, rule, lineage: CellLineage, next_id: int, dist=0.4):
    """
    Divide parent cell into two daughters.
    """
    axis = lineage.resolve_division_axis(rule, parent.orientation)

    pos_a = parent.position + dist * axis
    pos_b = parent.position - dist * axis

    daughter_a = Cell(
        cell_id=next_id,
        name=rule.daughters[0],
        position=pos_a,
        orientation=parent.orientation.copy(),
    )
    daughter_b = Cell(
        cell_id=next_id + 1,
        name=rule.daughters[1],
        position=pos_b,
        orientation=parent.orientation.copy(),
    )

    return daughter_a, daughter_b


def apply_constraints(cells, n_iter=10):
    """
    Apply overlap and eggshell constraints to all cells.
    """
    cell_list = list(cells.values())
    positions = np.array([c.position for c in cell_list])
    radii = np.full(len(cell_list), CELL_RADIUS)

    # Iteratively apply constraints
    for _ in range(n_iter):
        positions = project_overlap(positions, radii, n_iter=1)
        positions = project_shell(positions, EGGSHELL_A, EGGSHELL_B, EGGSHELL_C)

    # Update cell positions
    for i, cell in enumerate(cell_list):
        cell.position = positions[i]


def main():
    lineage = CellLineage("develmorph/lineages/celegans_early.yaml")

    # Initial cell: P0 at origin
    next_id = 0
    p0 = Cell(cell_id=next_id, name="P0", position=[0, 0, 0])
    next_id += 1

    cells = {"P0": p0}

    print(f"Eggshell: a={EGGSHELL_A}, b={EGGSHELL_B}, c={EGGSHELL_C}")
    print(f"Cell radius: {CELL_RADIUS}")
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
            rule for rule in program.divisions
            if rule.parent in cells
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
            print(f"    {name}: {cell.position.round(3)}")

        print()
        step += 1

    print(f"Simulation ended at step {step}")
    print(f"Final cells ({len(cells)}):")
    for name, cell in sorted(cells.items()):
        print(f"  {cell}")

    # Visualize with eggshell
    visualize_cells_3d(
        cells,
        title=f"C. elegans {len(cells)}-cell stage",
        eggshell=(EGGSHELL_A, EGGSHELL_B, EGGSHELL_C),
    )


if __name__ == "__main__":
    main()
