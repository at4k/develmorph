"""
Phase2-B demonstration: Orientation-based cell division.

Simulates C. elegans early embryo divisions using YAML-defined rules.
Orientation is inherited from parent; division axis is determined
relative to cell's local frame.
"""

import numpy as np
from develmorph.cell import Cell
from develmorph.celllineage import CellLineage
from develmorph.viz import visualize_cells_3d


def divide_cell(parent: Cell, rule, lineage: CellLineage, next_id: int, dist=0.5):
    """
    Divide parent cell into two daughters.

    Daughters inherit parent's orientation.
    Separation is along the resolved division axis.
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


def main():
    lineage = CellLineage("develmorph/lineages/celegans_early.yaml")

    # Initial cell: P0 at origin
    next_id = 0
    p0 = Cell(cell_id=next_id, name="P0", position=[0, 0, 0])
    next_id += 1

    cells = {"P0": p0}

    print("Initial state:")
    print(f"  {p0}")
    print(f"  e1={p0.orientation[:, 0]}")
    print()

    step = 1
    while True:
        program = lineage.get_step_program(step)
        if program is None:
            break

        # Check if any division is possible
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
            print(f"    {a.name}: pos={a.position}")
            print(f"    {b.name}: pos={b.position}")

            # Remove parent, add daughters
            del cells[rule.parent]
            cells[a.name] = a
            cells[b.name] = b

        print()
        step += 1

    print(f"Simulation ended at step {step}")
    print(f"Final cells ({len(cells)}):")
    for name, cell in sorted(cells.items()):
        print(f"  {cell}")

    # Visualize
    visualize_cells_3d(cells, title=f"C. elegans {len(cells)}-cell stage")


if __name__ == "__main__":
    main()
