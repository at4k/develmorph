"""
Phase2-B demonstration: Orientation-based cell division with eggshell.

Simulates C. elegans early embryo divisions using YAML-defined rules.
Uses XPBD solver with surface-point cell model (octahedron).
Cells are constrained inside an ellipsoidal eggshell with volume conservation.
"""

import numpy as np

from develmorph.cell import Cell
from develmorph.cell_body import build_particle_arrays, create_cell_body
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
    """Divide parent cell into two daughters (biological state only)."""
    axis = lineage.resolve_division_axis(rule, parent.orientation)
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


def apply_constraints(cells, cell_bodies, positions, active_adhesions, n_iter=10):
    """Apply XPBD constraints with surface-point model."""
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    positions, _ = xpbd_step(
        positions,
        velocities,
        inv_masses,
        dt=0.01,
        cell_bodies=cell_bodies,
        shell_params=(EGGSHELL_A, EGGSHELL_B, EGGSHELL_C),
        n_iter=n_iter,
        active_adhesions=active_adhesions,
    )

    # Update Cell positions from CellBody centers
    cell_list = list(cells.values())
    for i, cell in enumerate(cell_list):
        cell.position = positions[cell_bodies[i].center_idx].copy()

    return positions


def main():
    lineage = CellLineage("develmorph/lineages/celegans_early.yaml")

    # Initial cell: P0 at origin
    next_id = 0
    p0 = Cell(cell_id=next_id, name="P0", position=[0, 0, 0], radius=INITIAL_RADIUS)
    next_id += 1

    cells = {"P0": p0}

    # Create CellBody for P0
    positions = np.zeros((0, 3))
    cb_p0, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=INITIAL_RADIUS,
        positions=positions,
    )
    cell_bodies = [cb_p0]
    cell_body_map = {"P0": cb_p0}
    active_adhesions = set()

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
    print()
    print("Initial state:")
    print(f"  {p0}")
    print()

    step_num = 1
    while True:
        program = lineage.get_step_program(step_num)
        if program is None:
            break

        divisions_to_execute = [
            rule for rule in program.divisions if rule.parent in cells
        ]

        if not divisions_to_execute:
            break

        print(f"Step {step_num}:")

        for rule in divisions_to_execute:
            parent = cells[rule.parent]
            parent_cb = cell_body_map[rule.parent]

            a, b = divide_cell(parent, rule, lineage, next_id)
            next_id += 2

            print(f"  {rule.parent} -> {a.name}, {b.name} (axis: {rule.axis})")

            # Remove parent
            del cells[rule.parent]
            del cell_body_map[rule.parent]
            cell_bodies.remove(parent_cb)

            # Remove parent particles
            parent_indices = sorted(parent_cb.all_indices(), reverse=True)
            positions = np.delete(positions, parent_indices, axis=0)

            # Create daughter CellBodies
            cb_a, positions = create_cell_body(
                center=a.position, radius=a.radius, positions=positions
            )
            cb_b, positions = create_cell_body(
                center=b.position, radius=b.radius, positions=positions
            )

            cells[a.name] = a
            cells[b.name] = b
            cell_bodies.extend([cb_a, cb_b])
            cell_body_map[a.name] = cb_a
            cell_body_map[b.name] = cb_b

        # Rebuild indices after all divisions
        positions, _ = build_particle_arrays(cell_bodies, positions)
        active_adhesions.clear()

        # Apply constraints
        positions = apply_constraints(cells, cell_bodies, positions, active_adhesions)

        print(f"  After constraints: {len(cells)} cells")
        for name, cell in sorted(cells.items()):
            print(f"    {name}: r={cell.radius:.3f}, pos={cell.position.round(3)}")

        history.append(
            {
                name: Cell(
                    c.cell_id, c.name, c.position.copy(), c.radius, c.orientation.copy()
                )
                for name, c in cells.items()
            }
        )

        print()
        step_num += 1

    print(f"Simulation ended at step {step_num}")
    print(f"Final cells ({len(cells)}):")
    for name, cell in sorted(cells.items()):
        print(f"  {cell}")

    visualize_timelapse_napari(
        history,
        eggshell=(EGGSHELL_A, EGGSHELL_B, EGGSHELL_C),
    )


if __name__ == "__main__":
    main()
