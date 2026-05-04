import numpy as np

from develmorph.cell_body import build_particle_arrays
from develmorph.celllineage import CellLineage
from develmorph.division import divide_cell_body
from develmorph.polarity import PolarityField
from develmorph.solver import step


class World:
    def __init__(self, cell_bodies, positions, dt=0.01, shell_params=(5.0, 5.0, 5.0), lineage=None):

        self.cell_bodies = list(cell_bodies)
        self.positions = positions.copy()

        self.velocities = np.zeros_like(positions)
        self.inv_masses = np.ones(len(positions))

        self.dt = dt
        self.time = 0
        self.step_count = 0

        self.shell_params = shell_params
        self.lineage = lineage

        self.division_steps = []

        self.polarity = PolarityField()
        centers = np.array([positions[cb.center_idx] for cb in cell_bodies])
        self.cell_polarity = self.polarity.value(centers)
        self.active_adhesions = set()

    def set_divisions(self, steps):
        self.division_steps = steps

    def _apply_lineage_divisions(self):
        program = self.lineage.get_step_program(self.step_count)
        if program is None:
            return

        for rule in program.divisions:
            idx = next(
                (i for i, cb in enumerate(self.cell_bodies) if cb.name == rule.parent),
                None,
            )
            if idx is None:
                continue

            parent = self.cell_bodies[idx]
            axis = self.lineage.resolve_division_axis(rule, parent.orientation)
            d_orientations = (parent.orientation.copy(), parent.orientation.copy())
            daughters, self.positions = divide_cell_body(
                parent,
                self.positions,
                axis=axis,
                volume_ratio=rule.volume_ratio,
                daughter_names=tuple(rule.daughters),
                daughter_orientations=d_orientations,
            )

            self.cell_bodies.pop(idx)
            self.cell_bodies.extend(daughters)

            self.positions, self.inv_masses = build_particle_arrays(
                self.cell_bodies, self.positions
            )
            self.velocities = np.zeros_like(self.positions)
            self.active_adhesions.clear()

    def step(self):

        # Events
        if self.lineage is not None:
            self._apply_lineage_divisions()
        elif self.step_count in self.division_steps:
            idx = max(
                range(len(self.cell_bodies)),
                key=lambda i: self.cell_bodies[i].target_volume,
            )
            parent = self.cell_bodies[idx]

            daughters, self.positions = divide_cell_body(parent, self.positions)

            # Replace parent with daughters in cell list
            self.cell_bodies.pop(idx)
            self.cell_bodies.extend(daughters)

            # Rebuild flat arrays
            self.positions, self.inv_masses = build_particle_arrays(
                self.cell_bodies, self.positions
            )
            self.velocities = np.zeros_like(self.positions)

            # Clear adhesion pairs (stale indices after division)
            self.active_adhesions.clear()

            # Update polarity
            centers = np.array(
                [self.positions[cb.center_idx] for cb in self.cell_bodies]
            )
            self.cell_polarity = self.polarity.value(centers)

        # Physics
        self.positions, self.velocities = step(
            self.positions,
            self.velocities,
            self.inv_masses,
            self.dt,
            self.cell_bodies,
            self.shell_params,
            active_adhesions=self.active_adhesions,
        )

        centers = np.array([self.positions[cb.center_idx] for cb in self.cell_bodies])
        self.cell_polarity = self.polarity.value(centers)

        self.step_count += 1
        self.time += self.dt

    def run(self, n_steps):

        traj = []

        for _ in range(n_steps):
            self.step()
            traj.append(self.positions.copy())

        return traj
