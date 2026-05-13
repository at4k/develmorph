from __future__ import annotations

from typing import Optional

import numpy as np

from develmorph.cell_body import build_particle_arrays
from develmorph.celllineage import CellLineage
from develmorph.division import divide_cell_body
from develmorph.polarity import PolarityField
from develmorph.solver import step


def _shift_cell_indices(cell_bodies, deleted_indices, exclude=None):
    """
    Update particle indices in cell_bodies after np.delete removed some particles.

    divide_cell_body calls np.delete which shifts all remaining indices.
    Any surviving cell whose indices fall above deleted positions must be
    decremented to match the new compacted array.
    """
    for cb in cell_bodies:
        if cb is exclude:
            continue

        def _shift(idx, deleted=deleted_indices):
            return idx - sum(1 for pi in deleted if pi < idx)

        cb.center_idx = _shift(cb.center_idx)
        cb.surface_indices = [_shift(si) for si in cb.surface_indices]
        cb.center_surface_edges = [(cb.center_idx, si) for si in cb.surface_indices]
        cb.surface_surface_edges = [
            (cb.surface_indices[a], cb.surface_indices[b]) for a, b in cb.edges
        ]


class World:
    def __init__(self, cell_bodies, positions, dt=0.01, shell_params=(5.0, 5.0, 5.0), lineage: Optional[CellLineage] = None, n_iter: int = 10, damping: float = 0.0, shell_compliance: float = 0.0, collision_compliance: float = 0.0, membrane_compliance: float = 0.001, volume_compliance: float = 1.0, contact_overlap: float = 0.0, surface_contact_compliance=None, area_compliance=None):

        self.cell_bodies = list(cell_bodies)
        self.positions = positions.copy()

        self.velocities = np.zeros_like(positions)
        self.inv_masses = np.ones(len(positions))

        self.dt = dt
        self.time = 0
        self.step_count = 0

        self.shell_params = shell_params
        self.lineage = lineage
        self.n_iter = n_iter
        self.damping = damping
        self.shell_compliance = shell_compliance
        self.collision_compliance = collision_compliance
        self.membrane_compliance = membrane_compliance
        self.volume_compliance = volume_compliance
        self.contact_overlap = contact_overlap
        self.surface_contact_compliance = surface_contact_compliance
        self.area_compliance = area_compliance

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
            parent_all_indices = sorted(parent.all_indices())

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

            # np.delete inside divide_cell_body shifts all surviving particle
            # indices downward; fix them before build_particle_arrays reads them.
            _shift_cell_indices(self.cell_bodies, parent_all_indices, exclude=parent)

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
            parent_all_indices = sorted(parent.all_indices())

            daughters, self.positions = divide_cell_body(parent, self.positions)

            _shift_cell_indices(self.cell_bodies, parent_all_indices, exclude=parent)

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
            n_iter=self.n_iter,
            active_adhesions=self.active_adhesions,
            damping=self.damping,
            shell_compliance=self.shell_compliance,
            collision_compliance=self.collision_compliance,
            membrane_compliance=self.membrane_compliance,
            volume_compliance=self.volume_compliance,
            contact_overlap=self.contact_overlap,
            surface_contact_compliance=self.surface_contact_compliance,
            area_compliance=self.area_compliance,
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
