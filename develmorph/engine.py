from __future__ import annotations

from typing import Optional

import numpy as np

from develmorph.cell_body import build_particle_arrays
from develmorph.celllineage import CellLineage
from develmorph.division import divide_cell_body, divide_physics_cell
from develmorph.polarity import PolarityField
from develmorph.solver import step, step_pc


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
    def __init__(self, cell_bodies, positions, dt=0.01, shell_params=(5.0, 5.0, 5.0), lineage: Optional[CellLineage] = None, n_iter: int = 10, damping: float = 0.0, shell_compliance: float = 0.0, collision_compliance: float = 0.0, membrane_compliance: float = 0.001, volume_compliance: float = 1.0, contact_overlap: float = 0.0, surface_contact_compliance=None, area_compliance=None, tet_volume_compliance=None, attach_margin: float = 0.3, detach_margin: float = 0.8, center_shell_margin_factor: float = 0.0, ):

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
        self.tet_volume_compliance = tet_volume_compliance
        self.attach_margin = attach_margin
        self.detach_margin = detach_margin
        self.center_shell_margin_factor = center_shell_margin_factor

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
            tet_volume_compliance=self.tet_volume_compliance,
            attach_margin=self.attach_margin,
            detach_margin=self.detach_margin,
            center_shell_margin_factor=self.center_shell_margin_factor,
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


class WorldPC:
    """
    Simulation world for PhysicsCell (center-only XPBD).

    Each cell is a single center point with a radius scalar.
    Shell constraint uses offset ellipsoid: effective axes = shell_axes - cell.radius.
    Division is handled via lineage rules; index-shifting is managed internally.
    """

    def __init__(
        self,
        physics_cells,
        positions,
        dt=0.01,
        shell_params=(5.0, 5.0, 5.0),
        lineage=None,
        n_iter=10,
        damping=0.0,
        shell_compliance=0.0,
        collision_compliance=0.0,
        adhesion_compliance=0.1,
        attach_margin=0.3,
        detach_margin=0.8,
        use_area_adhesion=False,
    ):
        self.physics_cells = list(physics_cells)
        self.positions = positions.copy()
        self.velocities = np.zeros_like(positions)
        self.inv_masses = np.ones(len(positions))
        self.dt = dt
        self.shell_params = shell_params
        self.lineage = lineage
        self.n_iter = n_iter
        self.damping = damping
        self.shell_compliance = shell_compliance
        self.collision_compliance = collision_compliance
        self.adhesion_compliance = adhesion_compliance
        self.attach_margin = attach_margin
        self.detach_margin = detach_margin
        self.use_area_adhesion = use_area_adhesion
        self.step_count = 0
        self.active_adhesions = set()

    def step(self):
        """Advance simulation by one time step."""
        if self.lineage is not None:
            self._apply_lineage_divisions()

        self.positions, self.velocities = step_pc(
            self.positions,
            self.velocities,
            self.inv_masses,
            self.dt,
            self.physics_cells,
            self.shell_params,
            n_iter=self.n_iter,
            active_adhesions=self.active_adhesions,
            adhesion_compliance=self.adhesion_compliance,
            attach_margin=self.attach_margin,
            detach_margin=self.detach_margin,
            damping=self.damping,
            shell_compliance=self.shell_compliance,
            collision_compliance=self.collision_compliance,
            use_area_adhesion=self.use_area_adhesion,
        )
        self.step_count += 1

    def run(self, n_steps):
        """Run n_steps and return list of position snapshots."""
        traj = []
        for _ in range(n_steps):
            self.step()
            traj.append(self.positions.copy())
        return traj

    def _apply_lineage_divisions(self):
        """Apply all division rules scheduled for the current step_count."""
        program = self.lineage.get_step_program(self.step_count)
        if program is None:
            return

        for rule in program.divisions:
            idx = next(
                (i for i, pc in enumerate(self.physics_cells) if pc.name == rule.parent),
                None,
            )
            if idx is None:
                continue

            parent = self.physics_cells[idx]
            parent_center_idx = parent.center_idx
            axis = self.lineage.resolve_division_axis(rule, parent.orientation)

            daughters, self.positions = divide_physics_cell(
                parent,
                self.positions,
                axis=axis,
                volume_ratio=rule.volume_ratio,
                daughter_names=tuple(rule.daughters),
                daughter_orientations=(
                    parent.orientation.copy(),
                    parent.orientation.copy(),
                ),
            )

            # Shift center_idx of all surviving cells whose index was above parent
            for pc in self.physics_cells:
                if pc is not parent and pc.center_idx > parent_center_idx:
                    pc.center_idx -= 1

            self.physics_cells.pop(idx)
            self.physics_cells.extend(daughters)
            self.velocities = np.zeros_like(self.positions)
            self.inv_masses = np.ones(len(self.positions))
            self.active_adhesions.clear()
