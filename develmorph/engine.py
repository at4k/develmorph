import numpy as np

from develmorph.cell_body import build_particle_arrays
from develmorph.division import divide_cell_body
from develmorph.polarity import PolarityField
from develmorph.solver import step


class World:
    def __init__(self, cell_bodies, positions, dt=0.01, shell_params=(5.0, 5.0, 5.0)):

        self.cell_bodies = list(cell_bodies)
        self.positions = positions.copy()

        self.velocities = np.zeros_like(positions)
        self.inv_masses = np.ones(len(positions))

        self.dt = dt
        self.time = 0
        self.step_count = 0

        self.shell_params = shell_params

        self.division_steps = []

        self.polarity = PolarityField()
        centers = np.array([positions[cb.center_idx] for cb in cell_bodies])
        self.cell_polarity = self.polarity.value(centers)

    def set_divisions(self, steps):
        self.division_steps = steps

    def step(self):

        # Events
        if self.step_count in self.division_steps:
            # Divide largest cell (by target volume)
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
