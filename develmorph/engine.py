import numpy as np

from develmorph.division import divide_cell
from develmorph.polarity import PolarityField
from develmorph.solver import step


class World:
    def __init__(self, positions, radii, dt=0.01, shell_params=(5.0, 5.0, 5.0)):

        self.positions = positions.copy()
        self.radii = radii.copy()

        self.velocities = np.zeros_like(positions)
        self.inv_masses = np.ones(len(positions))

        self.dt = dt
        self.time = 0
        self.step_count = 0

        self.shell_params = shell_params

        self.division_steps = []

        self.polarity = PolarityField()
        self.cell_polarity = self.polarity.value(self.positions)

    def set_divisions(self, steps):
        self.division_steps = steps

    def step(self):

        # Events
        if self.step_count in self.division_steps:
            idx = np.argmax(self.radii)

            self.positions, self.radii = divide_cell(self.positions, self.radii, idx)

            parent_p = self.cell_polarity[idx]
            self.cell_polarity = np.delete(self.cell_polarity, idx)
            self.cell_polarity = np.insert(
                self.cell_polarity, idx, [parent_p, parent_p]
            )

            self.velocities = np.zeros_like(self.positions)
            self.inv_masses = np.ones(len(self.positions))

        # Physics
        self.positions, self.velocities = step(
            self.positions,
            self.velocities,
            self.inv_masses,
            self.radii,
            self.dt,
            self.shell_params,
        )

        self.cell_polarity = self.polarity.value(self.positions)

        self.step_count += 1
        self.time += self.dt

    def run(self, n_steps):

        traj = []

        for _ in range(n_steps):
            self.step()

            traj.append((self.positions.copy(), self.radii.copy()))

        return traj
