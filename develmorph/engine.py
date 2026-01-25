import numpy as np

from develmorph.solver import step
from develmorph.division import divide_cell


class World:

    def __init__(self, positions, radii, dt=0.01):

        self.positions = positions.copy()
        self.radii = radii.copy()

        self.velocities = np.zeros_like(positions)

        self.dt = dt
        self.time = 0
        self.step_count = 0

        self.division_steps = []

    def set_divisions(self, steps):
        self.division_steps = steps

    def step(self):

        # Events
        if self.step_count in self.division_steps:

            idx = np.argmax(self.radii)

            self.positions, self.radii = divide_cell(self.positions, self.radii, idx)

            self.velocities = np.zeros_like(self.positions)

        # Physics
        self.positions, self.velocities = step(
            self.positions, self.velocities, self.radii, self.dt
        )

        self.step_count += 1
        self.time += self.dt

    def run(self, n_steps):

        traj = []

        for _ in range(n_steps):

            self.step()

            traj.append((self.positions.copy(), self.radii.copy()))

        return traj
