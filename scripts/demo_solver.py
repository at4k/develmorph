import numpy as np

from develmorph.solver import simulate
from develmorph.viz import plot_cells


def main():

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
        ]
    )

    radii = np.array([0.5, 0.5])

    traj = simulate(
        positions,
        radii,
        n_steps=300,
        dt=0.02,
    )

    plot_cells(traj[-1], radii)


if __name__ == "__main__":
    main()
