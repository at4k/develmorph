import numpy as np

from develmorph.solver import simulate
from develmorph.viz import plot_cells


def main():

    pos = np.array([[0, 0, 0]])
    r = np.array([1.0])

    traj = simulate(
        pos,
        r,
        n_steps=300,
        dt=0.02,
        division_steps=[50, 100, 150],
    )

    last_pos, last_r = traj[-1]
    plot_cells(last_pos, last_r)


if __name__ == "__main__":
    main()
