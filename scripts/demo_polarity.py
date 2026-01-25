from develmorph.engine import World
from develmorph.viz import plot_cells_with_polarity

import numpy as np


def main():

    pos = np.array([[0, 0, 0]])
    r = np.array([1.0])

    w = World(pos, r, dt=0.02)

    w.set_divisions([50, 100, 150])

    traj = w.run(300)

    pos, r = traj[-1]

    p = w.cell_polarity

    plot_cells_with_polarity(pos, r, p)


if __name__ == "__main__":
    main()
