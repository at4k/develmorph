import numpy as np

from develmorph.engine import World
from develmorph.viz import plot_cells


def main():

    pos = np.array([[0, 0, 0]])
    r = np.array([1.0])

    world = World(pos, r, dt=0.02)

    world.set_divisions([50, 100, 150])

    traj = world.run(300)

    p, r = traj[-1]

    plot_cells(p, r)


if __name__ == "__main__":
    main()
