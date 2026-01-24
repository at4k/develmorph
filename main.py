import numpy as np
from viz import plot_cells


def main():
    # Initialize one cell at origin
    positions = np.array([[0.0, 0.0, 0.0]])
    radii = np.array([1.0])

    plot_cells(positions, radii)


if __name__ == "__main__":
    main()
