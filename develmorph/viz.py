import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_cells(positions, radii):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xs, ys, zs = positions.T

    ax.scatter(xs, ys, zs)
    plt.show()
