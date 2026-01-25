import matplotlib.pyplot as plt


def make_figure(positions, radii):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    xs, ys, zs = positions.T

    ax.scatter(xs, ys, zs)

    return fig, ax


def plot_cells(positions, radii):

    fig, ax = make_figure(positions, radii)
    plt.show()
