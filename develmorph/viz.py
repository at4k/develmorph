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


def plot_cells_with_polarity(pos, r, p):

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=r * 500, c=p, cmap="coolwarm")

    plt.colorbar(sc)
    plt.show()
