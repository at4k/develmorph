# File: develmorph/viz.py
# Phase2-A: 3D visualization of developmental time series

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


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


def visualize_lineage_3d(df: pd.DataFrame):
    """
    Visualize lineage-driven cell trajectories in 3D.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of LineageRecorder.to_dataframe().
        Required columns: step, cell_id, lineage_id, x, y, z
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for cell_id, group in df.groupby("cell_id"):
        ax.plot(
            group["x"],
            group["y"],
            group["z"],
            marker="o",
            label=f"cell {cell_id}",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Phase2-A Lineage-driven Cell Division")

    plt.legend()
    plt.show()
