def test_plot_runs_without_error():
    import numpy as np
    from viz import plot_cells

    positions = np.zeros((1, 3))
    radii = np.ones(1)
    plot_cells(positions, radii)
    assert True
