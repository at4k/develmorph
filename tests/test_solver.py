import numpy as np
from develmorph.solver import step


def test_solver_separates_cells():

    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ]
    )

    vel = np.zeros_like(pos)
    radii = np.array([0.5, 0.5])

    new_pos = step(positions=pos, velocities=vel, radii=radii, dt=0.1)

    dist0 = np.linalg.norm(pos[0] - pos[1])
    dist1 = np.linalg.norm(new_pos[0] - new_pos[1])

    assert dist1 > dist0
