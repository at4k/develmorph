import numpy as np

from develmorph.project import CollisionConstraint, ShellConstraint


def step(
    positions,
    velocities,
    inv_masses,
    radii,
    dt,
    shell_params,
    n_iter=10,
    compliance=0.0,
):
    """
    Advance physical state by one time step using XPBD.

    Args:
        positions: (N, 3) particle positions
        velocities: (N, 3) particle velocities
        inv_masses: (N,) inverse masses
        radii: (N,) particle radii
        dt: time step
        shell_params: (a, b, c) ellipsoid semi-axes
        n_iter: number of constraint iterations
        compliance: constraint compliance (0.0 = rigid)

    Returns:
        (new_positions, new_velocities)
    """
    n = len(positions)
    x_prev = positions.copy()

    # 1. Predict positions (no external forces yet)
    x = positions + dt * velocities

    # 2. Build constraints
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(
                CollisionConstraint(i, j, radii[i] + radii[j], compliance)
            )
        constraints.append(ShellConstraint(i, *shell_params, radii[i], compliance))

    # 3. XPBD iteration
    lambdas = [0.0] * len(constraints)
    for _ in range(n_iter):
        for k, c in enumerate(constraints):
            lambdas[k] = c.project(x, inv_masses, dt, lambdas[k])

    # 4. Update velocities
    new_vel = (x - x_prev) / dt if dt > 0 else np.zeros_like(x)

    return x, new_vel
