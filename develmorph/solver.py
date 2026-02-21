import numpy as np

from develmorph.project import (
    CollisionConstraint,
    DistanceConstraint,
    ShellConstraint,
    VolumeConstraint,
)


def _max_surface_dist(positions, cb):
    """Max distance from center to any surface point."""
    center = positions[cb.center_idx]
    return max(np.linalg.norm(positions[si] - center) for si in cb.surface_indices)


def step(
    positions,
    velocities,
    inv_masses,
    dt,
    cell_bodies,
    shell_params,
    n_iter=10,
    membrane_compliance=0.01,
    volume_compliance=0.0,
):
    """
    Advance physical state by one time step using XPBD.

    Args:
        positions: (N, 3) particle positions
        velocities: (N, 3) particle velocities
        inv_masses: (N,) inverse masses
        dt: time step
        cell_bodies: list of CellBody
        shell_params: (a, b, c) ellipsoid semi-axes
        n_iter: number of constraint iterations
        membrane_compliance: compliance for membrane distance constraints
        volume_compliance: compliance for volume constraints (0.0 = incompressible)

    Returns:
        (new_positions, new_velocities)
    """
    x_prev = positions.copy()

    # 1. Predict positions
    x = positions + dt * velocities

    # 2. Build constraints in priority order
    volume_constraints = []
    distance_constraints = []
    collision_constraints = []
    shell_constraints = []

    for cb in cell_bodies:
        # Volume constraint (per cell)
        volume_constraints.append(
            VolumeConstraint(
                center_idx=cb.center_idx,
                surface_indices=cb.surface_indices,
                triangles=cb.surface_global_triangles(),
                target_volume=cb.target_volume,
                compliance=volume_compliance,
            )
        )

        # Distance constraints: center-surface
        for si in cb.surface_indices:
            rest_len = np.linalg.norm(positions[si] - positions[cb.center_idx])
            distance_constraints.append(
                DistanceConstraint(cb.center_idx, si, rest_len, membrane_compliance)
            )

        # Distance constraints: surface-surface (octahedron edges)
        for a, b in cb.surface_surface_edges:
            rest_len = np.linalg.norm(positions[a] - positions[b])
            distance_constraints.append(
                DistanceConstraint(a, b, rest_len, membrane_compliance)
            )

        # Shell constraints: center + surface points
        max_r = _max_surface_dist(positions, cb)
        shell_constraints.append(
            ShellConstraint(cb.center_idx, *shell_params, max_r, 0.0)
        )
        for si in cb.surface_indices:
            shell_constraints.append(ShellConstraint(si, *shell_params, 0.0, 0.0))

    # Collision constraints: center-to-center between cells
    for i in range(len(cell_bodies)):
        for j in range(i + 1, len(cell_bodies)):
            ci = cell_bodies[i]
            cj = cell_bodies[j]
            ri = _max_surface_dist(positions, ci)
            rj = _max_surface_dist(positions, cj)
            collision_constraints.append(
                CollisionConstraint(ci.center_idx, cj.center_idx, ri + rj, 0.0)
            )

    # Ordered: volume first (highest priority), then distance, collision, shell
    all_constraints = (
        volume_constraints
        + distance_constraints
        + collision_constraints
        + shell_constraints
    )

    # 3. XPBD iteration
    lambdas = [0.0] * len(all_constraints)
    for _ in range(n_iter):
        for k, c in enumerate(all_constraints):
            lambdas[k] = c.project(x, inv_masses, dt, lambdas[k])

    # 4. Update velocities
    new_vel = (x - x_prev) / dt if dt > 0 else np.zeros_like(x)

    return x, new_vel
