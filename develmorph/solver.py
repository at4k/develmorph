import numpy as np

from develmorph.project import (
    AdhesionConstraint,
    CollisionConstraint,
    DistanceConstraint,
    ShellConstraint,
    VolumeConstraint,
)


def _max_surface_dist(positions, cb):
    """Max distance from center to any surface point."""
    center = positions[cb.center_idx]
    return max(np.linalg.norm(positions[si] - center) for si in cb.surface_indices)


def _update_adhesion_pairs(
    positions, cell_bodies, active_adhesions, attach_margin, detach_margin
):
    """
    Update the set of active adhesion pairs using hysteresis logic.

    - Remove pairs whose cells exceed the detach threshold or have stale indices.
    - Add new pairs whose cells are within the attach threshold.

    Uses cell_bodies list indices (not particle indices) for pair identification.
    """
    n = len(cell_bodies)

    # Phase 1: Remove stale or detached pairs
    to_remove = set()
    for i, j in active_adhesions:
        # Stale index check
        if i >= n or j >= n:
            to_remove.add((i, j))
            continue
        ci = cell_bodies[i]
        cj = cell_bodies[j]
        ri = _max_surface_dist(positions, ci)
        rj = _max_surface_dist(positions, cj)
        dist = np.linalg.norm(positions[ci.center_idx] - positions[cj.center_idx])
        if dist > ri + rj + detach_margin:
            to_remove.add((i, j))
    active_adhesions -= to_remove

    # Phase 2: Attach new pairs within attach threshold
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in active_adhesions:
                continue
            ci = cell_bodies[i]
            cj = cell_bodies[j]
            ri = _max_surface_dist(positions, ci)
            rj = _max_surface_dist(positions, cj)
            dist = np.linalg.norm(positions[ci.center_idx] - positions[cj.center_idx])
            if dist < ri + rj + attach_margin:
                active_adhesions.add((i, j))


def step(
    positions,
    velocities,
    inv_masses,
    dt,
    cell_bodies,
    shell_params,
    n_iter=10,
    membrane_compliance=0.001,
    volume_compliance=1.0,
    active_adhesions=None,
    adhesion_compliance=0.1,
    attach_margin=0.3,
    detach_margin=0.8,
    damping=0.0,
    shell_compliance=0.0,
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
        active_adhesions: mutable set of (i,j) cell index pairs, or None for
            legacy CollisionConstraint behavior
        adhesion_compliance: compliance for adhesion constraints
        attach_margin: distance margin above ri+rj for forming new adhesions
        detach_margin: distance margin above ri+rj for breaking adhesions
        damping: velocity damping factor [0.0, 1.0]. 0.0 = fully inertial,
            1.0 = fully overdamped (zero velocity carry-over between steps).
            Use 1.0 for biological cells (low Reynolds number / viscosity-dominated).

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

        # Distance constraints: center-surface (rest lengths from CellBody)
        for (ci, si), rest_len in zip(cb.center_surface_edges, cb.center_surface_rest):
            distance_constraints.append(
                DistanceConstraint(ci, si, rest_len, membrane_compliance)
            )

        # Distance constraints: surface-surface (rest lengths from CellBody)
        for (a, b), rest_len in zip(cb.surface_surface_edges, cb.surface_surface_rest):
            distance_constraints.append(
                DistanceConstraint(a, b, rest_len, membrane_compliance)
            )

        # Shell constraints: center + surface points.
        # Use nominal radius from target_volume to avoid instability when
        # membrane deformation temporarily inflates _max_surface_dist.
        nominal_r = (3.0 * cb.target_volume / 4.0) ** (1.0 / 3.0)
        shell_constraints.append(
            ShellConstraint(cb.center_idx, *shell_params, nominal_r, shell_compliance)
        )
        for si in cb.surface_indices:
            shell_constraints.append(ShellConstraint(si, *shell_params, 0.0, shell_compliance))

    # Collision constraints: always applied for all cell pairs (repulsion)
    for i in range(len(cell_bodies)):
        for j in range(i + 1, len(cell_bodies)):
            ci = cell_bodies[i]
            cj = cell_bodies[j]
            ri = _max_surface_dist(positions, ci)
            rj = _max_surface_dist(positions, cj)
            collision_constraints.append(
                CollisionConstraint(ci.center_idx, cj.center_idx, ri + rj, 0.0)
            )

    # Adhesion constraints: attraction for active pairs only
    adhesion_constraints = []
    if active_adhesions is not None:
        _update_adhesion_pairs(
            positions, cell_bodies, active_adhesions, attach_margin, detach_margin
        )
        for i, j in active_adhesions:
            ci = cell_bodies[i]
            cj = cell_bodies[j]
            ri = _max_surface_dist(positions, ci)
            rj = _max_surface_dist(positions, cj)
            adhesion_constraints.append(
                AdhesionConstraint(
                    ci.center_idx, cj.center_idx, ri + rj, adhesion_compliance
                )
            )

    # Ordered: volume -> distance -> adhesion/collision -> shell
    # Volume first, then distance restores membrane shape
    all_constraints = (
        volume_constraints
        + distance_constraints
        + adhesion_constraints
        + collision_constraints
        + shell_constraints
    )

    # 3. XPBD iteration
    lambdas = [0.0] * len(all_constraints)
    for _ in range(n_iter):
        for k, c in enumerate(all_constraints):
            lambdas[k] = c.project(x, inv_masses, dt, lambdas[k])

    # 4. Update velocities (overdamped for biological cells: damping=1.0 → vel=0)
    new_vel = (x - x_prev) / dt * (1.0 - damping) if dt > 0 else np.zeros_like(x)

    return x, new_vel
