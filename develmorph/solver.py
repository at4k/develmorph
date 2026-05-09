import numpy as np

from develmorph.project import (
    AdhesionConstraint,
    CollisionConstraint,
    DistanceConstraint,
    HalfSpaceConstraint,
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
        ri = (3.0 * ci.target_volume / 4.0) ** (1.0 / 3.0)
        rj = (3.0 * cj.target_volume / 4.0) ** (1.0 / 3.0)
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
            ri = (3.0 * ci.target_volume / 4.0) ** (1.0 / 3.0)
            rj = (3.0 * cj.target_volume / 4.0) ** (1.0 / 3.0)
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
    collision_compliance=0.0,
    contact_overlap=0.0,
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

        # Distance constraints: center-surface
        # Stiffer than surface-surface so the center tracks the cell, but soft
        # enough (alpha_tilde = 10) that SurfacePenetrationConstraint
        # (alpha_tilde = 0, 12x stronger per iter) can overcome it and produce
        # visible contact-zone flattening.
        center_compliance = 0.001
        for (ci, si), rest_len in zip(cb.center_surface_edges, cb.center_surface_rest):
            distance_constraints.append(
                DistanceConstraint(ci, si, rest_len, center_compliance)
            )

        # Distance constraints: surface-surface (deformable membrane)
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

    # Collision constraints: center-to-center repulsion.
    # When contact_overlap > 0, equilibrium distance is shrunk so centers overlap
    # slightly; HalfSpaceConstraints then prevent surface particles from crossing
    # into the opposing cell's half-space, producing direction-unbiased contact
    # flattening and volume redistribution into the free side.
    equil_dist = 1.0 - contact_overlap
    for i in range(len(cell_bodies)):
        for j in range(i + 1, len(cell_bodies)):
            ci = cell_bodies[i]
            cj = cell_bodies[j]
            ri = (3.0 * ci.target_volume / 4.0) ** (1.0 / 3.0)
            rj = (3.0 * cj.target_volume / 4.0) ** (1.0 / 3.0)
            collision_constraints.append(
                CollisionConstraint(
                    ci.center_idx, cj.center_idx,
                    (ri + rj) * equil_dist, collision_compliance,
                )
            )

    # Surface contact constraints.
    # contact_overlap > 0: HalfSpaceConstraints — every surface particle of cell i
    # is constrained to stay on cell i's side of the contact plane with cell j.
    # Force direction is always along the contact normal, removing directional
    # bias that appeared with the nearest-pair CollisionConstraint approach.
    # contact_overlap == 0 (default/legacy): nearest-pair CollisionConstraint.
    surface_penetration_constraints = []
    if contact_overlap > 0.0:
        for i in range(len(cell_bodies)):
            for j in range(len(cell_bodies)):
                if i == j:
                    continue
                ci = cell_bodies[i]
                cj = cell_bodies[j]
                ri = (3.0 * ci.target_volume / 4.0) ** (1.0 / 3.0)
                rj = (3.0 * cj.target_volume / 4.0) ** (1.0 / 3.0)
                center_dist = np.linalg.norm(
                    positions[ci.center_idx] - positions[cj.center_idx]
                )
                if center_dist > ri + rj + 0.3 or center_dist < 1e-9:
                    continue
                n_vec = (
                    positions[cj.center_idx] - positions[ci.center_idx]
                ) / center_dist
                M = (positions[ci.center_idx] + positions[cj.center_idx]) / 2.0
                for si in ci.surface_indices:
                    surface_penetration_constraints.append(
                        HalfSpaceConstraint(si, M, n_vec, compliance=0.0)
                    )
    else:
        for i in range(len(cell_bodies)):
            for j in range(i + 1, len(cell_bodies)):
                ci = cell_bodies[i]
                cj = cell_bodies[j]
                ri = (3.0 * ci.target_volume / 4.0) ** (1.0 / 3.0)
                rj = (3.0 * cj.target_volume / 4.0) ** (1.0 / 3.0)
                center_dist = np.linalg.norm(
                    positions[ci.center_idx] - positions[cj.center_idx]
                )
                if center_dist > ri + rj + 0.3:
                    continue
                si_c = min(
                    ci.surface_indices,
                    key=lambda s: np.linalg.norm(positions[s] - positions[cj.center_idx]),
                )
                sj_c = min(
                    cj.surface_indices,
                    key=lambda s: np.linalg.norm(positions[s] - positions[ci.center_idx]),
                )
                contact_r = 0.05 * (ri + rj)
                surface_penetration_constraints.append(
                    CollisionConstraint(si_c, sj_c, contact_r, collision_compliance)
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
            ri = (3.0 * ci.target_volume / 4.0) ** (1.0 / 3.0)
            rj = (3.0 * cj.target_volume / 4.0) ** (1.0 / 3.0)
            adhesion_constraints.append(
                AdhesionConstraint(
                    ci.center_idx, cj.center_idx,
                    (ri + rj) * equil_dist, adhesion_compliance,
                )
            )

    # Ordered: volume -> distance -> adhesion -> collision -> surface_penetration -> shell
    # Volume first, then distance restores membrane shape; surface penetration after
    # collision so center positions are settled before surface points are corrected.
    all_constraints = (
        volume_constraints
        + distance_constraints
        + adhesion_constraints
        + collision_constraints
        + surface_penetration_constraints
        + shell_constraints
    )

    # 3. XPBD iteration
    lambdas = [0.0] * len(all_constraints)
    for _ in range(n_iter):
        for k, c in enumerate(all_constraints):
            lambdas[k] = c.project(x, inv_masses, dt, lambdas[k])

    # 3b. Inversion guard: if the signed mesh volume goes negative the center
    # has escaped outside the surface hull, reversing all restoring forces.
    # Snap it to the surface centroid so the next step starts from a valid state.
    for cb in cell_bodies:
        tris = cb.surface_global_triangles()
        vol = sum(np.dot(x[i0], np.cross(x[i1], x[i2])) for i0, i1, i2 in tris) / 6.0
        if vol <= 0.0:
            x[cb.center_idx] = x[np.array(cb.surface_indices)].mean(axis=0)

    # 4. Update velocities (overdamped for biological cells: damping=1.0 → vel=0)
    new_vel = (x - x_prev) / dt * (1.0 - damping) if dt > 0 else np.zeros_like(x)

    return x, new_vel
