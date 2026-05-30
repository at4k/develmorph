import math

import numpy as np

from develmorph.power_diagram import compute_voronoi_cells
from develmorph.project import (
    AdhesionConstraint,
    CollisionConstraint,
    DistanceConstraint,
    HalfSpaceConstraint,
    ShellConstraint,
    TetrahedralVolumeConstraint,
    TriangleAreaConstraint,
    VolumeConstraint,
    VoronoiVolumeConstraint,
)
from develmorph.spatial_hash import SpatialHash


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
        ri = ci.nominal_radius()
        rj = cj.nominal_radius()
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
            ri = ci.nominal_radius()
            rj = cj.nominal_radius()
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
    surface_contact_compliance=None,
    area_compliance=None,
    tet_volume_compliance=None,
    center_shell_margin_factor: float = 0.0,
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
        area_compliance: compliance for per-triangle stretch prevention. None (default)
            disables it; when set, applies TriangleAreaConstraint to each face.
        tet_volume_compliance: compliance for per-tetrahedron volume constraints.
            When set, replaces VolumeConstraint + DistanceConstraint +
            TriangleAreaConstraint with one TetrahedralVolumeConstraint per
            surface triangle. None (default) preserves the old path.
        center_shell_margin_factor: fraction of nominal_radius used as shell margin for
            center particles. 0.0 (default) = no margin (center can touch shell boundary).
            Values > 0 prevent centers from reaching the boundary, stopping surface
            particles on the shell side from being geometrically impossible to place.
    Returns:
        (new_positions, new_velocities)
    """
    x_prev = positions.copy()

    # 1. Predict positions
    x = positions + dt * velocities

    # 2. Build constraints in priority order
    # New path: per-tetrahedron volume constraints (tet_volume_compliance is not None)
    # Old path: VolumeConstraint + DistanceConstraint + TriangleAreaConstraint
    tet_constraints = []
    volume_constraints = []
    distance_constraints = []
    area_constraints = []

    if tet_volume_compliance is not None:
        for cb in cell_bodies:
            for (a, b, c), v_target in zip(cb.triangles, cb.face_volumes):
                i0 = cb.surface_indices[a]
                i1 = cb.surface_indices[b]
                i2 = cb.surface_indices[c]
                tet_constraints.append(
                    TetrahedralVolumeConstraint(
                        cb.center_idx, i0, i1, i2, v_target, tet_volume_compliance
                    )
                )
    else:
        for cb in cell_bodies:
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

        # Area constraints: per-triangle stretch prevention.
        # None = disabled (backward compatible). Fires only when A > A_target.
        if area_compliance is not None:
            for cb in cell_bodies:
                for (a, b, c), target_a in zip(cb.triangles, cb.face_areas):
                    i0 = cb.surface_indices[a]
                    i1 = cb.surface_indices[b]
                    i2 = cb.surface_indices[c]
                    area_constraints.append(
                        TriangleAreaConstraint(i0, i1, i2, target_a, area_compliance)
                    )

    # Shell constraints: center + surface points.
    # Surface particles always use margin=0 (can touch shell boundary).
    # Center uses margin = nominal_r * center_shell_margin_factor.
    # factor=0 (default): center can reach shell boundary — matches legacy behavior.
    # factor>0: keeps center at least factor*r from boundary in the ellipsoidal
    # sense, preventing surface particles on the shell side from being crushed
    # (they would need to be at rest_length from center but beyond the shell).
    # NOTE: large factors (>0.5) constrain cells in non-AP directions too, which
    # can limit dorsal/ventral displacement in packed diamond configurations.
    collision_constraints = []
    shell_constraints = []
    for cb in cell_bodies:
        nominal_r = cb.nominal_radius()
        center_margin = nominal_r * center_shell_margin_factor
        shell_constraints.append(
            ShellConstraint(cb.center_idx, *shell_params, center_margin, shell_compliance)
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
            ri = ci.nominal_radius()
            rj = cj.nominal_radius()
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
                ri = ci.nominal_radius()
                rj = cj.nominal_radius()
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
                ri = ci.nominal_radius()
                rj = cj.nominal_radius()
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
            ri = ci.nominal_radius()
            rj = cj.nominal_radius()
            adhesion_constraints.append(
                AdhesionConstraint(
                    ci.center_idx, cj.center_idx,
                    (ri + rj) * equil_dist, adhesion_compliance,
                )
            )

    # Surface contact adhesion: pull nearby surface particles of adjacent cells
    # toward each other. Applied after HalfSpaceConstraints so particles are
    # attracted to the contact plane without crossing it.
    surface_contact_constraints = []
    if surface_contact_compliance is not None:
        for i in range(len(cell_bodies)):
            for j in range(i + 1, len(cell_bodies)):
                ci = cell_bodies[i]
                cj = cell_bodies[j]
                ri = ci.nominal_radius()
                rj = cj.nominal_radius()
                center_dist = np.linalg.norm(
                    positions[ci.center_idx] - positions[cj.center_idx]
                )
                if center_dist > ri + rj + 0.3 or center_dist < 1e-9:
                    continue
                cutoff = (ri + rj) * 0.5
                for si in ci.surface_indices:
                    for sj in cj.surface_indices:
                        d = np.linalg.norm(positions[si] - positions[sj])
                        if d < cutoff:
                            surface_contact_constraints.append(
                                AdhesionConstraint(si, sj, 0.0, surface_contact_compliance)
                            )

    # Ordered: tet -> volume -> distance -> area -> adhesion -> collision
    #          -> surface_penetration -> surface_contact -> shell
    # tet and (volume+distance+area) are mutually exclusive (one set will be empty).
    # surface_contact after HalfSpace: repulsion is settled before attraction fires.
    all_constraints = (
        tet_constraints
        + volume_constraints
        + distance_constraints
        + area_constraints
        + adhesion_constraints
        + collision_constraints
        + surface_penetration_constraints
        + surface_contact_constraints
        + shell_constraints
    )

    # 3. XPBD iteration
    lambdas = [0.0] * len(all_constraints)
    for _ in range(n_iter):
        for k, c in enumerate(all_constraints):
            lambdas[k] = c.project(x, inv_masses, dt, lambdas[k])

    # 3b. Inversion guard: if the center has escaped outside the surface hull
    # the center-relative tetrahedra volume goes non-positive, reversing all
    # restoring forces.  Snap center to surface centroid so the next step
    # starts from a valid state.
    # Uses the center-relative formula (a = pi - pc) to detect hull escape.
    # The origin-based formula dot(pi, cross(pj, pk)) stays positive even when
    # the center is outside, so it cannot detect this condition.
    for cb in cell_bodies:
        pc = x[cb.center_idx]
        tris = cb.surface_global_triangles()
        vol = sum(
            np.dot(x[i0] - pc, np.cross(x[i1] - pc, x[i2] - pc))
            for i0, i1, i2 in tris
        ) / 6.0
        if vol <= 0.0:
            x[cb.center_idx] = x[np.array(cb.surface_indices)].mean(axis=0)

    # 4. Update velocities (overdamped for biological cells: damping=1.0 → vel=0)
    new_vel = (x - x_prev) / dt * (1.0 - damping) if dt > 0 else np.zeros_like(x)

    return x, new_vel


def _update_adhesion_pairs_pc(positions, physics_cells, active_adhesions,
                               attach_margin, detach_margin):
    """
    Update active adhesion pairs for PhysicsCell list using hysteresis.

    Uses cell list indices (not particle indices) for pair identification.
    Attaches pairs within attach_margin of contact; detaches beyond detach_margin.
    """
    n = len(physics_cells)

    to_remove = set()
    for i, j in active_adhesions:
        if i >= n or j >= n:
            to_remove.add((i, j))
            continue
        ci, cj = physics_cells[i], physics_cells[j]
        dist = np.linalg.norm(positions[ci.center_idx] - positions[cj.center_idx])
        if dist > ci.radius + cj.radius + detach_margin:
            to_remove.add((i, j))
    active_adhesions -= to_remove

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in active_adhesions:
                continue
            ci, cj = physics_cells[i], physics_cells[j]
            dist = np.linalg.norm(positions[ci.center_idx] - positions[cj.center_idx])
            if dist < ci.radius + cj.radius + attach_margin:
                active_adhesions.add((i, j))


def step_pc(
    positions,
    velocities,
    inv_masses,
    dt,
    physics_cells,
    shell_params,
    n_iter=10,
    active_adhesions=None,
    adhesion_compliance=0.1,
    attach_margin=0.3,
    detach_margin=0.8,
    damping=0.0,
    shell_compliance=0.0,
    collision_compliance=0.0,
    use_area_adhesion=False,
    use_spatial_hash=False,
    use_voronoi: bool = False,
    voronoi_volume_compliance: float = 0.01,
    voronoi_n_ghost: int = 50,
):
    """
    Advance PhysicsCell state by one XPBD time step.

    Each cell is a single center particle. Constraints applied in priority order:
      adhesion -> collision -> shell

    Shell constraint uses ShellConstraint(i, a, b, c, radius=pc.radius) which
    constrains the center to stay within effective ellipsoid (a-r, b-r, c-r).
    This is the key fix vs. the old center-only model: cells near ellipsoid
    tips get correctly confined based on their physical radius.

    Args:
        positions: (N, 3) center positions (N = number of cells)
        velocities: (N, 3) velocities
        inv_masses: (N,) inverse masses (typically all 1.0)
        dt: time step
        physics_cells: list of PhysicsCell
        shell_params: (a, b, c) ellipsoid semi-axes
        n_iter: XPBD constraint iterations per step
        active_adhesions: mutable set of (i, j) cell-index pairs; None = no adhesion
        adhesion_compliance: compliance for adhesion constraints
        attach_margin: extra distance above r_i+r_j to form adhesion
        detach_margin: extra distance above r_i+r_j to break adhesion
        damping: velocity damping [0=inertial, 1=overdamped/biological]
        shell_compliance: compliance for shell constraint (0=rigid)
        collision_compliance: compliance for collision constraint (0=rigid)
        use_voronoi: if True, use Voronoi tessellation mode (Phase 5); replaces
            CollisionConstraint with VoronoiVolumeConstraint and sets shell
            radius offset to 0 so centers are constrained to the raw ellipsoid
        voronoi_volume_compliance: XPBD compliance for VoronoiVolumeConstraint
            (0.01 soft; increase if oscillations occur; frozen Voronoi is stable
            at n_iter=1, may diverge at n_iter>3)
        voronoi_n_ghost: number of ghost points for shell boundary in power diagram

    Returns:
        (new_positions, new_velocities)
    """
    x_prev = positions.copy()
    x = positions + dt * velocities

    if use_voronoi:
        a, b, c_ax = shell_params

        # Compute Voronoi tessellation from predicted positions (frozen for this step)
        voronoi_cells = compute_voronoi_cells(
            physics_cells, x, shell_params, n_ghost=voronoi_n_ghost
        )

        # Shell: center inside raw ellipsoid (no radius offset)
        shell_constraints = [
            ShellConstraint(pc.center_idx, a, b, c_ax, 0.0, shell_compliance)
            for pc in physics_cells
        ]

        # Volume: Voronoi region volume = target sphere volume
        volume_constraints = []
        for i, pc in enumerate(physics_cells):
            target_v = (4.0 / 3.0) * math.pi * pc.radius ** 3
            con = VoronoiVolumeConstraint(pc.center_idx, target_v, voronoi_volume_compliance)
            con.set_voronoi_cell(voronoi_cells[i])
            volume_constraints.append(con)

        # Adhesion: Voronoi contact-area-weighted
        adhesion_constraints = []
        if active_adhesions is not None:
            _update_adhesion_pairs_pc(
                x, physics_cells, active_adhesions, attach_margin, detach_margin
            )
            for i, j in active_adhesions:
                ci, cj = physics_cells[i], physics_cells[j]
                area = voronoi_cells[i].contact_areas.get(j, 0.0)
                effective_compliance = adhesion_compliance / (area + 1e-6)
                adhesion_constraints.append(
                    AdhesionConstraint(
                        ci.center_idx, cj.center_idx,
                        ci.radius + cj.radius,
                        effective_compliance,
                    )
                )

        all_constraints = adhesion_constraints + volume_constraints + shell_constraints
        lambdas = [0.0] * len(all_constraints)
        for _ in range(n_iter):
            for k, con in enumerate(all_constraints):
                lambdas[k] = con.project(x, inv_masses, dt, lambdas[k])

        new_vel = (x - x_prev) / dt * (1.0 - damping) if dt > 0 else np.zeros_like(x)
        return x, new_vel

    a, b, c_ax = shell_params

    # Shell: each center must fit its sphere inside the ellipsoid
    shell_constraints = [
        ShellConstraint(pc.center_idx, a, b, c_ax, pc.radius, shell_compliance)
        for pc in physics_cells
    ]

    # Collision: center-to-center >= r_i + r_j (repulsion)
    collision_constraints = []
    # Brute-force O(N^2) is cheaper for small N; spatial hash pays off above this threshold.
    _SPATIAL_HASH_MIN_CELLS = 16
    if use_spatial_hash and len(physics_cells) > _SPATIAL_HASH_MIN_CELLS:
        max_radius = max(pc.radius for pc in physics_cells)
        sh = SpatialHash(cell_size=2.0 * max_radius * 1.1)
        sh.build(physics_cells, x)
        candidate_pairs = sh.candidate_pairs()
    else:
        n = len(physics_cells)
        candidate_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}

    for i, j in candidate_pairs:
        ci, cj = physics_cells[i], physics_cells[j]
        collision_constraints.append(
            CollisionConstraint(
                ci.center_idx, cj.center_idx,
                ci.radius + cj.radius,
                collision_compliance,
            )
        )

    # Adhesion: attraction for active pairs only
    adhesion_constraints = []
    if active_adhesions is not None:
        _update_adhesion_pairs_pc(
            x, physics_cells, active_adhesions, attach_margin, detach_margin
        )
        for i, j in active_adhesions:
            ci, cj = physics_cells[i], physics_cells[j]
            rest = ci.radius + cj.radius

            if use_area_adhesion:
                dist = np.linalg.norm(x[ci.center_idx] - x[cj.center_idx])
                overlap = max(0.0, rest - dist)
                r_harm = 2.0 * ci.radius * cj.radius / (ci.radius + cj.radius)
                area = math.pi * overlap * r_harm
                effective_compliance = adhesion_compliance / (area + 1e-6)
            else:
                effective_compliance = adhesion_compliance

            adhesion_constraints.append(
                AdhesionConstraint(
                    ci.center_idx, cj.center_idx,
                    rest,
                    effective_compliance,
                )
            )

    # Priority: adhesion -> collision -> shell
    all_constraints = adhesion_constraints + collision_constraints + shell_constraints
    lambdas = [0.0] * len(all_constraints)

    for _ in range(n_iter):
        for k, con in enumerate(all_constraints):
            lambdas[k] = con.project(x, inv_masses, dt, lambdas[k])

    new_vel = (x - x_prev) / dt * (1.0 - damping) if dt > 0 else np.zeros_like(x)
    return x, new_vel
