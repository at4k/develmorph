# Tetrahedral Volume Constraint — Design Spec

Date: 2026-05-15
Branch: feat-xpbd-solver
Status: Approved

## Overview

Replace the current membrane model (DistanceConstraint + TriangleAreaConstraint +
global VolumeConstraint) with per-tetrahedron volume constraints. Each surface
triangle of a CellBody, together with the cell center point, forms a tetrahedron
whose volume is independently preserved by a new `TetrahedralVolumeConstraint`.

This produces a more physically motivated model: cell-internal pressure acts on
each membrane patch locally, rather than a spring network maintaining rest lengths.

## Motivation

Current membrane model:
- DistanceConstraint (center-surface, surface-surface): spring network for shape
- TriangleAreaConstraint: inequality stretch prevention
- VolumeConstraint: global divergence-theorem volume preservation

Problems with this approach:
- Three overlapping mechanisms with competing compliances are hard to tune
- DistanceConstraint rest lengths encode shape rigidly; deformation is penalized
  even when biologically desirable (e.g., cell flattening at contact)
- Global VolumeConstraint corrects all surface points simultaneously — poor
  locality when only one side of the cell deforms

## Core Constraint: TetrahedralVolumeConstraint

### Math

For surface triangle (p0, p1, p2) and center pc:

```
a = p0 - pc
b = p1 - pc
c = p2 - pc

V = (1/6) * dot(a, cross(b, c))

dV/dp0 = (1/6) * cross(b, c)
dV/dp1 = (1/6) * cross(c, a)
dV/dp2 = (1/6) * cross(a, b)
dV/dpc = -(dV/dp0 + dV/dp1 + dV/dp2)
```

Constraint: C = V - V_target = 0 (equality)

Each tetrahedron involves exactly 4 particles. The center participates in all
N_triangles tetrahedra, which naturally couples it to the surface centroid.

### Relation to Global Volume

By the divergence theorem:
  sum(V_tet_i for all faces) = global mesh volume

So per-tetrahedron equality constraints collectively imply global volume conservation.
The global VolumeConstraint becomes redundant and is removed.

### Shape Stability Risk

Per-tetrahedron constraints alone are under-constrained:
- Octahedron: 8 constraints, 21 DOF (15 deformation modes)
- Icosahedron: 20 constraints, 39 DOF (19 deformation modes)

Degenerate modes exist (surface points rotating while preserving tet volumes).
Mitigation: overdamped dynamics (damping=1.0) damps these modes passively.
Collision and adhesion constraints provide indirect shape stabilization.

If instability is observed after implementation, a weak center-surface
DistanceConstraint can be re-introduced as a fallback (see Fallback section).

## File Changes

### `develmorph/project.py`

Add new class `TetrahedralVolumeConstraint`. Existing classes are not removed
(backward compatibility).

```python
class TetrahedralVolumeConstraint:
    """
    XPBD equality constraint: volume of tetrahedron(center, p0, p1, p2) = target.
    """
    def __init__(self, center_idx, i0, i1, i2, target_volume, compliance=0.0):
        self.center_idx = center_idx
        self.i0 = i0
        self.i1 = i1
        self.i2 = i2
        self.target_volume = target_volume
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_acc):
        pc = x[self.center_idx]
        p0, p1, p2 = x[self.i0], x[self.i1], x[self.i2]
        a, b, c = p0 - pc, p1 - pc, p2 - pc
        V = np.dot(a, np.cross(b, c)) / 6.0
        C = V - self.target_volume
        if abs(C) < 1e-12:
            return lambda_acc
        g0 = np.cross(b, c) / 6.0
        g1 = np.cross(c, a) / 6.0
        g2 = np.cross(a, b) / 6.0
        gc = -(g0 + g1 + g2)
        w_grad_sq = (
            inv_masses[self.center_idx] * np.dot(gc, gc)
            + inv_masses[self.i0] * np.dot(g0, g0)
            + inv_masses[self.i1] * np.dot(g1, g1)
            + inv_masses[self.i2] * np.dot(g2, g2)
        )
        if w_grad_sq < 1e-12:
            return lambda_acc
        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0
        delta_lambda = (-C - alpha_tilde * lambda_acc) / (w_grad_sq + alpha_tilde)
        lambda_acc += delta_lambda
        x[self.center_idx] += delta_lambda * inv_masses[self.center_idx] * gc
        x[self.i0] += delta_lambda * inv_masses[self.i0] * g0
        x[self.i1] += delta_lambda * inv_masses[self.i1] * g1
        x[self.i2] += delta_lambda * inv_masses[self.i2] * g2
        return lambda_acc
```

### `develmorph/cell_body.py`

Add `face_volumes` to `CellBody.__init__()` and compute in `create_cell_body()`:

```python
# In create_cell_body(), alongside face_areas:
face_volumes = []
for a, b, c in triangles:
    p0 = new_positions[surface_indices[a]]
    p1 = new_positions[surface_indices[b]]
    p2 = new_positions[surface_indices[c]]
    a_vec = p0 - center
    b_vec = p1 - center
    c_vec = p2 - center
    face_volumes.append(np.dot(a_vec, np.cross(b_vec, c_vec)) / 6.0)
```

Invariant: `sum(face_volumes) == target_volume` (verified by test).

### `develmorph/solver.py`

Add `tet_volume_compliance=None` parameter to `step()`. When not None, use new path:

```python
# New path (tet_volume_compliance is not None):
tet_constraints = []
for cb in cell_bodies:
    for (a, b, c), v_target in zip(cb.triangles, cb.face_volumes):
        i0, i1, i2 = cb.surface_indices[a], cb.surface_indices[b], cb.surface_indices[c]
        tet_constraints.append(
            TetrahedralVolumeConstraint(cb.center_idx, i0, i1, i2, v_target,
                                        tet_volume_compliance)
        )

# Constraint order: tet -> collision -> adhesion -> surface_penetration -> shell
all_constraints = (
    tet_constraints
    + collision_constraints
    + adhesion_constraints
    + surface_penetration_constraints
    + surface_contact_constraints
    + shell_constraints
)
```

Old path (tet_volume_compliance=None): unchanged, all existing tests pass.

### `develmorph/engine.py`

Add `tet_volume_compliance=None` to `World.__init__()` and forward to `step()`.

## Constraint Priority

```
tet_volume  (local, equality, 4 particles)
collision   (inequality, repulsion)
adhesion    (inequality, attraction)
surface_penetration / halfspace
shell
```

tet_volume before collision: volume correction runs on the free (uncontacted) side
before contact constraints settle the touching side.

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `tet_volume_compliance` | `None` (disabled) | 0.0 = rigid; 0.01-0.1 for soft cells |

Start with `tet_volume_compliance=0.01` for initial experiments (softer than
volume_compliance=1.0 used currently, since there are N_triangles constraints
instead of 1, so total stiffness is higher).

## Test Plan

### Unit tests (write before implementation)

| Test | Assertion |
|------|-----------|
| `test_tet_volume_constraint_basic` | project() reduces |C| toward zero |
| `test_tet_volume_constraint_gradient` | dV/dpc == -(g0+g1+g2), verified numerically |
| `test_tet_volume_constraint_equality` | compliance=0, V converges to target after n_iter |
| `test_tet_volume_sign` | fires correctly when V < target (inverted face) |
| `test_face_volumes_in_create_cell_body` | face_volumes returned and len == n_triangles |
| `test_face_volumes_sum_equals_target_volume` | sum(face_volumes) ≈ target_volume |

### Integration tests

| Test | Assertion |
|------|-----------|
| `test_single_cell_tet_volume_stable` | 1 cell, 100 steps, sum(V_tet) ≈ target_volume |
| `test_two_cell_collision_tet_volume` | 2 cells collide, both maintain volume |
| `test_backward_compat_no_tet_param` | tet_volume_compliance=None, all 120 existing tests pass |

## Fallback

If pure tetrahedral constraints produce shape instability (center escapes surface
hull), re-introduce center-surface DistanceConstraint at weak compliance (~0.1)
as structural stabilizer. Surface-surface springs remain removed.

## Out of Scope

- Removing existing VolumeConstraint / DistanceConstraint classes (kept for compat)
- Changing collision, adhesion, or shell constraint behavior
- Parameter auto-tuning
