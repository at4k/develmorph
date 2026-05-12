# Membrane Area Constraint Design

**Date:** 2026-05-12
**Branch:** feat-xpbd-solver
**Status:** Approved

## Motivation

Cell membrane deformation at contact zones produces local triangle area imbalances. The existing
`DistanceConstraint` on surface-surface edges implicitly constrains area when compliance is low,
but does not directly detect or correct per-triangle area distortion. An explicit per-triangle
area constraint adds direct area preservation with configurable compliance.

## Constraint Formulation

For each triangular face (p0, p1, p2) on the cell surface mesh:

```
n_unnorm = cross(p1-p0, p2-p0)
A        = 0.5 * ||n_unnorm||
n̂        = n_unnorm / ||n_unnorm||
C        = A - A_target
```

**Inequality (stretch-only):** correction fires only when `C > 0` (area exceeds target).
Compression is permitted — this allows contact-zone flattening and wrinkle-like behavior,
matching the physical property of lipid bilayers (high area stretch modulus, but can fold).

### Gradients

Derived via the "opposite edge × unit normal" identity:

```
∂A/∂p0 = 0.5 * cross(p1-p2, n̂)
∂A/∂p1 = 0.5 * cross(p2-p0, n̂)
∂A/∂p2 = 0.5 * cross(p0-p1, n̂)
```

Center particle is not part of any triangle; it is excluded from this constraint.

### XPBD Update

```
w_grad_sq    = w0*|g0|² + w1*|g1|² + w2*|g2|²
alpha_tilde  = area_compliance / dt²
delta_lambda = (-C - alpha_tilde * lambda) / (w_grad_sq + alpha_tilde)
pi          += delta_lambda * wi * gi   (for i in {0,1,2})
```

## Architecture

### Files Changed

| File | Change |
|---|---|
| `develmorph/project.py` | Add `TriangleAreaConstraint` class |
| `develmorph/cell_body.py` | Add `face_areas: list[float]` to `CellBody`; compute in `create_cell_body` |
| `develmorph/solver.py` | Add `area_compliance=None` param; build and insert area constraints |
| `develmorph/engine.py` | Thread `area_compliance` through `World.__init__` and `World.step` |

### `face_areas` Storage

Computed once in `create_cell_body` from the initial mesh geometry. Stored as a plain `list[float]`
on `CellBody`. `build_particle_arrays` only reassigns particle indices and does not touch `face_areas`,
so no additional update is needed after division.

### Constraint Priority Order

```
volume → distance → area → adhesion → collision → surface_penetration → surface_contact → shell
```

Area sits between distance and adhesion: membrane shape constraints are resolved before
cell-cell interaction forces.

### Backward Compatibility

`area_compliance=None` skips all area constraint construction. All existing callers remain
unaffected without any argument change.

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `area_compliance` | `None` | `None` = disabled. Suggested starting value: `0.01` |

A value similar to or slightly larger than `membrane_compliance` is a reasonable starting point.
Setting it to `0.0` gives a hard (near-incompressible) area constraint.

## Testing

- Unit test: single triangle, perturb one vertex outward, verify area is restored
- Unit test: perturb inward (compression), verify no correction fires
- Integration test: cell under deformation conserves per-triangle area within tolerance
- Regression: existing tests with `area_compliance=None` still pass
