# Phase B Design: Surface Points + Volume Constraint

## Overview

Extend the XPBD solver from 1-cell = 1-particle to 1-cell = 1 center + 6 surface
points (octahedron). Add volume constraint (incompressible) and membrane distance
constraints (deformable). Cells behave like liquid droplets: volume is conserved
but shape changes freely under external forces.

This phase targets the early embryo (up to ~30 cells). The model will be
replaced by a coarser model at higher cell counts (future work).

## Design Decisions

- **Surface points**: 6 (octahedron vertices) — minimal but sufficient for
  "pushed in one direction, bulges in another" behavior
- **Cell data structure**: `CellBody` class holds indices into flat particle array
- **Solver**: remains flat-array based for performance; CellBody is metadata for
  constraint construction only
- **Volume constraint**: compliance ~= 0 (incompressible, cell interior is liquid)
- **Membrane constraints**: compliance > 0 (shape is deformable, membrane stiffness)
- **Inter-cell collision**: center-point based (Phase A style), rest_length =
  sum of max center-to-surface distances. Simple and avoids surface pass-through.
- **Shell constraint**: applied to both center (radius = max surface dist) and
  surface points (radius = 0)

## New File: `develmorph/cell_body.py`

```python
class CellBody:
    center_idx: int              # index in flat particle array
    surface_indices: list[int]   # 6 surface point indices
    triangles: list[tuple]       # 8 triangles (local indices 0-5)
    target_volume: float         # 4/3 * pi * r^3
    membrane_compliance: float   # membrane softness (> 0)
```

Octahedron vertices relative to center:
```
+x, -x, +y, -y, +z, -z  (each at distance = radius)
```

8 triangles (fixed topology, indices into surface_indices):
```
(0,2,4), (0,4,3), (0,3,5), (0,5,2),  # +x hemisphere
(1,4,2), (1,3,4), (1,5,3), (1,2,5)   # -x hemisphere
```

Helper function:
```python
def build_particle_arrays(cell_bodies) -> (positions, inv_masses):
    """Rebuild flat arrays from CellBody list. Reassigns all indices."""
```

## New Constraints in `develmorph/project.py`

### DistanceConstraint (equality)

```
C(x) = |p_i - p_j| - L_rest = 0
```

Unlike CollisionConstraint (inequality, C < 0 only), this corrects in both
directions. Used for:
- Center-to-surface: 6 constraints per cell
- Surface-to-adjacent-surface: 6 constraints per cell (octahedron edges)
- Total: 12 distance constraints per cell

### VolumeConstraint

```
V(x) = (1/6) * sum( dot(p0, cross(p1, p2)) )  for each triangle
C(x) = V(x) - V_target = 0
```

Gradient per vertex: sum of triangle normal contributions.
Compliance ~= 0 (incompressible).
Affects center point + all 6 surface points (7 particles).

## Changes to Existing Files

### `develmorph/solver.py`

New signature:
```python
def step(positions, velocities, inv_masses, dt,
         cell_bodies, shell_params, n_iter=10,
         membrane_compliance=0.01, volume_compliance=0.0):
```

Constraint construction order per iteration:
1. Volume constraints (highest priority — incompressible)
2. Distance constraints (membrane shape)
3. Collision constraints (inter-cell, center-based)
4. Shell constraints (boundary, all particles)

`radii` argument removed; derived from CellBody data.

### `develmorph/division.py`

Division now operates on CellBody:
1. Remove parent CellBody (7 particles)
2. Create 2 daughter CellBodies (14 particles)
3. Daughter centers: parent center +/- offset along division axis
4. Daughter surface points: octahedron at daughter radius
5. Daughter target_volume: parent volume * volume_ratio

Use `build_particle_arrays` to rebuild flat arrays after division.

### `develmorph/engine.py`

- Manage list of `CellBody` instead of just `radii`
- Call `build_particle_arrays` after division
- Pass `cell_bodies` to solver

## Test Strategy

### `tests/test_cell_body.py`
- Octahedron vertices at correct positions
- 8 triangles in mesh
- Target volume matches 4/3 * pi * r^3
- build_particle_arrays produces correct flat arrays

### `tests/test_project.py` (additions)
- DistanceConstraint: stretched -> contracts, compressed -> expands
- VolumeConstraint: deflated -> inflates, inflated -> deflates
- VolumeConstraint: pushed from one side -> bulges opposite side

### `tests/test_solver.py` (additions)
- Surface-point cell deforms on collision
- Volume preserved after deformation (within tolerance)
- Daughter volumes sum to parent volume after division

## Phase Boundary

Phase B is complete when:
- CellBody with octahedron surface points is implemented
- Volume constraint keeps cell volume constant under deformation
- Membrane constraints allow shape change
- Division produces correct daughter CellBodies
- Demo shows cells deforming and maintaining volume
