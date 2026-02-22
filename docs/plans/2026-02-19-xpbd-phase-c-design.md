# Phase C: Cell Adhesion Design

## Goal

Add cell-cell adhesion to the XPBD solver so that neighboring cells maintain
contact after touching, producing realistic packing behavior for 4-8 cell
stage C. elegans embryos.

## Approach

Replace center-to-center CollisionConstraint (repulsion only) with a
bidirectional **AdhesionConstraint** (repulsion + attraction). Adhesion pairs
are managed with hysteresis to prevent flickering at the boundary.

## Key Design Decisions

### 1. Scope: Packing Only

Phase C targets geometric packing — cells stick together at the correct
spacing. Differential adhesion (varying strength by cell type) and dynamic
adhesion lifecycle (state machines) are deferred to later phases.

### 2. Center-to-Center Adhesion

Adhesion operates between cell centers, not between individual surface points.
This is an extension of the existing CollisionConstraint infrastructure.

- Surface-to-surface adhesion would create up to 36 constraint pairs per cell
  pair (6x6 surface points), which is excessive for this phase.
- Center-to-center is sufficient for packing behavior: volume and distance
  constraints handle cell shape independently.

### 3. Rest Length = Surface Contact Distance

`rest_length = ri + rj` where ri, rj are the max surface-to-center distances
of each cell. This is the same value currently used by CollisionConstraint.
Cells are held at exactly touching distance — no overlap, no gap.

### 4. Hysteresis-Based Pair Management

Two distance thresholds prevent adhesion flickering:

- **Attach threshold**: `ri + rj + attach_margin` — when centers come within
  this distance, adhesion is created.
- **Detach threshold**: `ri + rj + detach_margin` (detach > attach) — adhesion
  is only broken when centers exceed this distance.

Once attached, a pair stays attached until it exceeds the detach threshold.
This prevents oscillation at the boundary.

### 5. Active Adhesions as External State

`active_adhesions: set[tuple[int, int]]` is maintained outside the solver and
passed into `step()`. Pairs are identified by cell_bodies list indices (not
particle indices, which change after `build_particle_arrays`).

The solver updates this set in-place each step:
- Adds pairs that enter the attach threshold
- Removes pairs that exceed the detach threshold
- Generates AdhesionConstraint only for active pairs

## Architecture

### New: AdhesionConstraint (`project.py`)

Bidirectional distance constraint between two particles. Identical to
DistanceConstraint in behavior — corrects both overlap (C < 0) and
separation (C > 0). Separated as a distinct class for semantic clarity
and future extensibility (e.g., compliance varying by cell type).

### Modified: `solver.step()`

New parameters:
- `active_adhesions: set[tuple[int, int]]` — mutable set, updated in-place
- `adhesion_compliance: float` — default 0.1
- `attach_margin: float` — default 0.3
- `detach_margin: float` — default 0.8

Changes:
- Replace CollisionConstraint loop with adhesion pair management logic
- Generate AdhesionConstraint for active pairs only
- Constraint order: volume -> distance -> adhesion -> shell

### Unchanged

- `project.py`: CollisionConstraint remains (may be used for surface-point
  collisions in the future)
- `cell_body.py`: No changes
- `division.py`: No changes. After division, new cell pairs that fall within
  the attach threshold will automatically gain adhesion in the next step.
- `engine.py`: Minimal change — pass `active_adhesions` set to solver

## Parameters

| Parameter | Meaning | Default |
|---|---|---|
| `adhesion_compliance` | Adhesion softness (0 = rigid bond) | 0.1 |
| `attach_margin` | Distance margin for creating adhesion | 0.3 |
| `detach_margin` | Distance margin for breaking adhesion | 0.8 |

## Constraint Priority Order

1. Volume (preserve cell volume)
2. Distance (preserve membrane shape)
3. **Adhesion** (maintain cell-cell contact)
4. Shell (keep inside eggshell)

Adhesion replaces Collision in the priority order. Since AdhesionConstraint
is bidirectional, it subsumes the repulsion behavior of CollisionConstraint.

## Division Behavior

No special adhesion handling at division. After `divide_cell_body`:
- Parent's adhesion pairs are implicitly removed (parent index no longer exists)
- Daughter cells that are within `attach_margin` of neighbors will gain
  adhesion in the next solver step automatically

## Test Strategy

- AdhesionConstraint unit tests: pull-back, push-back, no-correction
- Solver adhesion pair management: attach, detach, hysteresis
- Integration: two cells stay in contact after initial placement

## Future Extensions (Not In Scope)

- Differential adhesion by cell type
- Surface-point-level adhesion for contact flattening
- Adhesion state machine (weak -> strong -> permanent)
- Spatial hashing for 1000+ cell neighbor search
