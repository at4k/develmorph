# Phase A Design: XPBD Core Solver Replacement

## Overview

Replace the current simplified PBD solver (`solver.py` + `project.py`) with
a proper XPBD (Extended Position Based Dynamics) implementation. This phase
keeps the 1-cell = 1-particle model and focuses solely on collision and shell
constraints. No adhesion yet.

## Approach: Direct Replacement (Option 2)

Rewrite `solver.py` and `project.py` in place. No parallel implementations.

## Files Changed

| File | Change |
|------|--------|
| `project.py` | Delete `project_overlap` / `project_shell`. Replace with `Constraint` class hierarchy |
| `solver.py` | Overdamped PBD loop -> XPBD main loop |
| `engine.py` | Add `inv_masses` / `shell_params`, update `step()` call signature |
| `tests/test_solver.py` | Update to new `step()` signature |
| `tests/test_project.py` | Rewrite as constraint class unit tests |

## Files Unchanged

- `division.py` — pure geometry, independent of physics
- `celllineage.py` — division rules, independent of physics
- `polarity.py` — polarity field, independent of physics
- `lineage_recorder.py` — snapshot recording, unchanged
- `cell.py` — no changes in Phase A (mass/velocity on Cell deferred to Phase B)

## Constraint Classes (`project.py`)

```python
class Constraint:
    """Base class for XPBD constraints."""
    def project(self, x, inv_masses, dt, lambda_acc): ...

class CollisionConstraint(Constraint):
    """Inequality constraint: |pi - pj| >= ri + rj"""
    # i, j: particle indices
    # rest_length: ri + rj
    # compliance: alpha_tilde (0.0 = rigid)

class ShellConstraint(Constraint):
    """Inequality constraint: cell stays inside ellipsoidal boundary"""
    # i: particle index
    # a, b, c: ellipsoid semi-axes
    # radius: cell radius (shrinks effective shell)
    # compliance: alpha_tilde (0.0 = rigid)
```

### XPBD Position Correction Formula

```
C(x)        = |p_i - p_j| - L
alpha_tilde = compliance / dt^2
delta_lambda = (-C - alpha_tilde * lambda) / (w_i + w_j + alpha_tilde)
delta_x_i   = +delta_lambda * w_i * grad_i
delta_x_j   = -delta_lambda * w_j * grad_j
```

- `w` = `inv_mass` (inverse mass)
- `lambda` = Lagrange multiplier (accumulated per iteration)
- CollisionConstraint: only correct when C < 0 (overlapping)
- ShellConstraint: only correct when cell is outside boundary

## Solver Loop (`solver.py`)

```python
def step(positions, velocities, inv_masses, radii, dt,
         shell_params, n_iter=10, compliance=0.0):

    x_prev = positions.copy()

    # 1. Predict (no external forces for now)
    x = positions + dt * velocities

    # 2. Build constraints
    #    - CollisionConstraint for each pair
    #    - ShellConstraint for each particle

    # 3. XPBD iteration loop
    #    Reset lambda to 0 each step
    #    For each iteration, project all constraints

    # 4. Finalize: v = (x - x_prev) / dt

    return x, velocities
```

### Key Differences from Current Implementation

- Velocity is preserved (no overdamped zero-reset)
- Compliance provides explicit stiffness control
- Lagrange multipliers ensure physically consistent corrections
- shell_params injected externally (no hardcoded a=b=c=5.0)

## Engine Changes (`engine.py`)

- Add `self.inv_masses = np.ones(len(positions))` to `World.__init__`
- Add `self.shell_params = (a, b, c)` to `World.__init__`
- Update `World.step()` to pass new arguments to `solver.step()`
- Extend `inv_masses` array on cell division

## Test Strategy

### `test_project.py` -> Constraint Unit Tests

- CollisionConstraint resolves overlap (dist >= R after projection)
- ShellConstraint keeps cell inside boundary
- compliance=0.0 behaves as rigid constraint
- Larger compliance allows softer response

### `test_solver.py` -> Integration Tests

- Two overlapping cells separate after step()
- Cell outside shell returns inside after step()
- Post-division overlap resolves correctly

## Phase Boundary

Phase A is complete when:
- XPBD solver replaces old PBD solver
- All collision/shell tests pass
- Demo scripts produce equivalent or better results
- Compliance parameter demonstrably controls stiffness
