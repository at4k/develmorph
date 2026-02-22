# Phase C: Cell Adhesion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add center-to-center cell adhesion with hysteresis to the XPBD solver, replacing collision-only behavior with bidirectional distance constraints that keep neighboring cells in contact.

**Architecture:** AdhesionConstraint is a new bidirectional distance constraint in `project.py`. The solver manages adhesion pairs via a `set[tuple[int,int]]` with attach/detach thresholds for hysteresis. CollisionConstraint remains but is no longer used for center-to-center interactions.

**Tech Stack:** Python 3.13, NumPy, pytest, uv

---

### Task 1: AdhesionConstraint Failing Tests

**Files:**
- Modify: `tests/test_project.py`

**Step 1: Write the failing tests**

Add these tests at the end of `tests/test_project.py`:

```python
# --- AdhesionConstraint tests ---

from develmorph.project import AdhesionConstraint


def test_adhesion_pulls_back_when_separated():
    """Separated pair is pulled back to rest length (attraction)."""
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    dist = np.linalg.norm(x[0] - x[1])
    assert abs(dist - 1.0) < 0.05


def test_adhesion_pushes_apart_when_overlapping():
    """Overlapping pair is pushed apart to rest length (repulsion)."""
    x = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    dist = np.linalg.norm(x[0] - x[1])
    assert abs(dist - 1.0) < 0.05


def test_adhesion_no_correction_at_rest():
    """No correction when at rest length."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    assert np.allclose(x, x_before, atol=1e-6)


def test_adhesion_compliance_softens():
    """Higher compliance allows more deviation from rest length."""
    x_hard = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    x_soft = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c_hard = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c_soft = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=1.0)

    c_hard.project(x_hard, inv_masses, dt, 0.0)
    c_soft.project(x_soft, inv_masses, dt, 0.0)

    err_hard = abs(np.linalg.norm(x_hard[0] - x_hard[1]) - 1.0)
    err_soft = abs(np.linalg.norm(x_soft[0] - x_soft[1]) - 1.0)

    assert err_hard < err_soft
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_project.py -k "adhesion" -v`
Expected: FAIL with `ImportError: cannot import name 'AdhesionConstraint'`

**Step 3: Commit**

```bash
git add tests/test_project.py
git commit -m "test: add failing tests for AdhesionConstraint"
```

---

### Task 2: AdhesionConstraint Implementation

**Files:**
- Modify: `develmorph/project.py`

**Step 1: Implement AdhesionConstraint**

Add this class at the end of `develmorph/project.py`:

```python
class AdhesionConstraint:
    """
    XPBD equality constraint for cell-cell adhesion: |p_i - p_j| = rest_length.

    Bidirectional distance constraint between cell centers. Corrects both
    overlap (C < 0, repulsion) and separation (C > 0, attraction).
    Functionally identical to DistanceConstraint but semantically separate
    for cell-cell adhesion.
    """

    def __init__(self, i: int, j: int, rest_length: float, compliance: float = 0.0):
        self.i = i
        self.j = j
        self.rest_length = rest_length
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_acc):
        dx = x[self.i] - x[self.j]
        dist = np.linalg.norm(dx)

        if dist < 1e-12:
            return lambda_acc

        C = dist - self.rest_length

        if abs(C) < 1e-12:
            return lambda_acc

        grad = dx / dist
        w_sum = inv_masses[self.i] + inv_masses[self.j]

        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0

        denom = w_sum + alpha_tilde
        if denom < 1e-12:
            return lambda_acc

        delta_lambda = (-C - alpha_tilde * lambda_acc) / denom
        lambda_acc += delta_lambda

        x[self.i] += delta_lambda * inv_masses[self.i] * grad
        x[self.j] -= delta_lambda * inv_masses[self.j] * grad

        return lambda_acc
```

**Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_project.py -k "adhesion" -v`
Expected: 4 PASSED

**Step 3: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS (52 total: 48 existing + 4 new)

**Step 4: Commit**

```bash
git add develmorph/project.py tests/test_project.py
git commit -m "feat: implement AdhesionConstraint with XPBD"
```

---

### Task 3: Solver Adhesion Pair Management Failing Tests

**Files:**
- Modify: `tests/test_solver.py`

**Step 1: Write the failing tests**

Add these tests at the end of `tests/test_solver.py`:

```python
def test_solver_adhesion_attaches_nearby_cells():
    """Cells within attach threshold gain adhesion."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    cb2, positions = create_cell_body(
        center=np.array([1.2, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    active_adhesions = set()

    step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb1, cb2],
        shell_params=(10.0, 10.0, 10.0),
        active_adhesions=active_adhesions,
        attach_margin=0.3,
        detach_margin=0.8,
    )

    # Distance = 1.2, ri+rj ~ 0.5+0.5 = 1.0, threshold = 1.0 + 0.3 = 1.3
    # 1.2 < 1.3 so should attach
    assert (0, 1) in active_adhesions


def test_solver_adhesion_does_not_attach_far_cells():
    """Cells beyond attach threshold do not gain adhesion."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    cb2, positions = create_cell_body(
        center=np.array([5.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    active_adhesions = set()

    step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb1, cb2],
        shell_params=(10.0, 10.0, 10.0),
        active_adhesions=active_adhesions,
        attach_margin=0.3,
        detach_margin=0.8,
    )

    assert len(active_adhesions) == 0


def test_solver_adhesion_detaches_when_far():
    """Adhesion is removed when cells exceed detach threshold."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    cb2, positions = create_cell_body(
        center=np.array([5.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    # Pre-existing adhesion
    active_adhesions = {(0, 1)}

    step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb1, cb2],
        shell_params=(10.0, 10.0, 10.0),
        active_adhesions=active_adhesions,
        attach_margin=0.3,
        detach_margin=0.8,
    )

    # Distance = 5.0, ri+rj ~ 1.0, detach threshold = 1.0 + 0.8 = 1.8
    # 5.0 > 1.8 so should detach
    assert len(active_adhesions) == 0


def test_solver_adhesion_hysteresis():
    """Adhesion persists between attach and detach thresholds (hysteresis)."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    # Place at distance 1.5: beyond attach (1.3) but within detach (1.8)
    cb2, positions = create_cell_body(
        center=np.array([1.5, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    # Pre-existing adhesion (was attached previously)
    active_adhesions = {(0, 1)}

    step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb1, cb2],
        shell_params=(10.0, 10.0, 10.0),
        active_adhesions=active_adhesions,
        attach_margin=0.3,
        detach_margin=0.8,
    )

    # Distance = 1.5, ri+rj ~ 1.0
    # attach threshold = 1.3 (beyond), detach threshold = 1.8 (within)
    # Should stay attached due to hysteresis
    assert (0, 1) in active_adhesions


def test_solver_adhesion_keeps_cells_together():
    """Adhered cells stay near rest length over multiple steps."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    cb2, positions = create_cell_body(
        center=np.array([1.0, 0.0, 0.0]), radius=0.5, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    active_adhesions = set()

    # Run multiple steps
    for _ in range(5):
        positions, vel = step(
            positions=positions,
            velocities=vel,
            inv_masses=inv_masses,
            dt=0.01,
            cell_bodies=[cb1, cb2],
            shell_params=(10.0, 10.0, 10.0),
            active_adhesions=active_adhesions,
            adhesion_compliance=0.1,
            attach_margin=0.3,
            detach_margin=0.8,
        )

    dist = np.linalg.norm(positions[cb1.center_idx] - positions[cb2.center_idx])
    # Should be near ri+rj = 1.0
    assert 0.5 < dist < 2.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_solver.py -k "adhesion" -v`
Expected: FAIL with `TypeError: step() got an unexpected keyword argument 'active_adhesions'`

**Step 3: Commit**

```bash
git add tests/test_solver.py
git commit -m "test: add failing tests for solver adhesion pair management"
```

---

### Task 4: Solver Adhesion Pair Management Implementation

**Files:**
- Modify: `develmorph/solver.py`

**Step 1: Implement adhesion pair management in solver**

Replace the full `develmorph/solver.py` with:

```python
import numpy as np

from develmorph.project import (
    AdhesionConstraint,
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
    Update active adhesion pairs based on distance thresholds.

    Uses hysteresis: attach_margin < detach_margin.
    Pairs are identified by cell_bodies list indices.
    Modifies active_adhesions in-place.
    """
    to_remove = []
    for i, j in active_adhesions:
        if i >= len(cell_bodies) or j >= len(cell_bodies):
            to_remove.append((i, j))
            continue
        ci = cell_bodies[i]
        cj = cell_bodies[j]
        ri = _max_surface_dist(positions, ci)
        rj = _max_surface_dist(positions, cj)
        dist = np.linalg.norm(positions[ci.center_idx] - positions[cj.center_idx])
        if dist > ri + rj + detach_margin:
            to_remove.append((i, j))

    for pair in to_remove:
        active_adhesions.discard(pair)

    for i in range(len(cell_bodies)):
        for j in range(i + 1, len(cell_bodies)):
            if (i, j) in active_adhesions:
                continue
            ci = cell_bodies[i]
            cj = cell_bodies[j]
            ri = _max_surface_dist(positions, ci)
            rj = _max_surface_dist(positions, cj)
            dist = np.linalg.norm(
                positions[ci.center_idx] - positions[cj.center_idx]
            )
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
        volume_compliance: compliance for volume constraints
        active_adhesions: mutable set of (i, j) cell index pairs, updated in-place
        adhesion_compliance: compliance for adhesion constraints
        attach_margin: distance margin for creating adhesion
        detach_margin: distance margin for breaking adhesion

    Returns:
        (new_positions, new_velocities)
    """
    x_prev = positions.copy()

    # 1. Predict positions
    x = positions + dt * velocities

    # 2. Update adhesion pairs
    if active_adhesions is not None:
        _update_adhesion_pairs(
            x, cell_bodies, active_adhesions, attach_margin, detach_margin
        )

    # 3. Build constraints in priority order
    volume_constraints = []
    distance_constraints = []
    adhesion_constraints = []
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
        for (ci, si), rest_len in zip(
            cb.center_surface_edges, cb.center_surface_rest
        ):
            distance_constraints.append(
                DistanceConstraint(ci, si, rest_len, membrane_compliance)
            )

        # Distance constraints: surface-surface (rest lengths from CellBody)
        for (a, b), rest_len in zip(
            cb.surface_surface_edges, cb.surface_surface_rest
        ):
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

    # Adhesion constraints: for active pairs only
    if active_adhesions is not None:
        for i, j in active_adhesions:
            if i >= len(cell_bodies) or j >= len(cell_bodies):
                continue
            ci = cell_bodies[i]
            cj = cell_bodies[j]
            ri = _max_surface_dist(x, ci)
            rj = _max_surface_dist(x, cj)
            adhesion_constraints.append(
                AdhesionConstraint(
                    ci.center_idx, cj.center_idx, ri + rj, adhesion_compliance
                )
            )
    else:
        # Backward compatibility: use collision when no adhesion management
        for i in range(len(cell_bodies)):
            for j in range(i + 1, len(cell_bodies)):
                ci = cell_bodies[i]
                cj = cell_bodies[j]
                ri = _max_surface_dist(positions, ci)
                rj = _max_surface_dist(positions, cj)
                from develmorph.project import CollisionConstraint

                adhesion_constraints.append(
                    CollisionConstraint(
                        ci.center_idx, cj.center_idx, ri + rj, 0.0
                    )
                )

    # Ordered: volume -> distance -> adhesion -> shell
    all_constraints = (
        volume_constraints
        + distance_constraints
        + adhesion_constraints
        + shell_constraints
    )

    # 4. XPBD iteration
    lambdas = [0.0] * len(all_constraints)
    for _ in range(n_iter):
        for k, c in enumerate(all_constraints):
            lambdas[k] = c.project(x, inv_masses, dt, lambdas[k])

    # 5. Update velocities
    new_vel = (x - x_prev) / dt if dt > 0 else np.zeros_like(x)

    return x, new_vel
```

**Step 2: Run adhesion tests**

Run: `uv run pytest tests/test_solver.py -k "adhesion" -v`
Expected: 5 PASSED

**Step 3: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS (57 total: 48 + 4 adhesion constraint + 5 solver adhesion)

**Step 4: Commit**

```bash
git add develmorph/solver.py
git commit -m "feat: add adhesion pair management with hysteresis to solver"
```

---

### Task 5: Engine Integration

**Files:**
- Modify: `develmorph/engine.py`

**Step 1: Read current engine.py**

Read: `develmorph/engine.py`

**Step 2: Add active_adhesions to World**

Add `self.active_adhesions = set()` in `__init__` and pass it to `step()`:

In `__init__`, after `self.cell_polarity = ...`:
```python
self.active_adhesions = set()
```

In `self.step()`, change the `step()` call to pass `active_adhesions`:
```python
self.positions, self.velocities = step(
    self.positions,
    self.velocities,
    self.inv_masses,
    self.dt,
    self.cell_bodies,
    self.shell_params,
    active_adhesions=self.active_adhesions,
)
```

After division (after `self.velocities = np.zeros_like(self.positions)`), clear stale adhesion pairs:
```python
# Clear adhesion pairs (stale indices after division)
self.active_adhesions.clear()
```

**Step 3: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add develmorph/engine.py
git commit -m "feat: integrate adhesion pair management into World engine"
```

---

### Task 6: Demo Script Update

**Files:**
- Modify: `scripts/demo_orientation_division.py`

**Step 1: Read current demo script**

Read: `scripts/demo_orientation_division.py`

**Step 2: Add active_adhesions to demo**

In `main()`, after `cell_body_map = {"P0": cb_p0}`:
```python
active_adhesions = set()
```

In `apply_constraints()`, add `active_adhesions` parameter and pass it to `xpbd_step`:
```python
def apply_constraints(cells, cell_bodies, positions, active_adhesions, n_iter=10):
    """Apply XPBD constraints with surface-point model."""
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    positions, _ = xpbd_step(
        positions,
        velocities,
        inv_masses,
        dt=0.01,
        cell_bodies=cell_bodies,
        shell_params=(EGGSHELL_A, EGGSHELL_B, EGGSHELL_C),
        n_iter=n_iter,
        active_adhesions=active_adhesions,
    )

    # Update Cell positions from CellBody centers
    cell_list = list(cells.values())
    for i, cell in enumerate(cell_list):
        cell.position = positions[cell_bodies[i].center_idx].copy()

    return positions
```

After division (after `build_particle_arrays`), clear adhesion pairs:
```python
active_adhesions.clear()
```

Update the `apply_constraints` call to pass `active_adhesions`:
```python
positions = apply_constraints(cells, cell_bodies, positions, active_adhesions)
```

**Step 3: Run demo to verify**

Run: `uv run python scripts/demo_orientation_division.py`
Expected: Completes without error, shows 8-cell stage

**Step 4: Commit**

```bash
git add scripts/demo_orientation_division.py
git commit -m "refactor: update demo script for adhesion-enabled solver"
```

---

### Task 7: Final Verification

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 2: Run demo**

Run: `uv run python scripts/demo_orientation_division.py`
Expected: Completes without error

**Step 3: Commit any remaining changes**

```bash
git status
# If clean, skip. Otherwise commit.
```
