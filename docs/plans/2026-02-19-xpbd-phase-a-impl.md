# XPBD Phase A: Core Solver Replacement — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the simplified PBD solver with a proper XPBD solver supporting collision and shell constraints.

**Architecture:** Rewrite `project.py` with Constraint class hierarchy (CollisionConstraint, ShellConstraint). Rewrite `solver.py` with XPBD main loop (predict -> build constraints -> iterate -> finalize). Update `engine.py` to pass inv_masses and shell_params. TDD throughout.

**Tech Stack:** Python 3.13, NumPy, pytest, uv

---

### Task 1: CollisionConstraint — Failing Tests

**Files:**
- Modify: `tests/test_project.py`

**Step 1: Write failing tests for CollisionConstraint**

```python
import numpy as np
from develmorph.project import CollisionConstraint


def test_collision_resolves_overlap():
    """Two overlapping particles are pushed apart."""
    x = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    radii = np.array([0.5, 0.5])
    dt = 0.01

    c = CollisionConstraint(i=0, j=1, rest_length=radii[0] + radii[1], compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    dist = np.linalg.norm(x[0] - x[1])
    assert dist >= (radii[0] + radii[1]) * 0.99


def test_collision_no_correction_when_separated():
    """No correction when particles are already separated."""
    x = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    assert np.allclose(x, x_before)


def test_collision_symmetric_correction():
    """Equal-mass particles receive equal and opposite corrections."""
    x = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c.project(x, inv_masses, dt, 0.0)

    # Midpoint should remain at original midpoint
    midpoint = (x[0] + x[1]) / 2
    assert np.allclose(midpoint, [0.25, 0.0, 0.0])


def test_collision_respects_mass_ratio():
    """Heavier particle moves less."""
    x = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    inv_masses = np.array([0.5, 1.0])  # particle 0 is heavier
    dt = 0.01

    x_before = x.copy()
    c = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c.project(x, inv_masses, dt, 0.0)

    move_0 = np.linalg.norm(x[0] - x_before[0])
    move_1 = np.linalg.norm(x[1] - x_before[1])
    assert move_1 > move_0  # lighter moves more


def test_collision_compliance_softens():
    """Higher compliance produces smaller correction per iteration."""
    x_hard = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    x_soft = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c_hard = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c_soft = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=1.0)

    c_hard.project(x_hard, inv_masses, dt, 0.0)
    c_soft.project(x_soft, inv_masses, dt, 0.0)

    dist_hard = np.linalg.norm(x_hard[0] - x_hard[1])
    dist_soft = np.linalg.norm(x_soft[0] - x_soft[1])

    # Hard constraint corrects more in one step
    assert dist_hard > dist_soft
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_project.py -v`
Expected: FAIL (ImportError — CollisionConstraint does not exist)

**Step 3: Commit failing tests**

```bash
git add tests/test_project.py
git commit -m "test: add failing tests for CollisionConstraint"
```

---

### Task 2: CollisionConstraint — Implementation

**Files:**
- Modify: `develmorph/project.py`

**Step 1: Replace project.py contents with CollisionConstraint**

```python
import numpy as np


class CollisionConstraint:
    """
    XPBD inequality constraint: |p_i - p_j| >= rest_length.

    Only applies correction when particles overlap (C < 0).
    Modifies positions in-place.
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

        # Inequality: only correct overlap (C < 0)
        if C >= 0.0:
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

Run: `uv run pytest tests/test_project.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add develmorph/project.py
git commit -m "feat: implement CollisionConstraint with XPBD"
```

---

### Task 3: ShellConstraint — Failing Tests

**Files:**
- Modify: `tests/test_project.py`

**Step 1: Append failing tests for ShellConstraint**

```python
from develmorph.project import ShellConstraint


def test_shell_pushes_inside():
    """Particle outside ellipsoid is pushed back inside."""
    x = np.array([[10.0, 0.0, 0.0]])
    inv_masses = np.array([1.0])
    dt = 0.01

    c = ShellConstraint(i=0, a=5.0, b=5.0, c=5.0, radius=0.5, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    # Effective shell: a - radius = 4.5
    val = (x[0, 0] / 4.5) ** 2 + (x[0, 1] / 4.5) ** 2 + (x[0, 2] / 4.5) ** 2
    assert val <= 1.01  # inside with small tolerance


def test_shell_no_correction_when_inside():
    """No correction when particle is already inside."""
    x = np.array([[1.0, 0.0, 0.0]])
    inv_masses = np.array([1.0])
    dt = 0.01

    c = ShellConstraint(i=0, a=5.0, b=5.0, c=5.0, radius=0.5, compliance=0.0)
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    assert np.allclose(x, x_before)


def test_shell_respects_cell_radius():
    """Cell surface (not just center) stays inside shell."""
    # Place center at exactly (a - radius) on x-axis: should be on boundary
    a, r = 5.0, 0.5
    x = np.array([[a - r, 0.0, 0.0]])
    inv_masses = np.array([1.0])
    dt = 0.01

    c = ShellConstraint(i=0, a=a, b=5.0, c=5.0, radius=r, compliance=0.0)
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    # Should not move (exactly on boundary)
    assert np.allclose(x, x_before, atol=1e-6)


def test_shell_ellipsoid_axes():
    """Asymmetric ellipsoid constrains correctly on short axis."""
    # a=5, b=2, c=2; place at y=3 (outside on short axis)
    x = np.array([[0.0, 3.0, 0.0]])
    inv_masses = np.array([1.0])
    dt = 0.01

    c = ShellConstraint(i=0, a=5.0, b=2.0, c=2.0, radius=0.0, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    val = (x[0, 1] / 2.0) ** 2
    assert val <= 1.01
```

**Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_project.py::test_shell_pushes_inside -v`
Expected: FAIL (ImportError — ShellConstraint does not exist)

**Step 3: Commit failing tests**

```bash
git add tests/test_project.py
git commit -m "test: add failing tests for ShellConstraint"
```

---

### Task 4: ShellConstraint — Implementation

**Files:**
- Modify: `develmorph/project.py`

**Step 1: Add ShellConstraint to project.py**

Append after CollisionConstraint class:

```python
class ShellConstraint:
    """
    XPBD inequality constraint: particle stays inside ellipsoidal boundary.

    Effective boundary is shrunk by cell radius so the cell surface
    (not just center) remains inside. Constraint value:
        C(x) = (px/(a-r))^2 + (py/(b-r))^2 + (pz/(c-r))^2 - 1
    Only corrects when C > 0 (outside).
    Modifies positions in-place.
    """

    def __init__(self, i: int, a: float, b: float, c: float,
                 radius: float = 0.0, compliance: float = 0.0):
        self.i = i
        self.ea = max(a - radius, 0.1)
        self.eb = max(b - radius, 0.1)
        self.ec = max(c - radius, 0.1)
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_acc):
        p = x[self.i]
        sx = p[0] / self.ea
        sy = p[1] / self.eb
        sz = p[2] / self.ec

        C = sx * sx + sy * sy + sz * sz - 1.0

        # Inequality: only correct when outside (C > 0)
        if C <= 0.0:
            return lambda_acc

        # Gradient of C w.r.t. position
        grad = 2.0 * np.array([
            p[0] / (self.ea * self.ea),
            p[1] / (self.eb * self.eb),
            p[2] / (self.ec * self.ec),
        ])
        grad_norm_sq = np.dot(grad, grad)

        if grad_norm_sq < 1e-12:
            return lambda_acc

        w = inv_masses[self.i]
        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0

        denom = w * grad_norm_sq + alpha_tilde
        if denom < 1e-12:
            return lambda_acc

        delta_lambda = (-C - alpha_tilde * lambda_acc) / denom
        lambda_acc += delta_lambda

        x[self.i] += delta_lambda * w * grad

        return lambda_acc
```

**Step 2: Run all constraint tests**

Run: `uv run pytest tests/test_project.py -v`
Expected: All 9 tests PASS

**Step 3: Commit**

```bash
git add develmorph/project.py
git commit -m "feat: implement ShellConstraint with XPBD"
```

---

### Task 5: XPBD Solver — Failing Tests

**Files:**
- Modify: `tests/test_solver.py`

**Step 1: Rewrite test_solver.py with XPBD tests**

```python
import numpy as np
from develmorph.solver import step


def test_solver_separates_overlapping_cells():
    """Two overlapping cells are pushed apart by XPBD solver."""
    pos = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    inv_masses = np.array([1.0, 1.0])
    radii = np.array([0.5, 0.5])

    new_pos, new_vel = step(
        positions=pos, velocities=vel, inv_masses=inv_masses,
        radii=radii, dt=0.01, shell_params=(5.0, 5.0, 5.0),
    )

    dist = np.linalg.norm(new_pos[0] - new_pos[1])
    assert dist >= (radii[0] + radii[1]) * 0.95


def test_solver_preserves_separated_cells():
    """Already-separated cells are not moved significantly."""
    pos = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    inv_masses = np.array([1.0, 1.0])
    radii = np.array([0.5, 0.5])

    new_pos, _ = step(
        positions=pos, velocities=vel, inv_masses=inv_masses,
        radii=radii, dt=0.01, shell_params=(10.0, 10.0, 10.0),
    )

    assert np.allclose(new_pos, pos, atol=1e-6)


def test_solver_enforces_shell():
    """Cell outside shell is pushed back inside."""
    pos = np.array([[10.0, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    inv_masses = np.array([1.0])
    radii = np.array([0.5])

    new_pos, _ = step(
        positions=pos, velocities=vel, inv_masses=inv_masses,
        radii=radii, dt=0.01, shell_params=(5.0, 5.0, 5.0),
    )

    ea = 5.0 - 0.5
    val = (new_pos[0, 0] / ea) ** 2
    assert val <= 1.05


def test_solver_updates_velocity():
    """Velocity is updated to reflect position correction."""
    pos = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    inv_masses = np.array([1.0, 1.0])
    radii = np.array([0.5, 0.5])

    _, new_vel = step(
        positions=pos, velocities=vel, inv_masses=inv_masses,
        radii=radii, dt=0.01, shell_params=(5.0, 5.0, 5.0),
    )

    # Cells were pushed apart, so they should have nonzero velocity
    assert not np.allclose(new_vel, 0.0)
    # Velocity should be opposite for the two cells
    assert np.dot(new_vel[0], new_vel[1]) < 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_solver.py -v`
Expected: FAIL (step() signature mismatch)

**Step 3: Commit failing tests**

```bash
git add tests/test_solver.py
git commit -m "test: add failing tests for XPBD solver"
```

---

### Task 6: XPBD Solver — Implementation

**Files:**
- Modify: `develmorph/solver.py`

**Step 1: Rewrite solver.py with XPBD loop**

```python
import numpy as np
from develmorph.project import CollisionConstraint, ShellConstraint


def step(positions, velocities, inv_masses, radii, dt,
         shell_params, n_iter=10, compliance=0.0):
    """
    Advance physical state by one time step using XPBD.

    Args:
        positions: (N, 3) particle positions
        velocities: (N, 3) particle velocities
        inv_masses: (N,) inverse masses
        radii: (N,) particle radii
        dt: time step
        shell_params: (a, b, c) ellipsoid semi-axes
        n_iter: number of constraint iterations
        compliance: constraint compliance (0.0 = rigid)

    Returns:
        (new_positions, new_velocities)
    """
    n = len(positions)
    x_prev = positions.copy()

    # 1. Predict positions (no external forces yet)
    x = positions + dt * velocities

    # 2. Build constraints
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(
                CollisionConstraint(i, j, radii[i] + radii[j], compliance)
            )
        constraints.append(
            ShellConstraint(i, *shell_params, radii[i], compliance)
        )

    # 3. XPBD iteration
    lambdas = [0.0] * len(constraints)
    for _ in range(n_iter):
        for k, c in enumerate(constraints):
            lambdas[k] = c.project(x, inv_masses, dt, lambdas[k])

    # 4. Update velocities
    new_vel = (x - x_prev) / dt if dt > 0 else np.zeros_like(x)

    return x, new_vel
```

**Step 2: Run solver tests**

Run: `uv run pytest tests/test_solver.py -v`
Expected: All 4 tests PASS

**Step 3: Run constraint tests too (regression)**

Run: `uv run pytest tests/test_project.py tests/test_solver.py -v`
Expected: All 13 tests PASS

**Step 4: Commit**

```bash
git add develmorph/solver.py
git commit -m "feat: implement XPBD solver loop"
```

---

### Task 7: Engine Integration

**Files:**
- Modify: `develmorph/engine.py`

**Step 1: Update engine.py to new solver signature**

```python
import numpy as np

from develmorph.solver import step
from develmorph.division import divide_cell
from develmorph.polarity import PolarityField


class World:

    def __init__(self, positions, radii, dt=0.01,
                 shell_params=(5.0, 5.0, 5.0)):

        self.positions = positions.copy()
        self.radii = radii.copy()

        self.velocities = np.zeros_like(positions)
        self.inv_masses = np.ones(len(positions))

        self.dt = dt
        self.time = 0
        self.step_count = 0

        self.shell_params = shell_params

        self.division_steps = []

        self.polarity = PolarityField()
        self.cell_polarity = self.polarity.value(self.positions)

    def set_divisions(self, steps):
        self.division_steps = steps

    def step(self):

        # Events
        if self.step_count in self.division_steps:

            idx = np.argmax(self.radii)

            self.positions, self.radii = divide_cell(self.positions, self.radii, idx)

            parent_p = self.cell_polarity[idx]
            self.cell_polarity = np.delete(self.cell_polarity, idx)
            self.cell_polarity = np.insert(
                self.cell_polarity, idx, [parent_p, parent_p]
            )

            self.velocities = np.zeros_like(self.positions)
            self.inv_masses = np.ones(len(self.positions))

        # Physics
        self.positions, self.velocities = step(
            self.positions, self.velocities,
            self.inv_masses, self.radii, self.dt,
            self.shell_params,
        )

        self.cell_polarity = self.polarity.value(self.positions)

        self.step_count += 1
        self.time += self.dt

    def run(self, n_steps):

        traj = []

        for _ in range(n_steps):

            self.step()

            traj.append((self.positions.copy(), self.radii.copy()))

        return traj
```

**Step 2: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS (test_project, test_solver, test_physics, test_division, test_viz, test_celllineage_phase2a)

**Step 3: Commit**

```bash
git add develmorph/engine.py
git commit -m "feat: integrate XPBD solver into World engine"
```

---

### Task 8: Update Demo Script

**Files:**
- Modify: `scripts/demo_orientation_division.py`

**Step 1: Update apply_constraints to use XPBD solver**

Replace `apply_constraints` function and update imports:

```python
from develmorph.solver import step as xpbd_step
```

Replace `apply_constraints`:

```python
def apply_constraints(cells, n_iter=10):
    """
    Apply XPBD constraints to all cells.
    """
    cell_list = list(cells.values())
    positions = np.array([c.position for c in cell_list])
    radii = np.array([c.radius for c in cell_list])
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    positions, _ = xpbd_step(
        positions, velocities, inv_masses, radii,
        dt=0.01,
        shell_params=(EGGSHELL_A, EGGSHELL_B, EGGSHELL_C),
        n_iter=n_iter,
    )

    for i, cell in enumerate(cell_list):
        cell.position = positions[i]
```

Remove old imports:

```python
# Remove these:
# from develmorph.project import project_overlap, project_shell
```

**Step 2: Run demo to verify it works**

Run: `uv run python scripts/demo_orientation_division.py`
Expected: Simulation runs without errors, cells are constrained inside eggshell

**Step 3: Commit**

```bash
git add scripts/demo_orientation_division.py
git commit -m "refactor: update demo script to use XPBD solver"
```

---

### Task 9: Clean Up Unused Code

**Files:**
- Modify: `develmorph/physics.py` (evaluate if still needed)
- Modify: `tests/test_physics.py` (evaluate if still needed)

**Step 1: Verify physics.py is unused**

`physics.py` contains `compute_repulsion` which is a force-based approach.
It is NOT imported by `solver.py`, `engine.py`, or any demo script.
Only `test_physics.py` imports it. It can be kept as-is since it doesn't
conflict with XPBD, or removed for cleanliness.

Decision: **Keep for now** — it does no harm and may be useful for comparison.
Remove in a future cleanup pass.

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 3: Final commit for Phase A**

```bash
git add -A
git commit -m "chore: Phase A complete — XPBD core solver replacement"
```
