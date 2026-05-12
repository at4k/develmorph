# Membrane Area Constraint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-triangle area inequality constraint to the XPBD solver that prevents cell surface triangles from stretching beyond their rest area.

**Architecture:** New `TriangleAreaConstraint` class in `project.py` (inequality, C = A - A_target, fires only when C > 0). `CellBody` stores `face_areas` (computed at creation). `solver.step()` builds area constraints when `area_compliance` is not None. `World` threads the parameter through.

**Tech Stack:** Python 3.13, NumPy, uv / pytest

---

## File Map

| File | Change |
|---|---|
| `develmorph/project.py` | Add `TriangleAreaConstraint` |
| `develmorph/cell_body.py` | Add `face_areas` to `CellBody.__init__`; compute in `create_cell_body` |
| `develmorph/solver.py` | Add `area_compliance=None` param; build area constraints; import new class |
| `develmorph/engine.py` | Add `area_compliance=None` to `World.__init__` and `World.step` |
| `tests/test_project.py` | 3 unit tests for `TriangleAreaConstraint` |
| `tests/test_cell_body.py` | 1 test for `face_areas` storage |
| `tests/test_solver.py` | 1 integration test for `area_compliance` param |
| `tests/test_engine.py` | 1 smoke test for `World` with `area_compliance` |

---

## Task 1: `TriangleAreaConstraint` unit tests

**Files:**
- Modify: `tests/test_project.py`

- [ ] **Step 1: Write 3 failing tests**

Add to the bottom of `tests/test_project.py`:

```python
from develmorph.project import TriangleAreaConstraint


def test_triangle_area_no_correction_at_rest():
    """No correction when area equals target."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    x_before = x.copy()
    inv_masses = np.ones(3)
    c = TriangleAreaConstraint(0, 1, 2, target_area=0.5, compliance=0.0)
    c.project(x, inv_masses, dt=0.01, lambda_acc=0.0)
    np.testing.assert_array_almost_equal(x, x_before)


def test_triangle_area_corrects_stretch():
    """Correction fires and restores area when triangle is stretched."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],  # area = 1.0, target = 0.5
    ], dtype=float)
    inv_masses = np.ones(3)
    c = TriangleAreaConstraint(0, 1, 2, target_area=0.5, compliance=0.0)
    lam = 0.0
    for _ in range(20):
        lam = c.project(x, inv_masses, dt=0.01, lambda_acc=lam)
    n = np.cross(x[1] - x[0], x[2] - x[0])
    area = 0.5 * np.linalg.norm(n)
    assert area <= 0.5 * 1.01


def test_triangle_area_no_correction_on_compression():
    """No correction when area is below target (inequality: stretch only)."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],  # area = 0.25 < target 0.5
    ], dtype=float)
    x_before = x.copy()
    inv_masses = np.ones(3)
    c = TriangleAreaConstraint(0, 1, 2, target_area=0.5, compliance=0.0)
    c.project(x, inv_masses, dt=0.01, lambda_acc=0.0)
    np.testing.assert_array_equal(x, x_before)
```

- [ ] **Step 2: Run to confirm they fail**

```bash
uv run pytest tests/test_project.py::test_triangle_area_no_correction_at_rest tests/test_project.py::test_triangle_area_corrects_stretch tests/test_project.py::test_triangle_area_no_correction_on_compression -v
```

Expected: `ImportError: cannot import name 'TriangleAreaConstraint'`

---

## Task 2: Implement `TriangleAreaConstraint`

**Files:**
- Modify: `develmorph/project.py`

- [ ] **Step 3: Add class at bottom of `project.py`**

```python
class TriangleAreaConstraint:
    """
    XPBD inequality constraint: triangle area <= target_area.

    Fires only when A > A_target (stretch prevention).
    Compression is permitted — membranes can fold but not stretch.

    Indices i0, i1, i2 are surface particle indices (center excluded).
    """

    def __init__(
        self, i0: int, i1: int, i2: int, target_area: float, compliance: float = 0.0
    ):
        self.i0 = i0
        self.i1 = i1
        self.i2 = i2
        self.target_area = target_area
        self.compliance = compliance

    def project(self, x, inv_masses, dt, lambda_acc):
        p0 = x[self.i0]
        p1 = x[self.i1]
        p2 = x[self.i2]
        n_unnorm = np.cross(p1 - p0, p2 - p0)
        area2 = np.linalg.norm(n_unnorm)
        if area2 < 1e-12:
            return lambda_acc

        A = 0.5 * area2
        C = A - self.target_area

        # Inequality: only correct when area exceeds target
        if C <= 0.0:
            return lambda_acc

        n_hat = n_unnorm / area2
        grad0 = 0.5 * np.cross(p1 - p2, n_hat)
        grad1 = 0.5 * np.cross(p2 - p0, n_hat)
        grad2 = 0.5 * np.cross(p0 - p1, n_hat)

        w_grad_sq = (
            inv_masses[self.i0] * np.dot(grad0, grad0)
            + inv_masses[self.i1] * np.dot(grad1, grad1)
            + inv_masses[self.i2] * np.dot(grad2, grad2)
        )
        if w_grad_sq < 1e-12:
            return lambda_acc

        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0
        delta_lambda = (-C - alpha_tilde * lambda_acc) / (w_grad_sq + alpha_tilde)
        lambda_acc += delta_lambda

        x[self.i0] += delta_lambda * inv_masses[self.i0] * grad0
        x[self.i1] += delta_lambda * inv_masses[self.i1] * grad1
        x[self.i2] += delta_lambda * inv_masses[self.i2] * grad2

        return lambda_acc
```

- [ ] **Step 4: Run the 3 tests to confirm they pass**

```bash
uv run pytest tests/test_project.py::test_triangle_area_no_correction_at_rest tests/test_project.py::test_triangle_area_corrects_stretch tests/test_project.py::test_triangle_area_no_correction_on_compression -v
```

Expected: 3 PASSED

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
uv run pytest -q
```

Expected: 113 passed (all existing tests)

- [ ] **Step 6: Commit**

```bash
git add develmorph/project.py tests/test_project.py
git commit -m "feat: add TriangleAreaConstraint (stretch-only inequality)"
```

---

## Task 3: `face_areas` in `CellBody`

**Files:**
- Modify: `develmorph/cell_body.py`
- Modify: `tests/test_cell_body.py`

- [ ] **Step 7: Write failing test**

Add to the bottom of `tests/test_cell_body.py`:

```python
import math


def test_create_cell_body_oct_face_areas():
    """create_cell_body stores one face area per triangle (8 for oct)."""
    positions = np.zeros((0, 3))
    cb, _ = create_cell_body(center=np.zeros(3), radius=1.0, positions=positions)
    assert len(cb.face_areas) == 8
    # Octahedron r=1: each face is equilateral with side sqrt(2), area = sqrt(3)/2
    expected = math.sqrt(3) / 2
    for area in cb.face_areas:
        assert abs(area - expected) < 1e-10
```

- [ ] **Step 8: Run to confirm it fails**

```bash
uv run pytest tests/test_cell_body.py::test_create_cell_body_oct_face_areas -v
```

Expected: `AttributeError: 'CellBody' object has no attribute 'face_areas'`

- [ ] **Step 9: Add `face_areas` param to `CellBody.__init__`**

In `develmorph/cell_body.py`, add `face_areas=None` to `CellBody.__init__` and store it.
Insert after `self.surface_surface_rest = list(surface_surface_rest)` (line ~196):

```python
        self.face_areas = list(face_areas) if face_areas is not None else []
```

And add `face_areas=None` to the `__init__` signature:

```python
    def __init__(
        self,
        center_idx,
        surface_indices,
        target_volume,
        center_surface_rest,
        surface_surface_rest,
        membrane_compliance=0.01,
        triangles=None,
        edges=None,
        name=None,
        orientation=None,
        topology='oct',
        face_areas=None,
    ):
```

- [ ] **Step 10: Compute `face_areas` in `create_cell_body`**

In `develmorph/cell_body.py`, inside `create_cell_body`, add face area computation
after the `target_volume` block (after the existing `target_volume /= 6.0` line):

```python
    # Compute initial face areas (used by TriangleAreaConstraint)
    face_areas = []
    for a, b, c in triangles:
        p0 = new_positions[surface_indices[a]]
        p1 = new_positions[surface_indices[b]]
        p2 = new_positions[surface_indices[c]]
        n = np.cross(p1 - p0, p2 - p0)
        face_areas.append(0.5 * np.linalg.norm(n))
```

Then pass `face_areas=face_areas` to the `CellBody(...)` constructor call.

- [ ] **Step 11: Run the new test**

```bash
uv run pytest tests/test_cell_body.py::test_create_cell_body_oct_face_areas -v
```

Expected: PASSED

- [ ] **Step 12: Run full suite**

```bash
uv run pytest -q
```

Expected: 114 passed

- [ ] **Step 13: Commit**

```bash
git add develmorph/cell_body.py tests/test_cell_body.py
git commit -m "feat: store face_areas in CellBody via create_cell_body"
```

---

## Task 4: Integrate into `solver.py`

**Files:**
- Modify: `develmorph/solver.py`
- Modify: `tests/test_solver.py`

- [ ] **Step 14: Write failing test**

Add to `tests/test_solver.py`:

```python
def test_step_with_area_compliance():
    """step() with area_compliance runs without error and returns finite positions."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.zeros(3), radius=1.0, positions=positions
    )
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions,
        velocities=vel,
        inv_masses=inv_masses,
        dt=0.01,
        cell_bodies=[cb],
        shell_params=(5.0, 5.0, 5.0),
        area_compliance=0.01,
        n_iter=10,
    )
    assert np.all(np.isfinite(new_pos))
```

- [ ] **Step 15: Run to confirm it fails**

```bash
uv run pytest tests/test_solver.py::test_step_with_area_compliance -v
```

Expected: `TypeError: step() got an unexpected keyword argument 'area_compliance'`

- [ ] **Step 16: Update `solver.py`**

Add `TriangleAreaConstraint` to the import at the top of `develmorph/solver.py`:

```python
from develmorph.project import (
    AdhesionConstraint,
    CollisionConstraint,
    DistanceConstraint,
    HalfSpaceConstraint,
    ShellConstraint,
    TriangleAreaConstraint,
    VolumeConstraint,
)
```

Add `area_compliance=None` to `step()` signature (after `surface_contact_compliance=None`):

```python
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
):
```

Add area constraint construction after the `distance_constraints` loop (before collision constraints).
Insert after the `distance_constraints` block (around line 145):

```python
    # Area constraints: per-triangle stretch prevention.
    # None = disabled (backward compatible). Fires only when A > A_target.
    area_constraints = []
    if area_compliance is not None:
        for cb in cell_bodies:
            for (a, b, c), target_a in zip(cb.triangles, cb.face_areas):
                i0 = cb.surface_indices[a]
                i1 = cb.surface_indices[b]
                i2 = cb.surface_indices[c]
                area_constraints.append(
                    TriangleAreaConstraint(i0, i1, i2, target_a, area_compliance)
                )
```

Update `all_constraints` to include `area_constraints` between distance and adhesion:

```python
    all_constraints = (
        volume_constraints
        + distance_constraints
        + area_constraints
        + adhesion_constraints
        + collision_constraints
        + surface_penetration_constraints
        + surface_contact_constraints
        + shell_constraints
    )
```

- [ ] **Step 17: Run the new test**

```bash
uv run pytest tests/test_solver.py::test_step_with_area_compliance -v
```

Expected: PASSED

- [ ] **Step 18: Run full suite**

```bash
uv run pytest -q
```

Expected: 115 passed

- [ ] **Step 19: Commit**

```bash
git add develmorph/solver.py tests/test_solver.py
git commit -m "feat: add area_compliance param to step() using TriangleAreaConstraint"
```

---

## Task 5: Thread through `World` in `engine.py`

**Files:**
- Modify: `develmorph/engine.py`
- Modify: `tests/test_engine.py`

- [ ] **Step 20: Write failing test**

Add to `tests/test_engine.py`:

```python
def test_world_area_compliance_runs():
    """World with area_compliance runs without error."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.zeros(3), radius=1.0, positions=positions
    )
    world = World(
        cell_bodies=[cb],
        positions=positions,
        area_compliance=0.01,
    )
    world.run(5)
    assert np.all(np.isfinite(world.positions))
```

- [ ] **Step 21: Run to confirm it fails**

```bash
uv run pytest tests/test_engine.py::test_world_area_compliance_runs -v
```

Expected: `TypeError: World.__init__() got an unexpected keyword argument 'area_compliance'`

- [ ] **Step 22: Update `engine.py`**

In `World.__init__`, add `area_compliance=None` to the signature and store it:

```python
    def __init__(self, cell_bodies, positions, dt=0.01, shell_params=(5.0, 5.0, 5.0),
                 lineage=None, n_iter=10, damping=0.0, shell_compliance=0.0,
                 collision_compliance=0.0, membrane_compliance=0.001,
                 volume_compliance=1.0, contact_overlap=0.0,
                 surface_contact_compliance=None, area_compliance=None):
```

Add after `self.surface_contact_compliance = surface_contact_compliance`:

```python
        self.area_compliance = area_compliance
```

In `World.step`, add `area_compliance=self.area_compliance` to the `step(...)` call:

```python
        self.positions, self.velocities = step(
            self.positions,
            self.velocities,
            self.inv_masses,
            self.dt,
            self.cell_bodies,
            self.shell_params,
            n_iter=self.n_iter,
            active_adhesions=self.active_adhesions,
            damping=self.damping,
            shell_compliance=self.shell_compliance,
            collision_compliance=self.collision_compliance,
            membrane_compliance=self.membrane_compliance,
            volume_compliance=self.volume_compliance,
            contact_overlap=self.contact_overlap,
            surface_contact_compliance=self.surface_contact_compliance,
            area_compliance=self.area_compliance,
        )
```

- [ ] **Step 23: Run the new test**

```bash
uv run pytest tests/test_engine.py::test_world_area_compliance_runs -v
```

Expected: PASSED

- [ ] **Step 24: Run full suite**

```bash
uv run pytest -q
```

Expected: 116 passed

- [ ] **Step 25: Commit**

```bash
git add develmorph/engine.py tests/test_engine.py
git commit -m "feat: thread area_compliance through World"
```
