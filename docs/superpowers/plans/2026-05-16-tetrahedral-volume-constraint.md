# Tetrahedral Volume Constraint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the spring-network membrane model (DistanceConstraint + TriangleAreaConstraint + VolumeConstraint) with per-tetrahedron volume constraints that preserve each (surface triangle, center) tetrahedron's volume independently, producing a more physically motivated pressure-driven membrane model.

**Architecture:** Add `TetrahedralVolumeConstraint` to `project.py`, store per-face target volumes as `face_volumes` in `CellBody`, and introduce `tet_volume_compliance=None` opt-in parameter to `solver.step()` and `engine.World`. Old path (None) runs unchanged for backward compatibility.

**Tech Stack:** Python 3.13, NumPy, XPBD (Extended Position Based Dynamics), uv (test runner: `uv run pytest`)

---

## File Map

| File | Change |
|------|--------|
| `develmorph/cell_body.py` | Add `face_volumes` param to `CellBody.__init__()`, compute in `create_cell_body()` |
| `develmorph/project.py` | Add `TetrahedralVolumeConstraint` class |
| `develmorph/solver.py` | Add `tet_volume_compliance=None` param; branch new vs old constraint path |
| `develmorph/engine.py` | Add `tet_volume_compliance=None` to `World.__init__()`, forward to `step()` |
| `tests/test_cell_body.py` | Add `face_volumes` tests |
| `tests/test_project.py` | Add `TetrahedralVolumeConstraint` unit tests |
| `tests/test_solver.py` | Add integration tests + backward compat test |
| `tests/test_engine.py` | Add `World` wiring test |

---

## Task 1: Add `face_volumes` to `CellBody`

**Files:**
- Modify: `develmorph/cell_body.py`
- Test: `tests/test_cell_body.py`

- [ ] **Step 1.1: Write failing tests**

Append to `tests/test_cell_body.py`:

```python
def test_face_volumes_in_create_cell_body():
    cb, _ = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=np.empty((0, 3)),
    )
    assert hasattr(cb, 'face_volumes')
    assert len(cb.face_volumes) == len(cb.triangles)


def test_face_volumes_sum_equals_target_volume():
    cb, _ = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=np.empty((0, 3)),
    )
    assert abs(sum(cb.face_volumes) - cb.target_volume) < 1e-10


def test_face_volumes_all_positive():
    cb, _ = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=2.0,
        positions=np.empty((0, 3)),
    )
    assert all(v > 0.0 for v in cb.face_volumes)


def test_face_volumes_ico():
    cb, _ = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=np.empty((0, 3)),
        topology='ico',
    )
    assert len(cb.face_volumes) == len(cb.triangles)
    assert abs(sum(cb.face_volumes) - cb.target_volume) < 1e-10
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_cell_body.py::test_face_volumes_in_create_cell_body tests/test_cell_body.py::test_face_volumes_sum_equals_target_volume tests/test_cell_body.py::test_face_volumes_all_positive tests/test_cell_body.py::test_face_volumes_ico -v
```

Expected: FAIL with `AttributeError: 'CellBody' object has no attribute 'face_volumes'`

- [ ] **Step 1.3: Add `face_volumes` param to `CellBody.__init__()`**

In `develmorph/cell_body.py`, locate the `__init__` signature (line 166). Add `face_volumes=None` after `face_areas=None`:

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
        face_volumes=None,
    ):
```

Add to the body of `__init__()`, after the `self.face_areas` line (line 198):

```python
        self.face_volumes = list(face_volumes) if face_volumes is not None else []
```

- [ ] **Step 1.4: Compute `face_volumes` in `create_cell_body()`**

In `develmorph/cell_body.py`, in `create_cell_body()`, add the computation immediately after the `face_areas` block (after line 285):

```python
    # Compute per-tetrahedron target volumes (used by TetrahedralVolumeConstraint)
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

Pass to `CellBody(...)` constructor call (line 299): add `face_volumes=face_volumes` after `face_areas=face_areas`.

- [ ] **Step 1.5: Run tests to confirm they pass**

```bash
uv run pytest tests/test_cell_body.py::test_face_volumes_in_create_cell_body tests/test_cell_body.py::test_face_volumes_sum_equals_target_volume tests/test_cell_body.py::test_face_volumes_all_positive tests/test_cell_body.py::test_face_volumes_ico -v
```

Expected: 4 passed

- [ ] **Step 1.6: Run full test suite to check no regressions**

```bash
uv run pytest -v
```

Expected: all previously passing tests still pass (120+4 total)

- [ ] **Step 1.7: Commit**

```bash
git add develmorph/cell_body.py tests/test_cell_body.py
git commit -m "feat: add face_volumes to CellBody and create_cell_body"
```

---

## Task 2: Add `TetrahedralVolumeConstraint` to `project.py`

**Files:**
- Modify: `develmorph/project.py`
- Test: `tests/test_project.py`

- [ ] **Step 2.1: Write failing unit tests**

Append to `tests/test_project.py`. First add the import at the top of the file alongside the existing imports:

```python
from develmorph.project import TetrahedralVolumeConstraint
```

Then append the test functions:

```python
def test_tet_volume_constraint_basic():
    # Tetrahedron: center at origin, ideal triangle at unit axes.
    # V_ideal = dot((1,0,0), cross((0,1,0),(0,0,1))) / 6 = 1/6
    x = np.array([
        [0.0, 0.0, 0.0],  # center (idx 0)
        [1.0, 0.0, 0.0],  # p0    (idx 1)
        [0.0, 1.0, 0.0],  # p1    (idx 2)
        [0.0, 0.0, 1.0],  # p2    (idx 3)
    ], dtype=float)
    inv_masses = np.ones(4)
    target_v = 1.0 / 6.0

    # Compress: move p0 inward so V < target
    x[1] = [0.5, 0.0, 0.0]
    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_before = np.dot(a, np.cross(b, c)) / 6.0
    assert v_before < target_v

    constraint = TetrahedralVolumeConstraint(0, 1, 2, 3, target_v, compliance=0.0)
    lambda_acc = 0.0
    for _ in range(20):
        lambda_acc = constraint.project(x, inv_masses, 0.01, lambda_acc)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_after = np.dot(a, np.cross(b, c)) / 6.0
    assert abs(v_after - target_v) < abs(v_before - target_v) * 0.1


def test_tet_volume_constraint_gradient():
    # Verify dV/dpc = -(g0 + g1 + g2) via numerical differentiation.
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    def vol(pos):
        a = pos[1] - pos[0]; b = pos[2] - pos[0]; c = pos[3] - pos[0]
        return np.dot(a, np.cross(b, c)) / 6.0

    eps = 1e-6
    num_grad_c = np.zeros(3)
    for d in range(3):
        xp = x.copy(); xp[0, d] += eps
        xm = x.copy(); xm[0, d] -= eps
        num_grad_c[d] = (vol(xp) - vol(xm)) / (2 * eps)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    g0 = np.cross(b, c) / 6.0
    g1 = np.cross(c, a) / 6.0
    g2 = np.cross(a, b) / 6.0
    gc_analytic = -(g0 + g1 + g2)

    np.testing.assert_allclose(gc_analytic, num_grad_c, atol=1e-5)


def test_tet_volume_constraint_equality():
    # With compliance=0, many iterations converge volume to target.
    x = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],  # compressed; ideal position is (1,0,0)
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    inv_masses = np.ones(4)
    target_v = 1.0 / 6.0

    constraint = TetrahedralVolumeConstraint(0, 1, 2, 3, target_v, compliance=0.0)
    lambda_acc = 0.0
    for _ in range(100):
        lambda_acc = constraint.project(x, inv_masses, 0.01, lambda_acc)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    V = np.dot(a, np.cross(b, c)) / 6.0
    assert abs(V - target_v) < 1e-6


def test_tet_volume_sign_below_target():
    # When volume is below target, a single project() call increases it.
    x = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],  # compressed
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    inv_masses = np.ones(4)
    target_v = 1.0 / 6.0

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_before = np.dot(a, np.cross(b, c)) / 6.0
    assert v_before < target_v

    constraint = TetrahedralVolumeConstraint(0, 1, 2, 3, target_v, compliance=0.0)
    constraint.project(x, inv_masses, 0.01, 0.0)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_after = np.dot(a, np.cross(b, c)) / 6.0
    assert v_after > v_before


def test_tet_volume_sign_above_target():
    # When volume exceeds target, a single project() call reduces it.
    x = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],  # expanded
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    inv_masses = np.ones(4)
    target_v = 1.0 / 6.0

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_before = np.dot(a, np.cross(b, c)) / 6.0
    assert v_before > target_v

    constraint = TetrahedralVolumeConstraint(0, 1, 2, 3, target_v, compliance=0.0)
    constraint.project(x, inv_masses, 0.01, 0.0)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_after = np.dot(a, np.cross(b, c)) / 6.0
    assert v_after < v_before
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_project.py::test_tet_volume_constraint_basic tests/test_project.py::test_tet_volume_constraint_gradient tests/test_project.py::test_tet_volume_constraint_equality tests/test_project.py::test_tet_volume_sign_below_target tests/test_project.py::test_tet_volume_sign_above_target -v
```

Expected: FAIL with `ImportError: cannot import name 'TetrahedralVolumeConstraint'`

- [ ] **Step 2.3: Implement `TetrahedralVolumeConstraint` in `project.py`**

In `develmorph/project.py`, append after the `TriangleAreaConstraint` class (after line 426):

```python


class TetrahedralVolumeConstraint:
    """
    XPBD equality constraint: volume of tetrahedron(center, p0, p1, p2) = target.

    Each surface triangle of a cell, together with the cell center, forms a
    tetrahedron. Preserving each tetrahedron's volume independently produces
    a local pressure model where cytoplasm pushes outward on each membrane patch.

    Constraint: C = V - target_volume = 0

    Volume formula:
        a = p0 - pc, b = p1 - pc, c = p2 - pc
        V = dot(a, cross(b, c)) / 6

    Gradients:
        dV/dp0 = cross(b, c) / 6
        dV/dp1 = cross(c, a) / 6
        dV/dp2 = cross(a, b) / 6
        dV/dpc = -(dV/dp0 + dV/dp1 + dV/dp2)
    """

    def __init__(
        self, center_idx: int, i0: int, i1: int, i2: int,
        target_volume: float, compliance: float = 0.0
    ):
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

- [ ] **Step 2.4: Run unit tests to confirm they pass**

```bash
uv run pytest tests/test_project.py::test_tet_volume_constraint_basic tests/test_project.py::test_tet_volume_constraint_gradient tests/test_project.py::test_tet_volume_constraint_equality tests/test_project.py::test_tet_volume_sign_below_target tests/test_project.py::test_tet_volume_sign_above_target -v
```

Expected: 5 passed

- [ ] **Step 2.5: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass

- [ ] **Step 2.6: Commit**

```bash
git add develmorph/project.py tests/test_project.py
git commit -m "feat: add TetrahedralVolumeConstraint to project.py"
```

---

## Task 3: New constraint path in `solver.step()`

**Files:**
- Modify: `develmorph/solver.py`
- Test: `tests/test_solver.py`

- [ ] **Step 3.1: Write failing integration tests**

Append to `tests/test_solver.py`. First, ensure `create_cell_body` is imported (it already is in this file — check the top; if not, add it). Also add a helper for measuring actual cell volume from live positions:

```python
def _actual_volume(positions, cb):
    total = 0.0
    for a, b, c in cb.triangles:
        p0 = positions[cb.surface_indices[a]] - positions[cb.center_idx]
        p1 = positions[cb.surface_indices[b]] - positions[cb.center_idx]
        p2 = positions[cb.surface_indices[c]] - positions[cb.center_idx]
        total += np.dot(p0, np.cross(p1, p2)) / 6.0
    return total


def test_step_tet_volume_single_cell_stable():
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=np.empty((0, 3)),
    )
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))
    target_vol = cb.target_volume

    for _ in range(100):
        positions, velocities = step(
            positions, velocities, inv_masses, dt=0.01,
            cell_bodies=[cb],
            shell_params=(10.0, 10.0, 10.0),
            n_iter=10,
            tet_volume_compliance=0.01,
            damping=1.0,
        )

    assert abs(_actual_volume(positions, cb) - target_vol) / target_vol < 0.05


def test_step_tet_volume_two_cell_collision():
    cb1, pos = create_cell_body(np.array([-0.5, 0.0, 0.0]), 1.0, np.empty((0, 3)))
    cb2, pos = create_cell_body(np.array([0.5, 0.0, 0.0]), 1.0, pos)
    velocities = np.zeros_like(pos)
    inv_masses = np.ones(len(pos))

    for _ in range(100):
        pos, velocities = step(
            pos, velocities, inv_masses, dt=0.01,
            cell_bodies=[cb1, cb2],
            shell_params=(10.0, 10.0, 10.0),
            n_iter=10,
            tet_volume_compliance=0.01,
            damping=1.0,
        )

    for cb in [cb1, cb2]:
        actual = _actual_volume(pos, cb)
        assert abs(actual - cb.target_volume) / cb.target_volume < 0.1


def test_step_backward_compat_no_tet_param():
    # tet_volume_compliance=None (default) must use old path without error.
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=np.empty((0, 3)),
    )
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, new_vel = step(
        positions, velocities, inv_masses, dt=0.01,
        cell_bodies=[cb],
        shell_params=(10.0, 10.0, 10.0),
        n_iter=5,
    )
    assert new_pos.shape == positions.shape
    assert new_vel.shape == velocities.shape
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_solver.py::test_step_tet_volume_single_cell_stable tests/test_solver.py::test_step_tet_volume_two_cell_collision tests/test_solver.py::test_step_backward_compat_no_tet_param -v
```

Expected: FAIL with `TypeError: step() got an unexpected keyword argument 'tet_volume_compliance'`

- [ ] **Step 3.3: Add `TetrahedralVolumeConstraint` import to `solver.py`**

In `develmorph/solver.py`, add to the import block (line 3–11):

```python
from develmorph.project import (
    AdhesionConstraint,
    CollisionConstraint,
    DistanceConstraint,
    HalfSpaceConstraint,
    ShellConstraint,
    TetrahedralVolumeConstraint,
    TriangleAreaConstraint,
    VolumeConstraint,
)
```

- [ ] **Step 3.4: Add `tet_volume_compliance=None` parameter to `step()`**

In `develmorph/solver.py`, in the `step()` signature (line 63), add `tet_volume_compliance=None` after `area_compliance=None`:

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
    tet_volume_compliance=None,
):
```

- [ ] **Step 3.5: Replace primary constraint building with branched logic**

In `develmorph/solver.py`, find the comment `# 2. Build constraints in priority order` (around line 114). Replace the block that builds `volume_constraints`, `distance_constraints`, and `area_constraints` with this branched version:

```python
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

            center_compliance = 0.001
            for (ci, si), rest_len in zip(cb.center_surface_edges, cb.center_surface_rest):
                distance_constraints.append(
                    DistanceConstraint(ci, si, rest_len, center_compliance)
                )

            for (a, b), rest_len in zip(cb.surface_surface_edges, cb.surface_surface_rest):
                distance_constraints.append(
                    DistanceConstraint(a, b, rest_len, membrane_compliance)
                )

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

- [ ] **Step 3.6: Extract shell constraint building to a standalone loop**

In `develmorph/solver.py`, the current `for cb in cell_bodies:` loop builds `volume_constraints`, `distance_constraints`, AND `shell_constraints` all together. After Step 3.5 replaces that loop, shell constraints would be missing unless moved.

After the branched block from Step 3.5, add a separate loop that builds shell constraints for ALL cells (runs regardless of which branch was taken):

```python
    shell_constraints = []
    for cb in cell_bodies:
        nominal_r = cb.nominal_radius()
        shell_constraints.append(
            ShellConstraint(cb.center_idx, *shell_params, nominal_r, shell_compliance)
        )
        for si in cb.surface_indices:
            shell_constraints.append(ShellConstraint(si, *shell_params, 0.0, shell_compliance))
```

Then remove the shell constraint building from whatever the old per-cell loop looked like. (Search for `ShellConstraint` in solver.py to confirm only one copy remains.)

- [ ] **Step 3.7: Update `all_constraints` assembly**

Find the line that assigns `all_constraints`. Replace it:

```python
    # Ordered: tet -> volume -> distance -> area -> adhesion -> collision
    #          -> surface_penetration -> surface_contact -> shell
    # tet and (volume+distance+area) are mutually exclusive (one will be empty).
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
```

- [ ] **Step 3.8: Run integration tests**

```bash
uv run pytest tests/test_solver.py::test_step_tet_volume_single_cell_stable tests/test_solver.py::test_step_tet_volume_two_cell_collision tests/test_solver.py::test_step_backward_compat_no_tet_param -v
```

Expected: 3 passed

- [ ] **Step 3.9: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass

- [ ] **Step 3.10: Commit**

```bash
git add develmorph/solver.py tests/test_solver.py
git commit -m "feat: add tet_volume_compliance path to solver.step()"
```



---

## Task 4: Thread `tet_volume_compliance` through `engine.World`

**Files:**
- Modify: `develmorph/engine.py`
- Test: `tests/test_engine.py`

- [ ] **Step 4.1: Write failing test**

Append to `tests/test_engine.py`:

```python
def test_world_tet_volume_compliance_wired():
    from develmorph.cell_body import create_cell_body
    from develmorph.engine import World
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=np.empty((0, 3)),
    )
    world = World([cb], positions, tet_volume_compliance=0.01, damping=1.0)
    # Must not raise; positions must update
    world.step()
    assert world.positions.shape == positions.shape
```

(Check `tests/test_engine.py` top for existing imports; `numpy as np` is likely already present. Add `import numpy as np` if absent.)

- [ ] **Step 4.2: Run test to confirm it fails**

```bash
uv run pytest tests/test_engine.py::test_world_tet_volume_compliance_wired -v
```

Expected: FAIL with `TypeError: World.__init__() got an unexpected keyword argument 'tet_volume_compliance'`

- [ ] **Step 4.3: Add `tet_volume_compliance` to `World.__init__()`**

In `develmorph/engine.py`, find the `World.__init__()` signature (line 38). Add `tet_volume_compliance=None` at the end:

```python
    def __init__(
        self, cell_bodies, positions, dt=0.01, shell_params=(5.0, 5.0, 5.0),
        lineage=None, n_iter=10, damping=0.0, shell_compliance=0.0,
        collision_compliance=0.0, membrane_compliance=0.001,
        volume_compliance=1.0, contact_overlap=0.0,
        surface_contact_compliance=None, area_compliance=None,
        tet_volume_compliance=None,
    ):
```

Add to the body (after `self.area_compliance = area_compliance`, line 60):

```python
        self.tet_volume_compliance = tet_volume_compliance
```

- [ ] **Step 4.4: Forward `tet_volume_compliance` in `World.step()`**

In `develmorph/engine.py`, in the `step()` method, find the `step(...)` call (line 149). Add `tet_volume_compliance=self.tet_volume_compliance` to the call:

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
            tet_volume_compliance=self.tet_volume_compliance,
        )
```

- [ ] **Step 4.5: Run test**

```bash
uv run pytest tests/test_engine.py::test_world_tet_volume_compliance_wired -v
```

Expected: 1 passed

- [ ] **Step 4.6: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests pass

- [ ] **Step 4.7: Commit**

```bash
git add develmorph/engine.py tests/test_engine.py
git commit -m "feat: thread tet_volume_compliance through World"
```

---

## Done

At this point:
- `TetrahedralVolumeConstraint` is implemented and unit-tested
- `CellBody.face_volumes` holds per-face target volumes from initial geometry
- `solver.step(tet_volume_compliance=0.01)` uses the new per-tetrahedron path
- `World(tet_volume_compliance=0.01)` wires the parameter end-to-end
- All 120+ existing tests still pass (backward compat confirmed)

**Next experiment:** run `scripts/demo_4cell_stage.py` with `tet_volume_compliance=0.01` and observe cell shape stability vs the spring-network baseline. If cells drift/destabilize, add weak center-surface DistanceConstraint as described in the spec's Fallback section.
