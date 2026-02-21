# XPBD Phase B: Surface Points + Volume Constraint — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend cells from single particles to octahedron surface models with incompressible volume constraint and deformable membrane.

**Architecture:** New `CellBody` class maps cell structure (center + 6 surface points) onto flat particle arrays. New `DistanceConstraint` and `VolumeConstraint` classes added to `project.py`. Solver rebuilt to construct constraints from `CellBody` list. Division creates daughter CellBodies with inherited volume.

**Tech Stack:** Python 3.13, NumPy, pytest, uv

---

### Task 1: CellBody — Failing Tests

**Files:**
- Create: `tests/test_cell_body.py`

**Step 1: Write failing tests for CellBody**

```python
import numpy as np
import math
from develmorph.cell_body import CellBody, create_cell_body, build_particle_arrays


def test_create_cell_body_particle_count():
    """CellBody has 1 center + 6 surface = 7 particles."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(cb.surface_indices) == 6
    assert len(positions) == 7  # center + 6 surface


def test_create_cell_body_octahedron_positions():
    """Surface points form an octahedron at the given radius."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([1.0, 2.0, 3.0]),
        radius=0.5,
        positions=positions,
    )
    center = positions[cb.center_idx]
    for si in cb.surface_indices:
        dist = np.linalg.norm(positions[si] - center)
        assert abs(dist - 0.5) < 1e-10


def test_create_cell_body_triangle_count():
    """Octahedron has 8 triangular faces."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(cb.triangles) == 8


def test_create_cell_body_target_volume():
    """Target volume equals 4/3 * pi * r^3."""
    positions = np.zeros((0, 3))
    r = 1.5
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=r,
        positions=positions,
    )
    expected = (4.0 / 3.0) * math.pi * r ** 3
    assert abs(cb.target_volume - expected) < 1e-10


def test_create_multiple_cells_indices():
    """Multiple cells get non-overlapping indices."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    cb2, positions = create_cell_body(
        center=np.array([3.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    all_indices_1 = {cb1.center_idx} | set(cb1.surface_indices)
    all_indices_2 = {cb2.center_idx} | set(cb2.surface_indices)
    assert len(all_indices_1 & all_indices_2) == 0  # no overlap
    assert len(positions) == 14  # 7 + 7


def test_build_particle_arrays():
    """build_particle_arrays reconstructs flat arrays and reassigns indices."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    cb2, positions = create_cell_body(
        center=np.array([3.0, 0.0, 0.0]),
        radius=0.5,
        positions=positions,
    )
    cell_bodies = [cb1, cb2]
    new_positions, new_inv_masses = build_particle_arrays(cell_bodies)
    assert len(new_positions) == 14
    assert len(new_inv_masses) == 14
    # Center of cell 1 at origin
    assert np.allclose(new_positions[cb1.center_idx], [0.0, 0.0, 0.0])
    # Center of cell 2 at (3, 0, 0)
    assert np.allclose(new_positions[cb2.center_idx], [3.0, 0.0, 0.0])


def test_cell_body_edge_pairs():
    """CellBody has 12 edges: 6 center-surface + 6 surface-surface."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(cb.center_surface_edges) == 6
    assert len(cb.surface_surface_edges) == 12  # octahedron has 12 edges
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cell_body.py -v`
Expected: FAIL (ImportError — cell_body module does not exist)

**Step 3: Commit failing tests**

```bash
git add tests/test_cell_body.py
git commit -m "test: add failing tests for CellBody"
```

---

### Task 2: CellBody — Implementation

**Files:**
- Create: `develmorph/cell_body.py`

**Step 1: Implement CellBody**

```python
import numpy as np
import math


# Octahedron: 6 vertices along +/-x, +/-y, +/-z axes
OCTAHEDRON_DIRS = np.array([
    [1.0, 0.0, 0.0],   # 0: +x
    [-1.0, 0.0, 0.0],  # 1: -x
    [0.0, 1.0, 0.0],   # 2: +y
    [0.0, -1.0, 0.0],  # 3: -y
    [0.0, 0.0, 1.0],   # 4: +z
    [0.0, 0.0, -1.0],  # 5: -z
])

# 8 triangular faces of the octahedron (indices into surface points 0-5)
# Oriented so normals point outward
OCTAHEDRON_TRIANGLES = [
    (0, 2, 4),  # +x, +y, +z
    (0, 4, 3),  # +x, +z, -y
    (0, 3, 5),  # +x, -y, -z
    (0, 5, 2),  # +x, -z, +y
    (1, 4, 2),  # -x, +z, +y
    (1, 3, 4),  # -x, -y, +z
    (1, 5, 3),  # -x, -z, -y
    (1, 2, 5),  # -x, +y, -z
]

# 12 edges of the octahedron (surface-surface)
OCTAHEDRON_EDGES = [
    (0, 2), (0, 3), (0, 4), (0, 5),  # +x to equator
    (1, 2), (1, 3), (1, 4), (1, 5),  # -x to equator
    (2, 4), (4, 3), (3, 5), (5, 2),  # equator ring
]


class CellBody:
    """
    A cell represented as center point + 6 octahedron surface points.

    All indices refer to positions in the flat particle array.
    """

    def __init__(self, center_idx, surface_indices, target_volume,
                 membrane_compliance=0.01):
        self.center_idx = center_idx
        self.surface_indices = list(surface_indices)
        self.triangles = OCTAHEDRON_TRIANGLES
        self.target_volume = target_volume
        self.membrane_compliance = membrane_compliance

        # Edges for distance constraints
        self.center_surface_edges = [
            (center_idx, si) for si in surface_indices
        ]
        self.surface_surface_edges = [
            (surface_indices[a], surface_indices[b])
            for a, b in OCTAHEDRON_EDGES
        ]

    def all_indices(self):
        """Return all particle indices belonging to this cell."""
        return [self.center_idx] + self.surface_indices

    def surface_global_triangles(self):
        """Return triangles with global particle indices."""
        return [
            (self.surface_indices[a], self.surface_indices[b], self.surface_indices[c])
            for a, b, c in self.triangles
        ]


def create_cell_body(center, radius, positions, membrane_compliance=0.01):
    """
    Create a CellBody and append its particles to the positions array.

    Args:
        center: (3,) center position
        radius: cell radius
        positions: existing (N, 3) positions array to append to
        membrane_compliance: membrane softness

    Returns:
        (cell_body, new_positions): CellBody and updated positions array
    """
    center = np.asarray(center, dtype=float)
    n = len(positions)

    # Center point
    center_idx = n
    new_points = [center]

    # Surface points
    surface_indices = []
    for i, d in enumerate(OCTAHEDRON_DIRS):
        surface_indices.append(n + 1 + i)
        new_points.append(center + radius * d)

    target_volume = (4.0 / 3.0) * math.pi * radius ** 3

    if len(positions) == 0:
        new_positions = np.array(new_points)
    else:
        new_positions = np.vstack([positions, new_points])

    cell_body = CellBody(
        center_idx=center_idx,
        surface_indices=surface_indices,
        target_volume=target_volume,
        membrane_compliance=membrane_compliance,
    )

    return cell_body, new_positions


def build_particle_arrays(cell_bodies):
    """
    Rebuild flat arrays from a list of CellBodies.

    Reassigns all particle indices sequentially.

    Returns:
        (positions, inv_masses): flat arrays
    """
    # This function is for rebuilding after division.
    # It assumes each CellBody already has valid positions stored externally.
    # We just need to return correct shapes and update indices.
    raise NotImplementedError("Will be implemented in Task 4 with division")
```

Wait — `build_particle_arrays` needs positions stored somewhere. Let me reconsider.

The function should take positions + cell_bodies and repack. But since CellBody only holds indices, we need the current positions array too.

**Revised `build_particle_arrays`:**

```python
def build_particle_arrays(cell_bodies, positions):
    """
    Rebuild flat arrays from a list of CellBodies using current positions.

    Repacks particles contiguously and reassigns all indices.

    Args:
        cell_bodies: list of CellBody
        positions: current (M, 3) positions array

    Returns:
        (new_positions, new_inv_masses): repacked flat arrays
    """
    new_positions = []
    new_inv_masses = []
    idx = 0

    for cb in cell_bodies:
        # Collect current positions for this cell
        old_center = positions[cb.center_idx]
        old_surfaces = [positions[si] for si in cb.surface_indices]

        # Reassign indices
        cb.center_idx = idx
        new_positions.append(old_center)
        new_inv_masses.append(1.0)
        idx += 1

        new_surface_indices = []
        for sp in old_surfaces:
            new_surface_indices.append(idx)
            new_positions.append(sp)
            new_inv_masses.append(1.0)
            idx += 1

        cb.surface_indices = new_surface_indices

        # Rebuild edge lists with new indices
        cb.center_surface_edges = [
            (cb.center_idx, si) for si in cb.surface_indices
        ]
        cb.surface_surface_edges = [
            (cb.surface_indices[a], cb.surface_indices[b])
            for a, b in OCTAHEDRON_EDGES
        ]

    return np.array(new_positions), np.array(new_inv_masses)
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_cell_body.py -v`
Expected: All 8 tests PASS

**Step 3: Commit**

```bash
git add develmorph/cell_body.py
git commit -m "feat: implement CellBody with octahedron surface points"
```

---

### Task 3: DistanceConstraint — Failing Tests

**Files:**
- Modify: `tests/test_project.py`

**Step 1: Add failing tests for DistanceConstraint**

Append to existing `tests/test_project.py`:

```python
from develmorph.project import DistanceConstraint


def test_distance_contracts_when_stretched():
    """Stretched pair is pulled back together."""
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    dist = np.linalg.norm(x[0] - x[1])
    assert abs(dist - 1.0) < 0.05


def test_distance_expands_when_compressed():
    """Compressed pair is pushed apart."""
    x = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    dist = np.linalg.norm(x[0] - x[1])
    assert abs(dist - 1.0) < 0.05


def test_distance_preserves_midpoint():
    """Equal-mass correction preserves midpoint."""
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c.project(x, inv_masses, dt, 0.0)

    midpoint = (x[0] + x[1]) / 2
    assert np.allclose(midpoint, [1.0, 0.0, 0.0])


def test_distance_compliance_allows_deviation():
    """Higher compliance allows more deviation from rest length."""
    x_hard = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    x_soft = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c_hard = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c_soft = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=1.0)

    c_hard.project(x_hard, inv_masses, dt, 0.0)
    c_soft.project(x_soft, inv_masses, dt, 0.0)

    err_hard = abs(np.linalg.norm(x_hard[0] - x_hard[1]) - 1.0)
    err_soft = abs(np.linalg.norm(x_soft[0] - x_soft[1]) - 1.0)

    # Hard constraint corrects more
    assert err_hard < err_soft
```

**Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_project.py::test_distance_contracts_when_stretched -v`
Expected: FAIL (ImportError — DistanceConstraint does not exist)

**Step 3: Commit**

```bash
git add tests/test_project.py
git commit -m "test: add failing tests for DistanceConstraint"
```

---

### Task 4: DistanceConstraint — Implementation

**Files:**
- Modify: `develmorph/project.py`

**Step 1: Add DistanceConstraint to project.py**

Append after ShellConstraint class:

```python
class DistanceConstraint:
    """
    XPBD equality constraint: |p_i - p_j| = rest_length.

    Unlike CollisionConstraint (inequality), this corrects in both directions:
    pushes apart when too close AND pulls together when too far.
    Used for membrane distance maintenance.
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

**Step 2: Run all constraint tests**

Run: `uv run pytest tests/test_project.py -v`
Expected: All 13 tests PASS (9 existing + 4 new)

**Step 3: Commit**

```bash
git add develmorph/project.py
git commit -m "feat: implement DistanceConstraint (equality) with XPBD"
```

---

### Task 5: VolumeConstraint — Failing Tests

**Files:**
- Modify: `tests/test_project.py`

**Step 1: Add failing tests for VolumeConstraint**

Append to `tests/test_project.py`:

```python
import math
from develmorph.project import VolumeConstraint
from develmorph.cell_body import create_cell_body


def _make_test_cell(center, radius):
    """Helper: create a single CellBody and return (cell_body, positions, inv_masses)."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(center=np.array(center), radius=radius, positions=positions)
    inv_masses = np.ones(len(positions))
    return cb, positions, inv_masses


def test_volume_no_correction_at_target():
    """No correction when volume equals target."""
    cb, x, inv_masses = _make_test_cell([0.0, 0.0, 0.0], 1.0)
    dt = 0.01

    c = VolumeConstraint(
        center_idx=cb.center_idx,
        surface_indices=cb.surface_indices,
        triangles=cb.surface_global_triangles(),
        target_volume=cb.target_volume,
        compliance=0.0,
    )
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    # Volume is already at target, so no significant movement
    assert np.allclose(x, x_before, atol=0.05)


def test_volume_inflates_when_deflated():
    """Deflated cell inflates back toward target volume."""
    cb, x, inv_masses = _make_test_cell([0.0, 0.0, 0.0], 1.0)
    dt = 0.01

    # Shrink surface points to 50% radius
    center = x[cb.center_idx]
    for si in cb.surface_indices:
        x[si] = center + (x[si] - center) * 0.5

    c = VolumeConstraint(
        center_idx=cb.center_idx,
        surface_indices=cb.surface_indices,
        triangles=cb.surface_global_triangles(),
        target_volume=cb.target_volume,
        compliance=0.0,
    )

    lambda_acc = 0.0
    for _ in range(20):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    # Surface points should have moved outward
    for si in cb.surface_indices:
        dist = np.linalg.norm(x[si] - center)
        assert dist > 0.5  # expanded from 0.5


def test_volume_deflates_when_inflated():
    """Inflated cell deflates back toward target volume."""
    cb, x, inv_masses = _make_test_cell([0.0, 0.0, 0.0], 1.0)
    dt = 0.01

    # Expand surface points to 200% radius
    center = x[cb.center_idx]
    for si in cb.surface_indices:
        x[si] = center + (x[si] - center) * 2.0

    c = VolumeConstraint(
        center_idx=cb.center_idx,
        surface_indices=cb.surface_indices,
        triangles=cb.surface_global_triangles(),
        target_volume=cb.target_volume,
        compliance=0.0,
    )

    lambda_acc = 0.0
    for _ in range(20):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    # Surface points should have moved inward
    for si in cb.surface_indices:
        dist = np.linalg.norm(x[si] - center)
        assert dist < 2.0  # contracted from 2.0


def test_volume_bulge_when_pushed():
    """When pushed from one side, cell bulges on the other side to conserve volume."""
    cb, x, inv_masses = _make_test_cell([0.0, 0.0, 0.0], 1.0)
    dt = 0.01

    # Push +x surface point inward (simulate external force)
    x[cb.surface_indices[0]] = np.array([0.3, 0.0, 0.0])  # was [1,0,0]

    c = VolumeConstraint(
        center_idx=cb.center_idx,
        surface_indices=cb.surface_indices,
        triangles=cb.surface_global_triangles(),
        target_volume=cb.target_volume,
        compliance=0.0,
    )

    lambda_acc = 0.0
    for _ in range(20):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    # Other surface points should have moved outward to compensate
    # Check -x point (index 1) moved further from center
    dist_neg_x = np.linalg.norm(x[cb.surface_indices[1]] - x[cb.center_idx])
    assert dist_neg_x > 1.0  # bulged outward from original 1.0
```

**Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_project.py::test_volume_no_correction_at_target -v`
Expected: FAIL (ImportError — VolumeConstraint does not exist)

**Step 3: Commit**

```bash
git add tests/test_project.py
git commit -m "test: add failing tests for VolumeConstraint"
```

---

### Task 6: VolumeConstraint — Implementation

**Files:**
- Modify: `develmorph/project.py`

**Step 1: Add VolumeConstraint to project.py**

Append after DistanceConstraint class:

```python
class VolumeConstraint:
    """
    XPBD equality constraint: mesh volume = target_volume.

    Uses signed volume of triangulated surface mesh:
        V = (1/6) * sum( dot(p0, cross(p1, p2)) )

    Compliance ~= 0 for incompressible (liquid cell interior).
    Modifies positions in-place.
    """

    def __init__(self, center_idx, surface_indices, triangles,
                 target_volume, compliance=0.0):
        self.center_idx = center_idx
        self.surface_indices = list(surface_indices)
        self.triangles = triangles  # list of (i, j, k) global indices
        self.target_volume = target_volume
        self.compliance = compliance
        # All particle indices affected by this constraint
        self.all_indices = [center_idx] + list(surface_indices)

    def _compute_volume(self, x):
        """Compute signed volume of the triangulated mesh."""
        vol = 0.0
        for i0, i1, i2 in self.triangles:
            vol += np.dot(x[i0], np.cross(x[i1], x[i2]))
        return vol / 6.0

    def _compute_gradients(self, x):
        """
        Compute gradient of volume w.r.t. each particle position.

        Returns dict: particle_index -> gradient (3,)
        """
        grads = {}
        for idx in self.all_indices:
            grads[idx] = np.zeros(3)

        for i0, i1, i2 in self.triangles:
            # dV/dp0 = (1/6) * cross(p1, p2)
            # dV/dp1 = (1/6) * cross(p2, p0)
            # dV/dp2 = (1/6) * cross(p0, p1)
            grads[i0] += np.cross(x[i1], x[i2]) / 6.0
            grads[i1] += np.cross(x[i2], x[i0]) / 6.0
            grads[i2] += np.cross(x[i0], x[i1]) / 6.0

        return grads

    def project(self, x, inv_masses, dt, lambda_acc):
        V = self._compute_volume(x)
        C = V - self.target_volume

        if abs(C) < 1e-12:
            return lambda_acc

        grads = self._compute_gradients(x)

        # Weighted gradient norm squared: sum(w_i * |grad_i|^2)
        w_grad_sq = 0.0
        for idx in self.all_indices:
            w_grad_sq += inv_masses[idx] * np.dot(grads[idx], grads[idx])

        if w_grad_sq < 1e-12:
            return lambda_acc

        alpha_tilde = self.compliance / (dt * dt) if dt > 0 else 0.0

        denom = w_grad_sq + alpha_tilde
        if denom < 1e-12:
            return lambda_acc

        delta_lambda = (-C - alpha_tilde * lambda_acc) / denom
        lambda_acc += delta_lambda

        for idx in self.all_indices:
            x[idx] += delta_lambda * inv_masses[idx] * grads[idx]

        return lambda_acc
```

**Step 2: Run all constraint tests**

Run: `uv run pytest tests/test_project.py -v`
Expected: All 17 tests PASS (13 existing + 4 new)

**Step 3: Commit**

```bash
git add develmorph/project.py
git commit -m "feat: implement VolumeConstraint with XPBD"
```

---

### Task 7: Solver Rewrite — Failing Tests

**Files:**
- Modify: `tests/test_solver.py`

**Step 1: Add surface-model solver tests**

Append to existing `tests/test_solver.py`:

```python
from develmorph.cell_body import create_cell_body


def test_solver_surface_cell_maintains_volume():
    """Cell with surface points maintains volume after solver step."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions, velocities=velocities, inv_masses=inv_masses,
        dt=0.01, cell_bodies=[cb], shell_params=(5.0, 5.0, 5.0),
    )

    # Compute volume after solver step
    vol = 0.0
    for i0, i1, i2 in cb.surface_global_triangles():
        vol += np.dot(new_pos[i0], np.cross(new_pos[i1], new_pos[i2]))
    vol /= 6.0

    assert abs(vol - cb.target_volume) / cb.target_volume < 0.1


def test_solver_two_surface_cells_separate():
    """Two overlapping surface-model cells are pushed apart."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=0.5,
        positions=positions,
    )
    cb2, positions = create_cell_body(
        center=np.array([0.5, 0.0, 0.0]),
        radius=0.5,
        positions=positions,
    )
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions, velocities=velocities, inv_masses=inv_masses,
        dt=0.01, cell_bodies=[cb1, cb2], shell_params=(5.0, 5.0, 5.0),
    )

    dist = np.linalg.norm(new_pos[cb1.center_idx] - new_pos[cb2.center_idx])
    assert dist > 0.5  # pushed apart


def test_solver_surface_cell_deforms_on_collision():
    """Cell deforms (surface points shift) when pushed against shell boundary."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([4.0, 0.0, 0.0]),  # close to shell boundary
        radius=1.0,
        positions=positions,
    )
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions, velocities=velocities, inv_masses=inv_masses,
        dt=0.01, cell_bodies=[cb], shell_params=(5.0, 5.0, 5.0),
        n_iter=20,
    )

    # +x surface point should have been pushed back (closer to center)
    center = new_pos[cb.center_idx]
    dist_plus_x = np.linalg.norm(new_pos[cb.surface_indices[0]] - center)
    dist_minus_x = np.linalg.norm(new_pos[cb.surface_indices[1]] - center)

    # +x side compressed, -x side same or expanded
    assert dist_plus_x < 1.0 or dist_minus_x > 1.0
```

**Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_solver.py::test_solver_surface_cell_maintains_volume -v`
Expected: FAIL (step() got unexpected keyword argument 'cell_bodies')

**Step 3: Commit**

```bash
git add tests/test_solver.py
git commit -m "test: add failing tests for surface-model XPBD solver"
```

---

### Task 8: Solver Rewrite — Implementation

**Files:**
- Modify: `develmorph/solver.py`

**Step 1: Rewrite solver to support CellBody**

```python
import numpy as np
from develmorph.project import (
    CollisionConstraint,
    ShellConstraint,
    DistanceConstraint,
    VolumeConstraint,
)


def _max_surface_dist(positions, cb):
    """Max distance from center to any surface point."""
    center = positions[cb.center_idx]
    return max(
        np.linalg.norm(positions[si] - center)
        for si in cb.surface_indices
    )


def step(positions, velocities, inv_masses, dt,
         cell_bodies, shell_params, n_iter=10,
         membrane_compliance=0.01, volume_compliance=0.0):
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

    Returns:
        (new_positions, new_velocities)
    """
    x_prev = positions.copy()

    # 1. Predict positions
    x = positions + dt * velocities

    # 2. Build constraints in priority order
    volume_constraints = []
    distance_constraints = []
    collision_constraints = []
    shell_constraints = []

    for cb in cell_bodies:
        # Volume constraint (per cell)
        volume_constraints.append(VolumeConstraint(
            center_idx=cb.center_idx,
            surface_indices=cb.surface_indices,
            triangles=cb.surface_global_triangles(),
            target_volume=cb.target_volume,
            compliance=volume_compliance,
        ))

        # Distance constraints: center-surface
        center = x[cb.center_idx]
        for si in cb.surface_indices:
            rest_len = np.linalg.norm(positions[si] - positions[cb.center_idx])
            distance_constraints.append(
                DistanceConstraint(cb.center_idx, si, rest_len, membrane_compliance)
            )

        # Distance constraints: surface-surface (octahedron edges)
        for a, b in cb.surface_surface_edges:
            rest_len = np.linalg.norm(positions[a] - positions[b])
            distance_constraints.append(
                DistanceConstraint(a, b, rest_len, membrane_compliance)
            )

        # Shell constraints: center + surface
        max_r = _max_surface_dist(positions, cb)
        shell_constraints.append(
            ShellConstraint(cb.center_idx, *shell_params, max_r, 0.0)
        )
        for si in cb.surface_indices:
            shell_constraints.append(
                ShellConstraint(si, *shell_params, 0.0, 0.0)
            )

    # Collision constraints: center-to-center between cells
    for i in range(len(cell_bodies)):
        for j in range(i + 1, len(cell_bodies)):
            ci = cell_bodies[i]
            cj = cell_bodies[j]
            ri = _max_surface_dist(positions, ci)
            rj = _max_surface_dist(positions, cj)
            collision_constraints.append(
                CollisionConstraint(ci.center_idx, cj.center_idx, ri + rj, 0.0)
            )

    # Ordered constraint list
    all_constraints = (
        volume_constraints +
        distance_constraints +
        collision_constraints +
        shell_constraints
    )

    # 3. XPBD iteration
    lambdas = [0.0] * len(all_constraints)
    for _ in range(n_iter):
        for k, c in enumerate(all_constraints):
            lambdas[k] = c.project(x, inv_masses, dt, lambdas[k])

    # 4. Update velocities
    new_vel = (x - x_prev) / dt if dt > 0 else np.zeros_like(x)

    return x, new_vel
```

**Step 2: Run all solver tests**

Run: `uv run pytest tests/test_solver.py -v`
Expected: All 7 tests PASS (4 old + 3 new)

Note: The 4 old tests use `radii` argument which is no longer in the signature. They need updating first. See Step 2b.

**Step 2b: Update old solver tests to use cell_bodies**

The old tests in `test_solver.py` used `radii=...` which no longer exists. Replace the 4 old tests:

```python
import numpy as np
from develmorph.solver import step
from develmorph.cell_body import create_cell_body


def test_solver_separates_overlapping_cells():
    """Two overlapping cells are pushed apart by XPBD solver."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions)
    cb2, positions = create_cell_body(center=np.array([0.4, 0.0, 0.0]), radius=0.5, positions=positions)
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, new_vel = step(
        positions=positions, velocities=vel, inv_masses=inv_masses,
        dt=0.01, cell_bodies=[cb1, cb2], shell_params=(5.0, 5.0, 5.0),
    )

    dist = np.linalg.norm(new_pos[cb1.center_idx] - new_pos[cb2.center_idx])
    assert dist > 0.4  # pushed apart


def test_solver_preserves_separated_cells():
    """Already-separated cells are not moved significantly."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions)
    cb2, positions = create_cell_body(center=np.array([5.0, 0.0, 0.0]), radius=0.5, positions=positions)
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions, velocities=vel, inv_masses=inv_masses,
        dt=0.01, cell_bodies=[cb1, cb2], shell_params=(10.0, 10.0, 10.0),
    )

    assert np.allclose(new_pos[cb1.center_idx], [0.0, 0.0, 0.0], atol=0.1)
    assert np.allclose(new_pos[cb2.center_idx], [5.0, 0.0, 0.0], atol=0.1)


def test_solver_enforces_shell():
    """Cell outside shell is pushed back inside."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(center=np.array([10.0, 0.0, 0.0]), radius=0.5, positions=positions)
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step(
        positions=positions, velocities=vel, inv_masses=inv_masses,
        dt=0.01, cell_bodies=[cb], shell_params=(5.0, 5.0, 5.0),
        n_iter=20,
    )

    ea = 5.0 - 0.5
    val = (new_pos[cb.center_idx][0] / ea) ** 2
    assert val <= 1.5  # moved toward inside


def test_solver_updates_velocity():
    """Velocity is updated to reflect position correction."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(center=np.array([0.0, 0.0, 0.0]), radius=0.5, positions=positions)
    cb2, positions = create_cell_body(center=np.array([0.4, 0.0, 0.0]), radius=0.5, positions=positions)
    vel = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    _, new_vel = step(
        positions=positions, velocities=vel, inv_masses=inv_masses,
        dt=0.01, cell_bodies=[cb1, cb2], shell_params=(5.0, 5.0, 5.0),
    )

    assert not np.allclose(new_vel, 0.0)
```

**Step 3: Run full test suite**

Run: `uv run pytest tests/test_solver.py tests/test_project.py tests/test_cell_body.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add develmorph/solver.py tests/test_solver.py
git commit -m "feat: rewrite XPBD solver for surface-point cell model"
```

---

### Task 9: Division Rewrite — Failing Tests

**Files:**
- Modify: `tests/test_division.py`

**Step 1: Add tests for CellBody-based division**

Replace `tests/test_division.py` contents:

```python
import numpy as np
import math
from develmorph.division import divide_cell_body
from develmorph.cell_body import create_cell_body, build_particle_arrays


def test_division_produces_two_daughters():
    """Division creates exactly 2 daughter CellBodies."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions,
    )

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]),
    )

    assert len(daughters) == 2


def test_division_conserves_volume():
    """Daughter volumes sum to parent volume."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions,
    )
    parent_vol = parent.target_volume

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]),
    )

    daughter_vol = sum(d.target_volume for d in daughters)
    assert abs(daughter_vol - parent_vol) < 1e-10


def test_division_asymmetric_volume():
    """Asymmetric division distributes volume correctly."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions,
    )

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]),
        volume_ratio=0.6,
    )

    ratio = daughters[0].target_volume / (daughters[0].target_volume + daughters[1].target_volume)
    assert abs(ratio - 0.6) < 1e-10


def test_division_daughters_along_axis():
    """Daughter centers are offset along the division axis."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions,
    )

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]),
    )

    c1 = positions[daughters[0].center_idx]
    c2 = positions[daughters[1].center_idx]

    # Daughter 1 in +x, daughter 2 in -x
    assert c1[0] > 0
    assert c2[0] < 0


def test_division_particle_count():
    """Division replaces 7 particles with 14."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions,
    )
    assert len(positions) == 7

    daughters, positions = divide_cell_body(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]),
    )

    assert len(positions) == 14
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_division.py -v`
Expected: FAIL (ImportError — divide_cell_body does not exist)

**Step 3: Commit**

```bash
git add tests/test_division.py
git commit -m "test: add failing tests for CellBody-based division"
```

---

### Task 10: Division Rewrite — Implementation

**Files:**
- Modify: `develmorph/division.py`

**Step 1: Rewrite division.py for CellBody**

```python
import numpy as np
import math
from develmorph.cell_body import CellBody, create_cell_body, OCTAHEDRON_DIRS


def divide_cell_body(parent, positions, axis=None, volume_ratio=0.5):
    """
    Divide a CellBody into two daughters.

    Removes parent particles and creates two new CellBodies with
    octahedron surface points.

    Args:
        parent: CellBody to divide
        positions: (N, 3) particle positions array
        axis: division axis (unit vector). Random if None.
        volume_ratio: fraction of volume for first daughter (0.0-1.0)

    Returns:
        (daughters, new_positions): list of 2 CellBodies and updated positions
    """
    if axis is None:
        axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    parent_center = positions[parent.center_idx].copy()
    parent_volume = parent.target_volume
    parent_radius = (3.0 * parent_volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    # Daughter radii from volume ratio
    r1 = parent_radius * (volume_ratio ** (1.0 / 3.0))
    r2 = parent_radius * ((1.0 - volume_ratio) ** (1.0 / 3.0))

    # Daughter centers along axis
    offset1 = 0.6 * r1
    offset2 = 0.6 * r2
    center1 = parent_center + offset1 * axis
    center2 = parent_center - offset2 * axis

    # Remove parent particles from positions array
    parent_indices = sorted(parent.all_indices(), reverse=True)
    positions = np.delete(positions, parent_indices, axis=0)

    # Create daughters (appending to cleaned positions)
    d1, positions = create_cell_body(
        center=center1, radius=r1, positions=positions,
        membrane_compliance=parent.membrane_compliance,
    )
    d2, positions = create_cell_body(
        center=center2, radius=r2, positions=positions,
        membrane_compliance=parent.membrane_compliance,
    )

    return [d1, d2], positions
```

**Step 2: Run division tests**

Run: `uv run pytest tests/test_division.py -v`
Expected: All 5 tests PASS

**Step 3: Commit**

```bash
git add develmorph/division.py
git commit -m "feat: rewrite cell division for CellBody surface model"
```

---

### Task 11: Engine Integration

**Files:**
- Modify: `develmorph/engine.py`

**Step 1: Rewrite engine for CellBody model**

```python
import numpy as np

from develmorph.cell_body import create_cell_body, build_particle_arrays
from develmorph.division import divide_cell_body
from develmorph.polarity import PolarityField
from develmorph.solver import step


class World:

    def __init__(self, cell_bodies, positions, dt=0.01,
                 shell_params=(5.0, 5.0, 5.0)):

        self.cell_bodies = list(cell_bodies)
        self.positions = positions.copy()

        self.velocities = np.zeros_like(positions)
        self.inv_masses = np.ones(len(positions))

        self.dt = dt
        self.time = 0
        self.step_count = 0

        self.shell_params = shell_params

        self.division_steps = []

        self.polarity = PolarityField()
        centers = np.array([positions[cb.center_idx] for cb in cell_bodies])
        self.cell_polarity = self.polarity.value(centers)

    def set_divisions(self, steps):
        self.division_steps = steps

    def step(self):

        # Events
        if self.step_count in self.division_steps:
            # Divide largest cell (by target volume)
            idx = max(range(len(self.cell_bodies)),
                      key=lambda i: self.cell_bodies[i].target_volume)
            parent = self.cell_bodies[idx]

            daughters, self.positions = divide_cell_body(
                parent, self.positions,
            )

            # Replace parent with daughters in cell list
            self.cell_bodies.pop(idx)
            self.cell_bodies.extend(daughters)

            # Rebuild flat arrays
            self.positions, self.inv_masses = build_particle_arrays(
                self.cell_bodies, self.positions,
            )
            self.velocities = np.zeros_like(self.positions)

            # Update polarity
            centers = np.array([
                self.positions[cb.center_idx] for cb in self.cell_bodies
            ])
            self.cell_polarity = self.polarity.value(centers)

        # Physics
        self.positions, self.velocities = step(
            self.positions, self.velocities,
            self.inv_masses, self.dt,
            self.cell_bodies, self.shell_params,
        )

        centers = np.array([
            self.positions[cb.center_idx] for cb in self.cell_bodies
        ])
        self.cell_polarity = self.polarity.value(centers)

        self.step_count += 1
        self.time += self.dt

    def run(self, n_steps):

        traj = []

        for _ in range(n_steps):

            self.step()

            traj.append(self.positions.copy())

        return traj
```

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add develmorph/engine.py
git commit -m "feat: integrate CellBody model into World engine"
```

---

### Task 12: Update Demo Script

**Files:**
- Modify: `scripts/demo_orientation_division.py`

**Step 1: Update demo to use CellBody model**

Key changes:
- Import `create_cell_body` instead of using raw positions
- `apply_constraints` passes `cell_bodies` to solver
- Track `cell_bodies` list alongside `cells` dict
- Division creates CellBody daughters

The demo script uses `Cell` objects (from `cell.py`) for lineage/orientation tracking AND now needs `CellBody` for physics. These are parallel data structures: `Cell` holds biological state (name, orientation, lineage), `CellBody` holds physical state (particles, volume).

```python
"""
Phase2-B demonstration: Orientation-based cell division with eggshell.
Uses XPBD solver with surface-point cell model.
"""

import numpy as np
from develmorph.cell import Cell
from develmorph.cell_body import create_cell_body, build_particle_arrays
from develmorph.celllineage import CellLineage
from develmorph.solver import step as xpbd_step
from develmorph.viz import visualize_cells_3d, visualize_timelapse_napari


EGGSHELL_A = 2.0
EGGSHELL_B = 1.0
EGGSHELL_C = 1.0
INITIAL_RADIUS = 0.9


def divide_cell(parent: Cell, rule, lineage: CellLineage, next_id: int):
    """Divide parent cell into two daughters (biological state only)."""
    axis = lineage.resolve_division_axis(rule, parent.orientation)
    dist = parent.radius * 0.6
    new_radius = parent.radius * Cell.DIVISION_RADIUS_FACTOR

    pos_a = parent.position + dist * axis
    pos_b = parent.position - dist * axis

    daughter_a = Cell(
        cell_id=next_id, name=rule.daughters[0],
        position=pos_a, radius=new_radius,
        orientation=parent.orientation.copy(),
    )
    daughter_b = Cell(
        cell_id=next_id + 1, name=rule.daughters[1],
        position=pos_b, radius=new_radius,
        orientation=parent.orientation.copy(),
    )
    return daughter_a, daughter_b


def apply_constraints(cells, cell_bodies, positions, n_iter=10):
    """Apply XPBD constraints with surface-point model."""
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    positions, _ = xpbd_step(
        positions, velocities, inv_masses,
        dt=0.01,
        cell_bodies=cell_bodies,
        shell_params=(EGGSHELL_A, EGGSHELL_B, EGGSHELL_C),
        n_iter=n_iter,
    )

    # Update Cell positions from CellBody centers
    cell_list = list(cells.values())
    for i, cell in enumerate(cell_list):
        cell.position = positions[cell_bodies[i].center_idx].copy()

    return positions


def main():
    lineage = CellLineage("develmorph/lineages/celegans_early.yaml")

    next_id = 0
    p0 = Cell(cell_id=next_id, name="P0", position=[0, 0, 0], radius=INITIAL_RADIUS)
    next_id += 1

    cells = {"P0": p0}

    # Create CellBody for P0
    positions = np.zeros((0, 3))
    cb_p0, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=INITIAL_RADIUS,
        positions=positions,
    )
    cell_bodies = [cb_p0]
    cell_body_map = {"P0": cb_p0}

    history = []
    history.append({name: Cell(c.cell_id, c.name, c.position.copy(), c.radius, c.orientation.copy())
                    for name, c in cells.items()})

    print(f"Eggshell: a={EGGSHELL_A}, b={EGGSHELL_B}, c={EGGSHELL_C}")
    print(f"Initial radius: {INITIAL_RADIUS}")
    print()

    step_num = 1
    while True:
        program = lineage.get_step_program(step_num)
        if program is None:
            break

        divisions_to_execute = [
            rule for rule in program.divisions
            if rule.parent in cells
        ]
        if not divisions_to_execute:
            break

        print(f"Step {step_num}:")

        for rule in divisions_to_execute:
            parent = cells[rule.parent]
            parent_cb = cell_body_map[rule.parent]

            a, b = divide_cell(parent, rule, lineage, next_id)
            next_id += 2

            print(f"  {rule.parent} -> {a.name}, {b.name} (axis: {rule.axis})")

            # Remove parent
            del cells[rule.parent]
            del cell_body_map[rule.parent]
            cell_bodies.remove(parent_cb)

            # Remove parent particles
            parent_indices = sorted(parent_cb.all_indices(), reverse=True)
            positions = np.delete(positions, parent_indices, axis=0)

            # Create daughter CellBodies
            cb_a, positions = create_cell_body(
                center=a.position, radius=a.radius, positions=positions,
            )
            cb_b, positions = create_cell_body(
                center=b.position, radius=b.radius, positions=positions,
            )

            cells[a.name] = a
            cells[b.name] = b
            cell_bodies.extend([cb_a, cb_b])
            cell_body_map[a.name] = cb_a
            cell_body_map[b.name] = cb_b

        # Rebuild indices after all divisions
        positions, _ = build_particle_arrays(cell_bodies, positions)

        # Apply constraints
        positions = apply_constraints(cells, cell_bodies, positions)

        print(f"  After constraints: {len(cells)} cells")
        for name, cell in sorted(cells.items()):
            print(f"    {name}: r={cell.radius:.3f}, pos={cell.position.round(3)}")

        history.append({name: Cell(c.cell_id, c.name, c.position.copy(), c.radius, c.orientation.copy())
                        for name, c in cells.items()})

        print()
        step_num += 1

    print(f"Simulation ended at step {step_num}")
    print(f"Final cells ({len(cells)}):")
    for name, cell in sorted(cells.items()):
        print(f"  {cell}")

    visualize_timelapse_napari(
        history,
        eggshell=(EGGSHELL_A, EGGSHELL_B, EGGSHELL_C),
    )


if __name__ == "__main__":
    main()
```

**Step 2: Run demo to verify**

Run: `uv run python scripts/demo_orientation_division.py`
Expected: Simulation runs, cells divide with volume conservation

**Step 3: Commit**

```bash
git add scripts/demo_orientation_division.py
git commit -m "refactor: update demo for CellBody surface-point model"
```

---

### Task 13: Final Verification

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 2: Commit**

```bash
git add -A
git commit -m "chore: Phase B complete — surface points + volume constraint"
```
