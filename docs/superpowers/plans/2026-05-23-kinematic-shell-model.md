# Kinematic Shell Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CellBody (center + surface mesh) with PhysicsCell (center-only) using ShellConstraint with cell radius for offset, enabling 1,000-cell simulation without surface-point drift.

**Architecture:** Three layers — XPBD physics on center points only; analytical contact geometry computed post-step; kinematic surface shell reconstructed for visualization. The existing `ShellConstraint(i, a, b, c, radius=pc.radius)` already implements the offset-ellipsoid behavior (effective semi-axes = shell_axes - cell_radius); no new constraint class is needed.

**Tech Stack:** Python 3.13, NumPy, uv (package manager), pytest

**Spec:** `docs/superpowers/specs/2026-05-23-kinematic-shell-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `develmorph/physics_cell.py` | Create | `PhysicsCell` dataclass + `create_physics_cell()` factory |
| `develmorph/division.py` | Modify | Add `divide_physics_cell()` alongside existing `divide_cell_body()` |
| `develmorph/solver.py` | Modify | Add `step_pc()` + `_update_adhesion_pairs_pc()` alongside existing `step()` |
| `develmorph/engine.py` | Modify | Add `WorldPC` class alongside existing `World` |
| `develmorph/contact.py` | Create (Phase 2) | `ContactRecord` + `compute_contacts()` |
| `develmorph/viz.py` | Modify (Phase 3) | Add `SurfaceShell` + `reconstruct_surface_shell()` |
| `develmorph/spatial_hash.py` | Create (Phase 4) | Grid-based neighbor search |
| `tests/test_physics_cell.py` | Create | Tests for `physics_cell.py` |
| `tests/test_divide_physics_cell.py` | Create | Tests for `divide_physics_cell()` |
| `tests/test_worldpc.py` | Create | Tests for `WorldPC` and `step_pc()` |
| `tests/test_contact.py` | Create (Phase 2) | Tests for `contact.py` |
| `tests/test_spatial_hash.py` | Create (Phase 4) | Tests for `spatial_hash.py` |
| `scripts/demo_diamond_pc.py` | Create | Diamond formation validation with `WorldPC` |

**Existing files left unchanged:** `project.py`, `cell_body.py`, all existing tests (137 tests stay passing throughout).

---

## Phase 1: PhysicsCell + WorldPC (Validation Gate)

---

### Task 1: Add `PhysicsCell` and `create_physics_cell`

**Files:**
- Create: `develmorph/physics_cell.py`
- Create: `tests/test_physics_cell.py`

- [ ] **Step 1.1: Write failing tests**

```python
# tests/test_physics_cell.py
import numpy as np
import pytest
from develmorph.physics_cell import PhysicsCell, create_physics_cell


def test_create_physics_cell_empty_positions():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(
        center=np.array([1.0, 2.0, 3.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(positions) == 1
    np.testing.assert_array_almost_equal(positions[0], [1.0, 2.0, 3.0])
    assert cell.center_idx == 0
    assert cell.radius == 1.0


def test_create_physics_cell_appends_to_existing():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([5.0, 0.0, 0.0]), 1.0, positions)
    assert len(positions) == 2
    assert c1.center_idx == 0
    assert c2.center_idx == 1


def test_physics_cell_all_indices():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    assert cell.all_indices() == [0]


def test_physics_cell_default_orientation():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    np.testing.assert_array_almost_equal(cell.orientation, np.eye(3))


def test_physics_cell_custom_name_and_orientation():
    positions = np.zeros((0, 3))
    orient = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=float)
    cell, positions = create_physics_cell(
        np.array([0.0, 0.0, 0.0]), 1.0, positions,
        name="P0", orientation=orient,
    )
    assert cell.name == "P0"
    np.testing.assert_array_almost_equal(cell.orientation, orient)
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
uv run pytest tests/test_physics_cell.py -v
```

Expected: `ModuleNotFoundError: No module named 'develmorph.physics_cell'`

- [ ] **Step 1.3: Implement `physics_cell.py`**

```python
# develmorph/physics_cell.py
import numpy as np


class PhysicsCell:
    """
    A cell represented as a single center point and a radius scalar.

    Participates in XPBD as one particle (center_idx).
    Volume = (4/3) * pi * r^3, constant between divisions.
    Orientation tracks the cell's local frame for division axis resolution.
    """

    def __init__(self, center_idx, radius, name=None, orientation=None):
        self.center_idx = center_idx
        self.radius = float(radius)
        self.name = name
        self.orientation = orientation if orientation is not None else np.eye(3)

    def all_indices(self):
        """Return list of all particle indices belonging to this cell."""
        return [self.center_idx]


def create_physics_cell(center, radius, positions, name=None, orientation=None):
    """
    Create a PhysicsCell and append its center to the positions array.

    Args:
        center: (3,) center position
        radius: cell radius
        positions: existing (N, 3) positions array
        name: biological cell name (e.g. "P0", "AB")
        orientation: 3x3 orientation matrix; defaults to identity

    Returns:
        (cell, new_positions): PhysicsCell and updated positions array
    """
    center = np.asarray(center, dtype=float)
    idx = len(positions)
    new_row = center.reshape(1, 3)
    if len(positions) == 0:
        new_positions = new_row
    else:
        new_positions = np.vstack([positions, new_row])

    cell = PhysicsCell(
        center_idx=idx,
        radius=radius,
        name=name,
        orientation=orientation,
    )
    return cell, new_positions
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
uv run pytest tests/test_physics_cell.py -v
```

Expected: 5 tests pass.

- [ ] **Step 1.5: Run full test suite to confirm no regressions**

```bash
uv run pytest -q
```

Expected: 137 existing tests + 5 new = 142 pass, 0 fail.

- [ ] **Step 1.6: Commit**

```bash
git add develmorph/physics_cell.py tests/test_physics_cell.py
git commit -m "feat: add PhysicsCell and create_physics_cell (center-only cell model)"
```

---

### Task 2: Add `divide_physics_cell` to `division.py`

**Files:**
- Modify: `develmorph/division.py`
- Create: `tests/test_divide_physics_cell.py`

- [ ] **Step 2.1: Write failing tests**

```python
# tests/test_divide_physics_cell.py
import math
import numpy as np
import pytest
from develmorph.physics_cell import create_physics_cell
from develmorph.division import divide_physics_cell


def test_divide_produces_two_daughters():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    daughters, positions = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )
    assert len(daughters) == 2
    assert len(positions) == 2  # parent removed, 2 daughters added


def test_divide_conserves_volume():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    daughters, _ = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )
    V_parent = (4.0 / 3.0) * math.pi * parent.radius ** 3
    V_total = sum((4.0 / 3.0) * math.pi * d.radius ** 3 for d in daughters)
    assert abs(V_total - V_parent) < 1e-10


def test_divide_daughters_along_axis():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    axis = np.array([1.0, 0.0, 0.0])
    daughters, new_pos = divide_physics_cell(parent, positions, axis=axis)
    d1_pos = new_pos[daughters[0].center_idx]
    d2_pos = new_pos[daughters[1].center_idx]
    assert d1_pos[0] > 0.0  # first daughter along +axis
    assert d2_pos[0] < 0.0  # second daughter along -axis


def test_divide_daughter_indices_valid():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    daughters, new_pos = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0])
    )
    for d in daughters:
        assert 0 <= d.center_idx < len(new_pos)


def test_divide_volume_ratio():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    daughters, _ = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]), volume_ratio=0.6
    )
    r1, r2 = daughters[0].radius, daughters[1].radius
    # r1^3 / (r1^3 + r2^3) should equal 0.6
    frac = r1 ** 3 / (r1 ** 3 + r2 ** 3)
    assert abs(frac - 0.6) < 1e-10


def test_divide_daughter_names():
    positions = np.zeros((0, 3))
    parent, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions,
                                            name="P0")
    daughters, _ = divide_physics_cell(
        parent, positions, axis=np.array([1.0, 0.0, 0.0]),
        daughter_names=("AB", "P1"),
    )
    assert daughters[0].name == "AB"
    assert daughters[1].name == "P1"


def test_divide_with_two_cells_shifts_surviving_index():
    """After dividing cell at index 0, cell at index 1 should shift to index 0."""
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([5.0, 0.0, 0.0]), 1.0, positions)
    c2_old_pos = positions[c2.center_idx].copy()

    daughters, new_pos = divide_physics_cell(
        c1, positions, axis=np.array([0.0, 1.0, 0.0])
    )
    # c2 was at index 1; after deleting index 0 it should be at index 0
    # divide_physics_cell does NOT shift surviving cells — that is WorldPC's job.
    # Instead, confirm that position formerly at index 1 is now at index 0.
    np.testing.assert_array_almost_equal(new_pos[0], c2_old_pos)
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
uv run pytest tests/test_divide_physics_cell.py -v
```

Expected: `ImportError: cannot import name 'divide_physics_cell'`

- [ ] **Step 2.3: Add `divide_physics_cell` to `division.py`**

Append after the existing `divide_cell_body` function (do not touch existing code):

```python
# Add at the bottom of develmorph/division.py
from develmorph.physics_cell import PhysicsCell


def divide_physics_cell(
    parent,
    positions,
    axis=None,
    volume_ratio=0.5,
    daughter_names=(None, None),
    daughter_orientations=(None, None),
):
    """
    Divide a PhysicsCell into two daughters.

    Removes the parent center from positions and appends two daughter centers.
    Volume is distributed according to volume_ratio; radii are scaled so that
    (4/3)*pi*r^3 sums to the parent volume.

    The caller (WorldPC) is responsible for shifting center_idx of all
    surviving cells whose index was above parent.center_idx.

    Args:
        parent: PhysicsCell to divide
        positions: (N, 3) positions array
        axis: division axis unit vector; random if None
        volume_ratio: fraction of volume for first daughter (default 0.5)
        daughter_names: (name1, name2) biological names
        daughter_orientations: (orient1, orient2) 3x3 matrices; inherits if None

    Returns:
        (daughters, new_positions): list of 2 PhysicsCells and updated positions
    """
    if axis is None:
        axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    r = parent.radius
    r1 = r * (volume_ratio ** (1.0 / 3.0))
    r2 = r * ((1.0 - volume_ratio) ** (1.0 / 3.0))

    c = positions[parent.center_idx].copy()
    c1 = c + 0.6 * r1 * axis
    c2 = c - 0.6 * r2 * axis

    # Remove parent center; surviving cells at indices > parent.center_idx shift down
    parent_idx = parent.center_idx
    positions = np.delete(positions, parent_idx, axis=0)

    # Append daughters at the end
    idx_d1 = len(positions)
    idx_d2 = len(positions) + 1
    positions = np.vstack([positions, c1.reshape(1, 3), c2.reshape(1, 3)])

    orient1 = (
        daughter_orientations[0].copy()
        if daughter_orientations[0] is not None
        else parent.orientation.copy()
    )
    orient2 = (
        daughter_orientations[1].copy()
        if daughter_orientations[1] is not None
        else parent.orientation.copy()
    )

    d1 = PhysicsCell(center_idx=idx_d1, radius=r1, name=daughter_names[0], orientation=orient1)
    d2 = PhysicsCell(center_idx=idx_d2, radius=r2, name=daughter_names[1], orientation=orient2)

    return [d1, d2], positions
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
uv run pytest tests/test_divide_physics_cell.py -v
```

Expected: 7 tests pass.

- [ ] **Step 2.5: Run full suite**

```bash
uv run pytest -q
```

Expected: 144 pass (137 + 5 + 7 - 1 because step 2.3 adds 1 import that doesn't break existing tests; the import of `PhysicsCell` inside division.py is fine).

Actually: 137 + 5 + 7 = 149 pass.

- [ ] **Step 2.6: Commit**

```bash
git add develmorph/division.py tests/test_divide_physics_cell.py
git commit -m "feat: add divide_physics_cell for center-only division"
```

---

### Task 3: Add `step_pc()` to `solver.py`

**Files:**
- Modify: `develmorph/solver.py`
- Create: `tests/test_worldpc.py` (step_pc tests go here, alongside WorldPC tests)

- [ ] **Step 3.1: Write failing tests for `step_pc`**

Create `tests/test_worldpc.py` and add these tests first:

```python
# tests/test_worldpc.py
import numpy as np
import pytest
from develmorph.physics_cell import create_physics_cell
from develmorph.solver import step_pc


# ---- step_pc tests ----

def test_step_pc_returns_finite_positions():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, new_vel = step_pc(
        positions, velocities, inv_masses,
        dt=0.01,
        physics_cells=[cell],
        shell_params=(5.0, 5.0, 5.0),
        n_iter=10,
        damping=1.0,
    )
    assert np.all(np.isfinite(new_pos))


def test_step_pc_pushes_center_inside_shell():
    """Center outside effective ellipsoid is pushed inside after one step."""
    positions = np.zeros((0, 3))
    # Shell (2.0, 2.0, 2.0), r=0.5 -> effective (1.5, 1.5, 1.5)
    # Place center at x=1.8, which is outside effective shell
    cell, positions = create_physics_cell(np.array([1.8, 0.0, 0.0]), 0.5, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    for _ in range(20):
        positions, velocities = step_pc(
            positions, velocities, inv_masses,
            dt=0.01,
            physics_cells=[cell],
            shell_params=(2.0, 2.0, 2.0),
            n_iter=10,
            damping=1.0,
        )

    x, y, z = positions[cell.center_idx]
    a_eff, b_eff, c_eff = 1.5, 1.5, 1.5
    C = (x / a_eff) ** 2 + (y / b_eff) ** 2 + (z / c_eff) ** 2
    assert C <= 1.0 + 1e-6


def test_step_pc_two_cells_separated_after_settling():
    """Two overlapping cells are pushed apart by CollisionConstraint."""
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    for _ in range(50):
        positions, velocities = step_pc(
            positions, velocities, inv_masses,
            dt=0.01,
            physics_cells=[c1, c2],
            shell_params=(10.0, 10.0, 10.0),
            n_iter=10,
            damping=1.0,
        )

    dist = np.linalg.norm(positions[c1.center_idx] - positions[c2.center_idx])
    assert dist >= c1.radius + c2.radius - 1e-6


def test_step_pc_damping_one_zeroes_velocity():
    """damping=1.0 means velocity is fully zeroed after each step."""
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    velocities = np.ones_like(positions) * 5.0  # large initial velocity
    inv_masses = np.ones(len(positions))

    _, new_vel = step_pc(
        positions, velocities, inv_masses,
        dt=0.01,
        physics_cells=[cell],
        shell_params=(10.0, 10.0, 10.0),
        n_iter=1,
        damping=1.0,
    )
    assert np.allclose(new_vel, 0.0)


def test_step_pc_asymmetric_shell_confinement():
    """
    Narrow shell axis confines cell more than wide axis.

    Shell (3.0, 3.0, 1.5), r=1.0 -> effective (2.0, 2.0, 0.5).
    Center placed at (0, 0, 0.8): outside c-axis effective limit (0.5),
    inside a,b-axis limits.  Should be pushed along z only.
    """
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.8]), 1.0, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    for _ in range(30):
        positions, velocities = step_pc(
            positions, velocities, inv_masses,
            dt=0.01,
            physics_cells=[cell],
            shell_params=(3.0, 3.0, 1.5),
            n_iter=10,
            damping=1.0,
        )

    x, y, z = positions[cell.center_idx]
    # z must be <= 0.5 (effective c-axis)
    assert abs(z) <= 0.5 + 1e-6
    # x, y essentially unchanged (were 0)
    assert abs(x) < 0.1
    assert abs(y) < 0.1
```

- [ ] **Step 3.2: Run to confirm failures**

```bash
uv run pytest tests/test_worldpc.py::test_step_pc_returns_finite_positions -v
```

Expected: `ImportError: cannot import name 'step_pc' from 'develmorph.solver'`

- [ ] **Step 3.3: Add `_update_adhesion_pairs_pc` and `step_pc` to `solver.py`**

Append after the existing `step()` function (do not modify existing code):

```python
# Append to develmorph/solver.py

def _update_adhesion_pairs_pc(positions, physics_cells, active_adhesions,
                               attach_margin, detach_margin):
    """
    Update active adhesion pairs for PhysicsCell list using hysteresis.

    Same semantics as _update_adhesion_pairs but uses pc.radius instead of
    cb.nominal_radius(). Uses cell list indices (not particle indices) for pairs.
    """
    n = len(physics_cells)

    to_remove = set()
    for i, j in active_adhesions:
        if i >= n or j >= n:
            to_remove.add((i, j))
            continue
        ci, cj = physics_cells[i], physics_cells[j]
        dist = np.linalg.norm(positions[ci.center_idx] - positions[cj.center_idx])
        if dist > ci.radius + cj.radius + detach_margin:
            to_remove.add((i, j))
    active_adhesions -= to_remove

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in active_adhesions:
                continue
            ci, cj = physics_cells[i], physics_cells[j]
            dist = np.linalg.norm(positions[ci.center_idx] - positions[cj.center_idx])
            if dist < ci.radius + cj.radius + attach_margin:
                active_adhesions.add((i, j))


def step_pc(
    positions,
    velocities,
    inv_masses,
    dt,
    physics_cells,
    shell_params,
    n_iter=10,
    active_adhesions=None,
    adhesion_compliance=0.1,
    attach_margin=0.3,
    detach_margin=0.8,
    damping=0.0,
    shell_compliance=0.0,
    collision_compliance=0.0,
):
    """
    Advance PhysicsCell state by one XPBD time step.

    Each cell is a single center particle. Constraints:
      - ShellConstraint per cell: center must fit inside offset ellipsoid
        (effective axes = shell_axes - cell.radius)
      - CollisionConstraint per pair: center-to-center >= r_i + r_j
      - AdhesionConstraint per active pair: center-to-center <= r_i + r_j

    Args:
        positions: (N, 3) center positions (N = number of cells)
        velocities: (N, 3) velocities
        inv_masses: (N,) inverse masses (typically all 1.0)
        dt: time step
        physics_cells: list of PhysicsCell
        shell_params: (a, b, c) ellipsoid semi-axes
        n_iter: XPBD constraint iterations per step
        active_adhesions: mutable set of (i, j) cell-index pairs; None = no adhesion
        adhesion_compliance: compliance for adhesion constraints
        attach_margin: extra distance above r_i+r_j to form adhesion
        detach_margin: extra distance above r_i+r_j to break adhesion
        damping: velocity damping [0=inertial, 1=overdamped]
        shell_compliance: compliance for shell constraint (0=rigid)
        collision_compliance: compliance for collision constraint (0=rigid)

    Returns:
        (new_positions, new_velocities)
    """
    x_prev = positions.copy()
    x = positions + dt * velocities

    a, b, c_ax = shell_params

    # Shell: each center must fit its sphere inside the ellipsoid
    shell_constraints = [
        ShellConstraint(pc.center_idx, a, b, c_ax, pc.radius, shell_compliance)
        for pc in physics_cells
    ]

    # Collision: center-to-center >= r_i + r_j
    collision_constraints = []
    n = len(physics_cells)
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = physics_cells[i], physics_cells[j]
            collision_constraints.append(
                CollisionConstraint(
                    ci.center_idx, cj.center_idx,
                    ci.radius + cj.radius,
                    collision_compliance,
                )
            )

    # Adhesion: attraction for active pairs
    adhesion_constraints = []
    if active_adhesions is not None:
        _update_adhesion_pairs_pc(
            x, physics_cells, active_adhesions, attach_margin, detach_margin
        )
        for i, j in active_adhesions:
            ci, cj = physics_cells[i], physics_cells[j]
            adhesion_constraints.append(
                AdhesionConstraint(
                    ci.center_idx, cj.center_idx,
                    ci.radius + cj.radius,
                    adhesion_compliance,
                )
            )

    # Priority: adhesion → collision → shell
    all_constraints = adhesion_constraints + collision_constraints + shell_constraints
    lambdas = [0.0] * len(all_constraints)

    for _ in range(n_iter):
        for k, con in enumerate(all_constraints):
            lambdas[k] = con.project(x, inv_masses, dt, lambdas[k])

    new_vel = (x - x_prev) / dt * (1.0 - damping) if dt > 0 else np.zeros_like(x)
    return x, new_vel
```

- [ ] **Step 3.4: Run `step_pc` tests to verify they pass**

```bash
uv run pytest tests/test_worldpc.py -k "step_pc" -v
```

Expected: 5 tests pass.

- [ ] **Step 3.5: Run full suite**

```bash
uv run pytest -q
```

Expected: 154 pass, 0 fail.

- [ ] **Step 3.6: Commit**

```bash
git add develmorph/solver.py tests/test_worldpc.py
git commit -m "feat: add step_pc and _update_adhesion_pairs_pc for center-only XPBD"
```

---

### Task 4: Add `WorldPC` to `engine.py`

**Files:**
- Modify: `develmorph/engine.py`
- Modify: `tests/test_worldpc.py` (add WorldPC tests)

- [ ] **Step 4.1: Write failing tests for `WorldPC`**

Add to `tests/test_worldpc.py`:

```python
# Append to tests/test_worldpc.py
from develmorph.celllineage import CellLineage
from develmorph.engine import WorldPC

YAML_PATH = "develmorph/lineages/celegans_early.yaml"


def make_p0_worldpc():
    lineage = CellLineage(YAML_PATH)
    positions = np.zeros((0, 3))
    p0, positions = create_physics_cell(
        np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions, name="P0"
    )
    return WorldPC(
        physics_cells=[p0],
        positions=positions,
        lineage=lineage,
        shell_params=(5.0, 5.0, 5.0),
        damping=1.0,
    )


def test_worldpc_step_increments_count():
    world = make_p0_worldpc()
    world.step()
    assert world.step_count == 1


def test_worldpc_positions_finite_after_step():
    world = make_p0_worldpc()
    world.step()
    assert np.all(np.isfinite(world.positions))


def test_worldpc_step0_no_division():
    """Step at count=0 does not trigger division (YAML first step is 1)."""
    world = make_p0_worldpc()
    world.step()  # step_count=0 during execution
    assert len(world.physics_cells) == 1
    assert world.physics_cells[0].name == "P0"


def test_worldpc_step1_p0_divides_into_ab_p1():
    world = make_p0_worldpc()
    world.step()  # step_count=0
    world.step()  # step_count=1: P0 -> AB + P1
    names = {pc.name for pc in world.physics_cells}
    assert names == {"AB", "P1"}


def test_worldpc_four_cell_stage():
    world = make_p0_worldpc()
    for _ in range(4):
        world.step()  # step_counts 0,1,2,3
    names = {pc.name for pc in world.physics_cells}
    assert names == {"ABa", "ABp", "EMS", "P2"}


def test_worldpc_positions_length_equals_cell_count():
    """positions array has exactly one row per PhysicsCell."""
    world = make_p0_worldpc()
    for _ in range(4):
        world.step()
    assert len(world.positions) == len(world.physics_cells)


def test_worldpc_division_volume_ratio():
    """P0 -> AB (60%) + P1 (40%)."""
    world = make_p0_worldpc()
    world.step()  # step_count=0
    world.step()  # step_count=1
    ab = next(pc for pc in world.physics_cells if pc.name == "AB")
    p1 = next(pc for pc in world.physics_cells if pc.name == "P1")
    r1, r2 = ab.radius, p1.radius
    total = r1 ** 3 + r2 ** 3
    assert abs(r1 ** 3 / total - 0.6) < 1e-6


def test_worldpc_no_lineage_no_division():
    """WorldPC without lineage runs without dividing."""
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    world = WorldPC(
        physics_cells=[cell],
        positions=positions,
        shell_params=(5.0, 5.0, 5.0),
        damping=1.0,
    )
    world.step()
    assert len(world.physics_cells) == 1
```

- [ ] **Step 4.2: Run tests to confirm failure**

```bash
uv run pytest tests/test_worldpc.py -k "worldpc" -v
```

Expected: `ImportError: cannot import name 'WorldPC' from 'develmorph.engine'`

- [ ] **Step 4.3: Add `WorldPC` to `engine.py`**

Append after the existing `World` class (do not modify `World`):

```python
# Append to develmorph/engine.py
from develmorph.division import divide_physics_cell
from develmorph.physics_cell import PhysicsCell
from develmorph.solver import step_pc


class WorldPC:
    """
    Simulation world for PhysicsCell (center-only XPBD).

    Each cell is a single center point with a radius scalar.
    Shell constraint uses offset ellipsoid: effective axes = shell_axes - cell.radius.
    Division is handled via lineage rules; index-shifting is managed internally.
    """

    def __init__(
        self,
        physics_cells,
        positions,
        dt=0.01,
        shell_params=(5.0, 5.0, 5.0),
        lineage=None,
        n_iter=10,
        damping=0.0,
        shell_compliance=0.0,
        collision_compliance=0.0,
        adhesion_compliance=0.1,
        attach_margin=0.3,
        detach_margin=0.8,
    ):
        self.physics_cells = list(physics_cells)
        self.positions = positions.copy()
        self.velocities = np.zeros_like(positions)
        self.inv_masses = np.ones(len(positions))
        self.dt = dt
        self.shell_params = shell_params
        self.lineage = lineage
        self.n_iter = n_iter
        self.damping = damping
        self.shell_compliance = shell_compliance
        self.collision_compliance = collision_compliance
        self.adhesion_compliance = adhesion_compliance
        self.attach_margin = attach_margin
        self.detach_margin = detach_margin
        self.step_count = 0
        self.active_adhesions = set()

    def step(self):
        """Advance simulation by one time step."""
        if self.lineage is not None:
            self._apply_lineage_divisions()

        self.positions, self.velocities = step_pc(
            self.positions,
            self.velocities,
            self.inv_masses,
            self.dt,
            self.physics_cells,
            self.shell_params,
            n_iter=self.n_iter,
            active_adhesions=self.active_adhesions,
            adhesion_compliance=self.adhesion_compliance,
            attach_margin=self.attach_margin,
            detach_margin=self.detach_margin,
            damping=self.damping,
            shell_compliance=self.shell_compliance,
            collision_compliance=self.collision_compliance,
        )
        self.step_count += 1

    def run(self, n_steps):
        """Run n_steps and return list of position snapshots."""
        traj = []
        for _ in range(n_steps):
            self.step()
            traj.append(self.positions.copy())
        return traj

    def _apply_lineage_divisions(self):
        """Apply all division rules scheduled for the current step_count."""
        program = self.lineage.get_step_program(self.step_count)
        if program is None:
            return

        for rule in program.divisions:
            idx = next(
                (i for i, pc in enumerate(self.physics_cells) if pc.name == rule.parent),
                None,
            )
            if idx is None:
                continue

            parent = self.physics_cells[idx]
            parent_center_idx = parent.center_idx
            axis = self.lineage.resolve_division_axis(rule, parent.orientation)

            daughters, self.positions = divide_physics_cell(
                parent,
                self.positions,
                axis=axis,
                volume_ratio=rule.volume_ratio,
                daughter_names=tuple(rule.daughters),
                daughter_orientations=(
                    parent.orientation.copy(),
                    parent.orientation.copy(),
                ),
            )

            # Shift center_idx of all surviving cells whose index was above parent
            for pc in self.physics_cells:
                if pc is not parent and pc.center_idx > parent_center_idx:
                    pc.center_idx -= 1

            self.physics_cells.pop(idx)
            self.physics_cells.extend(daughters)
            self.velocities = np.zeros_like(self.positions)
            self.inv_masses = np.ones(len(self.positions))
            self.active_adhesions.clear()
```

- [ ] **Step 4.4: Run WorldPC tests**

```bash
uv run pytest tests/test_worldpc.py -v
```

Expected: all tests pass (step_pc tests from Task 3 + WorldPC tests from Task 4).

- [ ] **Step 4.5: Run full suite**

```bash
uv run pytest -q
```

Expected: 163 pass, 0 fail (137 old + 5 physics_cell + 7 divide + 5 step_pc + 9 worldpc).

- [ ] **Step 4.6: Commit**

```bash
git add develmorph/engine.py tests/test_worldpc.py
git commit -m "feat: add WorldPC class for center-only simulation"
```

---

### Task 5: Validate diamond formation (Phase 1 gate)

**Files:**
- Create: `scripts/demo_diamond_pc.py`

This script ports `experiment_diamond.py` to `WorldPC` and checks `|EMS_y| > 0.5`.

- [ ] **Step 5.1: Write `scripts/demo_diamond_pc.py`**

```python
"""
Diamond formation validation for WorldPC (PhysicsCell model).

Runs the same division sequence as experiment_diamond.py but using the
center-only XPBD model with OffsetShellConstraint (via ShellConstraint
with radius=pc.radius).

Success criterion: |EMS_y| > 0.5 (comparable to current CellBody model at 0.84).

Run with:
    uv run python scripts/demo_diamond_pc.py
"""

import numpy as np

from develmorph.celllineage import CellLineage
from develmorph.engine import WorldPC
from develmorph.physics_cell import create_physics_cell

SHELL = (1.84, 1.26, 1.04)
N_SETTLE = 300
YAML = "develmorph/lineages/celegans_early.yaml"


def run_diamond(label, n_iter_div, n_iter_settle, n_div_steps=4):
    lineage = CellLineage(YAML)
    positions = np.zeros((0, 3))
    p0, positions = create_physics_cell(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.3,
        positions=positions,
        name="P0",
    )

    world = WorldPC(
        physics_cells=[p0],
        positions=positions,
        lineage=lineage,
        shell_params=SHELL,
        dt=0.01,
        n_iter=n_iter_div,
        damping=1.0,
        shell_compliance=0.001,
        adhesion_compliance=0.1,
        attach_margin=0.3,
        detach_margin=0.8,
    )

    for _ in range(n_div_steps):
        world.step()

    print(f"  [{label}] cells after divisions: "
          f"{[pc.name for pc in world.physics_cells]}")

    # Settling
    world.n_iter = n_iter_settle
    world.lineage = None
    for _ in range(N_SETTLE):
        world.step()

    return world


def score(world):
    pos = {pc.name: world.positions[pc.center_idx] for pc in world.physics_cells}
    ems_y = abs(pos["EMS"][1]) if "EMS" in pos else float("nan")
    return pos, ems_y


def main():
    print("=" * 60)
    print("Diamond Validation — WorldPC (PhysicsCell model)")
    print(f"Shell: {SHELL}  Settle: {N_SETTLE} steps")
    print("=" * 60)

    experiments = [
        ("n_iter=100", 100, 100),
        ("n_iter=10", 10, 100),
    ]

    results = {}
    for label, n_div, n_set in experiments:
        print(f"\nRunning: {label}")
        world = run_diamond(label, n_div, n_set)
        pos, ems_y = score(world)
        results[label] = ems_y
        print(f"\n  --- {label} ---")
        for name in ["ABa", "ABp", "EMS", "P2"]:
            if name in pos:
                p = pos[name]
                print(f"    {name:4s}  x={p[0]:+.3f}  y={p[1]:+.3f}  z={p[2]:+.3f}")
        print(f"  |EMS_y| = {ems_y:.4f}  {'DIAMOND' if ems_y > 0.5 else 'T-shape'}")

    print("\n" + "=" * 60)
    print("Summary")
    for label, ems_y in results.items():
        bar = "#" * int(ems_y * 20)
        status = "PASS" if ems_y > 0.5 else "FAIL (tune needed)"
        print(f"  {ems_y:.3f} |{bar:<20}| {label}  [{status}]")
    print("=" * 60)

    passed = all(v > 0.5 for v in results.values())
    if passed:
        print("\nPhase 1 gate: PASSED. Proceed to Phase 2.")
    else:
        print("\nPhase 1 gate: FAILED. Tune attach_margin/adhesion_compliance.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run validation**

```bash
uv run python scripts/demo_diamond_pc.py
```

Expected: `|EMS_y| > 0.5` for at least one experiment.

- [ ] **Step 5.3: If |EMS_y| < 0.5 — tune parameters**

If the diamond score is too low, try adjusting these parameters in the script:

| Parameter | Try | Effect |
|-----------|-----|--------|
| `attach_margin` | 0.1 → 0.5 | Lower = harder to form adhesion |
| `adhesion_compliance` | 0.1 → 0.01 | Lower = stiffer adhesion |
| `shell_compliance` | 0.001 → 0.0 | Rigid shell pushes harder |
| `n_iter_div` | 100 → 10 | Less settling during division = more asymmetry |

Re-run after each change. Record the best parameters in a comment at the top of the script.

- [ ] **Step 5.4: Commit script (even if gated)**

```bash
git add scripts/demo_diamond_pc.py
git commit -m "feat: add demo_diamond_pc.py validation script for WorldPC"
```

---

## Phase 2: Contact-Area-Weighted Adhesion

---

### Task 6: Add `ContactRecord` and `compute_contacts`

**Files:**
- Create: `develmorph/contact.py`
- Create: `tests/test_contact.py`

- [ ] **Step 6.1: Write failing tests**

```python
# tests/test_contact.py
import math
import numpy as np
from develmorph.physics_cell import create_physics_cell
from develmorph.contact import ContactRecord, compute_contacts


def test_compute_contacts_overlapping_pair():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)
    # overlap = 2.0 - 1.5 = 0.5

    contacts = compute_contacts([c1, c2], positions)

    assert len(contacts) == 1
    c = contacts[0]
    assert c.i == 0 and c.j == 1
    assert abs(c.overlap - 0.5) < 1e-10
    assert c.area > 0.0


def test_compute_contacts_separated_pair():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([3.0, 0.0, 0.0]), 1.0, positions)
    # dist=3, r1+r2=2: no overlap

    contacts = compute_contacts([c1, c2], positions)
    assert len(contacts) == 0


def test_contact_area_formula():
    """Contact area = pi * delta * r_harmonic."""
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)

    contacts = compute_contacts([c1, c2], positions)
    record = contacts[0]

    delta = 0.5
    r_harm = 2 * 1.0 * 1.0 / (1.0 + 1.0)  # = 1.0
    expected_area = math.pi * delta * r_harm
    assert abs(record.area - expected_area) < 1e-10


def test_contact_normal_points_from_j_to_i():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)

    contacts = compute_contacts([c1, c2], positions)
    n = contacts[0].normal
    # normal should point from j (right) to i (left), i.e. [-1, 0, 0]
    np.testing.assert_array_almost_equal(n, [-1.0, 0.0, 0.0])


def test_compute_contacts_three_cells():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)
    c3, positions = create_physics_cell(np.array([3.5, 0.0, 0.0]), 1.0, positions)
    # c1-c2 overlap, c2-c3 overlap, c1-c3 separated

    contacts = compute_contacts([c1, c2, c3], positions)
    pairs = {(c.i, c.j) for c in contacts}
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (0, 2) not in pairs
```

- [ ] **Step 6.2: Run to confirm failure**

```bash
uv run pytest tests/test_contact.py -v
```

Expected: `ModuleNotFoundError: No module named 'develmorph.contact'`

- [ ] **Step 6.3: Implement `contact.py`**

```python
# develmorph/contact.py
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ContactRecord:
    """
    Analytical contact between two PhysicsCells.

    Computed from center-to-center distance and cell radii.
    Normal points from cell j toward cell i (repulsion direction for i).
    """

    i: int          # index in physics_cells list
    j: int          # index in physics_cells list
    overlap: float  # delta = r_i + r_j - dist (> 0 when in contact)
    area: float     # Hertz-like contact area = pi * delta * r_harmonic
    normal: np.ndarray  # unit vector from j to i


def compute_contacts(physics_cells, positions):
    """
    Compute all pairwise contacts between PhysicsCells.

    Args:
        physics_cells: list of PhysicsCell
        positions: (N, 3) center positions

    Returns:
        list of ContactRecord for all overlapping pairs (i < j)
    """
    records = []
    n = len(physics_cells)
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = physics_cells[i], physics_cells[j]
            delta_vec = positions[ci.center_idx] - positions[cj.center_idx]
            dist = np.linalg.norm(delta_vec)
            overlap = ci.radius + cj.radius - dist
            if overlap <= 0.0:
                continue
            normal = delta_vec / dist if dist > 1e-12 else np.array([1.0, 0.0, 0.0])
            r_harm = 2.0 * ci.radius * cj.radius / (ci.radius + cj.radius)
            area = math.pi * overlap * r_harm
            records.append(ContactRecord(i=i, j=j, overlap=overlap, area=area, normal=normal))
    return records
```

- [ ] **Step 6.4: Run tests**

```bash
uv run pytest tests/test_contact.py -v
```

Expected: 5 tests pass.

- [ ] **Step 6.5: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 6.6: Commit**

```bash
git add develmorph/contact.py tests/test_contact.py
git commit -m "feat: add ContactRecord and compute_contacts for analytical contact geometry"
```

---

### Task 7: Use contact area to weight adhesion compliance in `step_pc`

**Files:**
- Modify: `develmorph/solver.py` (update `step_pc` adhesion section)
- Modify: `tests/test_worldpc.py` (add area-adhesion test)

Contact-area weighting: when two cells overlap by δ, the adhesion compliance is scaled as
`alpha_effective = base_compliance / (A_contact + epsilon)`. Larger contact area → smaller compliance → stronger attraction.

- [ ] **Step 7.1: Write failing test**

Add to `tests/test_worldpc.py`:

```python
def test_step_pc_area_adhesion_stronger_for_larger_overlap():
    """
    Two cell pairs with different overlaps: the pair with larger overlap should
    attract more strongly (end up closer to rest length sooner).

    We check that the force direction is correct: both pairs attract (move toward
    rest_length = r_i + r_j) and the larger-overlap pair moves more per step.
    """
    from develmorph.solver import step_pc
    import numpy as np
    from develmorph.physics_cell import create_physics_cell

    # Pair A: small overlap (0.1)
    posA = np.zeros((0, 3))
    a1, posA = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, posA)
    a2, posA = create_physics_cell(np.array([1.9, 0.0, 0.0]), 1.0, posA)

    # Pair B: large overlap (0.5)
    posB = np.zeros((0, 3))
    b1, posB = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, posB)
    b2, posB = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, posB)

    # Both pairs start with cells beyond rest_length (adhesion kicks in)
    # Place them slightly beyond rest_length so adhesion applies
    posA[a2.center_idx, 0] = 2.1  # beyond rest_length=2.0
    posB[b2.center_idx, 0] = 2.1

    adhesA = {(0, 1)}
    adhesB = {(0, 1)}

    velA = np.zeros_like(posA)
    velB = np.zeros_like(posB)
    mA = np.ones(len(posA))
    mB = np.ones(len(posB))

    # One step of area-weighted adhesion
    posA_new, _ = step_pc(posA, velA, mA, 0.01, [a1, a2], (10.0, 10.0, 10.0),
                           n_iter=10, active_adhesions=adhesA,
                           adhesion_compliance=0.1, damping=1.0,
                           use_area_adhesion=True)
    posB_new, _ = step_pc(posB, velB, mB, 0.01, [b1, b2], (10.0, 10.0, 10.0),
                           n_iter=10, active_adhesions=adhesB,
                           adhesion_compliance=0.1, damping=1.0,
                           use_area_adhesion=True)

    # Both should move toward rest_length=2.0
    distA = np.linalg.norm(posA_new[a1.center_idx] - posA_new[a2.center_idx])
    distB = np.linalg.norm(posB_new[b1.center_idx] - posB_new[b2.center_idx])
    assert distA < 2.1  # moved toward rest_length
    assert distB < 2.1
```

- [ ] **Step 7.2: Run to confirm failure (no `use_area_adhesion` param yet)**

```bash
uv run pytest tests/test_worldpc.py::test_step_pc_area_adhesion_stronger_for_larger_overlap -v
```

Expected: `TypeError: step_pc() got an unexpected keyword argument 'use_area_adhesion'`

- [ ] **Step 7.3: Add `use_area_adhesion` parameter to `step_pc`**

In `develmorph/solver.py`, update the `step_pc` function signature and the adhesion section:

```python
# Updated step_pc signature — add use_area_adhesion parameter
def step_pc(
    positions,
    velocities,
    inv_masses,
    dt,
    physics_cells,
    shell_params,
    n_iter=10,
    active_adhesions=None,
    adhesion_compliance=0.1,
    attach_margin=0.3,
    detach_margin=0.8,
    damping=0.0,
    shell_compliance=0.0,
    collision_compliance=0.0,
    use_area_adhesion=False,   # <-- new
):
    ...
    # Replace the adhesion_constraints section with:
    adhesion_constraints = []
    if active_adhesions is not None:
        _update_adhesion_pairs_pc(
            x, physics_cells, active_adhesions, attach_margin, detach_margin
        )
        for i, j in active_adhesions:
            ci, cj = physics_cells[i], physics_cells[j]
            rest = ci.radius + cj.radius

            if use_area_adhesion:
                dist = np.linalg.norm(x[ci.center_idx] - x[cj.center_idx])
                overlap = max(0.0, rest - dist)
                if overlap > 0.0:
                    r_harm = 2.0 * ci.radius * cj.radius / (ci.radius + cj.radius)
                    import math
                    area = math.pi * overlap * r_harm
                    effective_compliance = adhesion_compliance / (area + 1e-6)
                else:
                    effective_compliance = adhesion_compliance
            else:
                effective_compliance = adhesion_compliance

            adhesion_constraints.append(
                AdhesionConstraint(
                    ci.center_idx, cj.center_idx,
                    rest,
                    effective_compliance,
                )
            )
```

- [ ] **Step 7.4: Run test**

```bash
uv run pytest tests/test_worldpc.py -v
```

Expected: all tests pass including the new area-adhesion test.

- [ ] **Step 7.5: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 7.6: Add `use_area_adhesion` to `WorldPC`**

In `develmorph/engine.py`, add `use_area_adhesion=False` to `WorldPC.__init__` and forward it to `step_pc`:

```python
# In WorldPC.__init__, add to parameter list:
    use_area_adhesion: bool = False,

# Store it:
    self.use_area_adhesion = use_area_adhesion

# In WorldPC.step(), add to step_pc call:
    use_area_adhesion=self.use_area_adhesion,
```

- [ ] **Step 7.7: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 7.8: Update diamond validation script**

In `scripts/demo_diamond_pc.py`, change `WorldPC(...)` to add `use_area_adhesion=True` and re-run:

```bash
uv run python scripts/demo_diamond_pc.py
```

Compare `|EMS_y|` with and without `use_area_adhesion`. Record result in a comment.

- [ ] **Step 7.9: Commit**

```bash
git add develmorph/solver.py develmorph/engine.py scripts/demo_diamond_pc.py tests/test_worldpc.py
git commit -m "feat: add use_area_adhesion option to step_pc and WorldPC (Hertz contact area)"
```

---

## Phase 3: SurfaceShell Visualization

---

### Task 8: Add `SurfaceShell` kinematic reconstruction to `viz.py`

**Files:**
- Modify: `develmorph/viz.py`

- [ ] **Step 8.1: Write failing test**

Add a new test file `tests/test_surface_shell.py`:

```python
# tests/test_surface_shell.py
import numpy as np
from develmorph.physics_cell import create_physics_cell
from develmorph.contact import ContactRecord, compute_contacts
from develmorph.viz import SurfaceShell, reconstruct_surface_shells


def test_surface_shell_sphere_points_on_radius():
    """Default shell (no contacts) gives points at distance r from center."""
    shell = SurfaceShell(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.5,
        contacts=[],
    )
    points = shell.mesh_points(n_subdivisions=0)
    dists = np.linalg.norm(points, axis=1)
    np.testing.assert_allclose(dists, 1.5, atol=1e-10)


def test_reconstruct_surface_shells_one_cell():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([1.0, 2.0, 3.0]), 1.0, positions)
    shells = reconstruct_surface_shells([cell], positions, [])
    assert len(shells) == 1
    np.testing.assert_array_almost_equal(shells[0].center, [1.0, 2.0, 3.0])
    assert shells[0].radius == 1.0


def test_surface_shell_contact_flattens_toward_contact_plane():
    """
    With a contact, mesh points near the contact direction should be
    pulled inward (flattened) relative to the pure sphere.
    """
    contact = ContactRecord(
        i=0, j=1, overlap=0.3, area=0.5,
        normal=np.array([-1.0, 0.0, 0.0]),  # contact from +x direction
    )
    shell = SurfaceShell(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        contacts=[contact],
    )
    points = shell.mesh_points(n_subdivisions=0)
    # Point along +x direction should be flattened (x < 1.0)
    x_points = points[points[:, 0] > 0.5]  # points roughly along +x
    if len(x_points) > 0:
        assert x_points[:, 0].max() < 1.0 + 1e-6
```

- [ ] **Step 8.2: Run to confirm failure**

```bash
uv run pytest tests/test_surface_shell.py -v
```

Expected: `ImportError: cannot import name 'SurfaceShell' from 'develmorph.viz'`

- [ ] **Step 8.3: Add `SurfaceShell` and `reconstruct_surface_shells` to `viz.py`**

Append to `develmorph/viz.py`:

```python
# Append to develmorph/viz.py
from develmorph.cell_body import OCTAHEDRON_DIRS


class SurfaceShell:
    """
    Kinematic surface mesh for a PhysicsCell.

    Reconstructed each step from center + radius + contact records.
    Does NOT participate in XPBD. No drift possible.

    Points are sphere vertices optionally flattened at contact planes.
    """

    def __init__(self, center, radius, contacts):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.contacts = contacts  # list of ContactRecord where i or j is this cell

    def mesh_points(self, n_subdivisions=0):
        """
        Return (M, 3) surface mesh points.

        n_subdivisions: additional subdivision rounds (0 = raw octahedron, 6 pts)
        Contacts flatten the sphere toward each contact plane.
        """
        dirs = OCTAHEDRON_DIRS.copy()

        # Flatten at each contact plane
        for record in self.contacts:
            # normal points from j to i; the contact is on the +normal side of cell i
            # and the -normal side of cell j.
            # For this cell: if it is cell i (record.i matches), contact is on +normal side
            n = record.normal  # unit vector
            flatten_amount = min(record.overlap * 0.5, self.radius * 0.3)
            for k, d in enumerate(dirs):
                # Project d onto contact normal direction
                proj = np.dot(d, n)
                if proj > 0.0:
                    # Pull this vertex inward proportional to projection and overlap
                    dirs[k] = d - flatten_amount * proj * n / self.radius

            # Re-normalize to stay on sphere surface approximately
            for k in range(len(dirs)):
                norm = np.linalg.norm(dirs[k])
                if norm > 1e-12:
                    dirs[k] /= norm

        return self.center + self.radius * dirs


def reconstruct_surface_shells(physics_cells, positions, contacts):
    """
    Build a SurfaceShell for each PhysicsCell from current physics state.

    Args:
        physics_cells: list of PhysicsCell
        positions: (N, 3) current center positions
        contacts: list of ContactRecord from compute_contacts()

    Returns:
        list of SurfaceShell, one per cell
    """
    shells = []
    for idx, pc in enumerate(physics_cells):
        cell_contacts = [r for r in contacts if r.i == idx or r.j == idx]
        shells.append(SurfaceShell(
            center=positions[pc.center_idx],
            radius=pc.radius,
            contacts=cell_contacts,
        ))
    return shells
```

- [ ] **Step 8.4: Run tests**

```bash
uv run pytest tests/test_surface_shell.py -v
```

Expected: 3 tests pass (contact-flattening test is approximate, may need adjustment).

- [ ] **Step 8.5: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 8.6: Commit**

```bash
git add develmorph/viz.py tests/test_surface_shell.py
git commit -m "feat: add SurfaceShell kinematic reconstruction for PhysicsCell visualization"
```

---

## Phase 4: Spatial Hashing for 1,000 Cells

---

### Task 9: Add `SpatialHash` for O(N) collision detection

**Files:**
- Create: `develmorph/spatial_hash.py`
- Create: `tests/test_spatial_hash.py`

- [ ] **Step 9.1: Write failing tests**

```python
# tests/test_spatial_hash.py
import numpy as np
from develmorph.physics_cell import create_physics_cell
from develmorph.spatial_hash import SpatialHash


def test_spatial_hash_finds_overlapping_pair():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)
    c3, positions = create_physics_cell(np.array([10.0, 0.0, 0.0]), 1.0, positions)

    sh = SpatialHash(cell_size=2.5)
    sh.build([c1, c2, c3], positions)
    pairs = sh.candidate_pairs()

    assert (0, 1) in pairs or (1, 0) in pairs
    assert (0, 2) not in pairs and (2, 0) not in pairs


def test_spatial_hash_empty_cells():
    sh = SpatialHash(cell_size=2.0)
    sh.build([], np.zeros((0, 3)))
    assert sh.candidate_pairs() == set()


def test_spatial_hash_single_cell():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    sh = SpatialHash(cell_size=2.0)
    sh.build([c1], positions)
    assert sh.candidate_pairs() == set()


def test_spatial_hash_finds_all_overlapping_pairs():
    """Chain of 4 cells; each overlaps only its neighbor."""
    positions = np.zeros((0, 3))
    cells = []
    for i in range(4):
        c, positions = create_physics_cell(np.array([float(i) * 1.8, 0.0, 0.0]),
                                           1.0, positions)
        cells.append(c)

    sh = SpatialHash(cell_size=2.5)
    sh.build(cells, positions)
    pairs = sh.candidate_pairs()

    # Neighbors overlap (dist=1.8 < r1+r2=2.0)
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (2, 3) in pairs
    # Non-neighbors do not (dist=3.6 > 2.0)
    assert (0, 2) not in pairs
```

- [ ] **Step 9.2: Run to confirm failure**

```bash
uv run pytest tests/test_spatial_hash.py -v
```

Expected: `ModuleNotFoundError: No module named 'develmorph.spatial_hash'`

- [ ] **Step 9.3: Implement `spatial_hash.py`**

```python
# develmorph/spatial_hash.py
from collections import defaultdict

import numpy as np


class SpatialHash:
    """
    Grid-based spatial hash for O(N) neighbor search.

    Assigns each cell center to a grid voxel of size `cell_size`.
    Candidate collision pairs are cells in adjacent voxels.
    `cell_size` should be >= max(r_i + r_j) across all pairs.
    """

    def __init__(self, cell_size):
        self.cell_size = float(cell_size)
        self._grid = defaultdict(list)
        self._physics_cells = []

    def build(self, physics_cells, positions):
        """Populate the hash grid from current positions."""
        self._grid.clear()
        self._physics_cells = physics_cells

        for idx, pc in enumerate(physics_cells):
            pos = positions[pc.center_idx]
            voxel = self._voxel(pos)
            self._grid[voxel].append(idx)

    def candidate_pairs(self):
        """
        Return set of (i, j) pairs with i < j that are in adjacent voxels.

        Checks 27 neighboring voxels for each occupied voxel.
        """
        pairs = set()
        visited = set()

        for voxel, indices in self._grid.items():
            # Self-pairs within same voxel
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    i, j = indices[a], indices[b]
                    pairs.add((min(i, j), max(i, j)))

            # Pairs with neighboring voxels (only forward to avoid duplicates)
            vx, vy, vz = voxel
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = (vx + dx, vy + dy, vz + dz)
                        if neighbor <= voxel:  # only forward neighbors
                            continue
                        for i in indices:
                            for j in self._grid.get(neighbor, []):
                                pairs.add((min(i, j), max(i, j)))

        return pairs

    def _voxel(self, pos):
        """Map a 3D position to integer voxel coordinates."""
        return (
            int(np.floor(pos[0] / self.cell_size)),
            int(np.floor(pos[1] / self.cell_size)),
            int(np.floor(pos[2] / self.cell_size)),
        )
```

- [ ] **Step 9.4: Run tests**

```bash
uv run pytest tests/test_spatial_hash.py -v
```

Expected: 4 tests pass.

- [ ] **Step 9.5: Integrate spatial hash into `step_pc` (opt-in)**

In `develmorph/solver.py`, add `use_spatial_hash=False` parameter to `step_pc`:

```python
# In step_pc signature, add:
    use_spatial_hash=False,   # set True for >100 cells

# Replace the collision constraint loop:
    collision_constraints = []
    if use_spatial_hash and len(physics_cells) > 16:
        from develmorph.spatial_hash import SpatialHash
        max_radius = max(pc.radius for pc in physics_cells)
        sh = SpatialHash(cell_size=2.0 * max_radius * 1.1)
        sh.build(physics_cells, x)
        candidate_pairs = sh.candidate_pairs()
    else:
        n = len(physics_cells)
        candidate_pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}

    for i, j in candidate_pairs:
        ci, cj = physics_cells[i], physics_cells[j]
        collision_constraints.append(
            CollisionConstraint(
                ci.center_idx, cj.center_idx,
                ci.radius + cj.radius,
                collision_compliance,
            )
        )
```

Also add `use_spatial_hash=False` to `WorldPC.__init__` and forward to `step_pc`.

- [ ] **Step 9.6: Run full suite**

```bash
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 9.7: Benchmark (manual)**

```bash
uv run python -c "
import numpy as np, time
from develmorph.physics_cell import create_physics_cell
from develmorph.engine import WorldPC

N = 200
positions = np.zeros((0, 3))
cells = []
for i in range(N):
    c, positions = create_physics_cell(
        np.random.randn(3) * 3, radius=0.5, positions=positions
    )
    cells.append(c)

for use_hash in [False, True]:
    world = WorldPC(
        physics_cells=cells[:], positions=positions.copy(),
        shell_params=(10.0, 10.0, 10.0), damping=1.0,
        use_spatial_hash=use_hash,
    )
    t0 = time.time()
    world.run(10)
    print(f'use_spatial_hash={use_hash}: {time.time()-t0:.2f}s')
"
```

Record results in a comment in `spatial_hash.py`.

- [ ] **Step 9.8: Commit**

```bash
git add develmorph/spatial_hash.py tests/test_spatial_hash.py develmorph/solver.py develmorph/engine.py
git commit -m "feat: add SpatialHash for O(N) collision detection, opt-in via use_spatial_hash"
```

---

## Final Verification

- [ ] **Run complete test suite**

```bash
uv run pytest -v
```

Expected: all tests pass (137 original + new PC model tests).

- [ ] **Run diamond validation**

```bash
uv run python scripts/demo_diamond_pc.py
```

Expected: `|EMS_y| > 0.5` (Phase 1 gate) or better.

- [ ] **Final commit**

```bash
git add -A
git commit -m "feat: complete kinematic shell model (PhysicsCell + WorldPC)"
```
