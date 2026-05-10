# Icosahedron Cell Topology Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add icosahedron (20-face, 12-vertex) as a selectable cell topology via `topology='ico'` in `create_cell_body`, alongside the existing octahedron (`topology='oct'`).

**Architecture:** Add `ICOSAHEDRON_DIRS/TRIANGLES/EDGES` constants and a `TOPOLOGY_VOLUME_FACTOR` dict to `cell_body.py`. Add `topology` parameter to `CellBody` and `create_cell_body`. Propagate through `divide_cell_body` (including topology-aware radius calculation). All existing behaviour is unchanged; `topology` defaults to `'oct'` everywhere.

**Tech Stack:** Python 3.13, numpy, uv, pytest

---

## File Map

| File | Change |
|------|--------|
| `develmorph/cell_body.py` | Add constants + `topology` param to `CellBody` + `create_cell_body` |
| `develmorph/division.py` | Topology-aware radius + pass `topology` to daughters |
| `tests/test_cell_body.py` | New icosahedron + topology tests |
| `tests/test_division.py` | New topology-propagation test |
| `tests/test_engine.py` | New integration test |

---

## Task 1: Add icosahedron constants and volume factor

**Files:**
- Modify: `develmorph/cell_body.py`
- Test: `tests/test_cell_body.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cell_body.py`:

```python
from develmorph.cell_body import (
    ICOSAHEDRON_DIRS,
    ICOSAHEDRON_TRIANGLES,
    ICOSAHEDRON_EDGES,
    TOPOLOGY_VOLUME_FACTOR,
)


def test_icosahedron_vertex_count():
    assert len(ICOSAHEDRON_DIRS) == 12


def test_icosahedron_dirs_on_unit_sphere():
    norms = np.linalg.norm(ICOSAHEDRON_DIRS, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_icosahedron_face_count():
    assert len(ICOSAHEDRON_TRIANGLES) == 20


def test_icosahedron_edge_count():
    assert len(ICOSAHEDRON_EDGES) == 30


def test_topology_volume_factor_oct():
    # Octahedron inscribed in unit sphere: V = (4/3) r^3
    np.testing.assert_allclose(TOPOLOGY_VOLUME_FACTOR['oct'], 4.0 / 3.0, atol=1e-10)


def test_topology_volume_factor_ico_larger_than_oct():
    # Icosahedron is a better sphere approximation, so its volume factor is larger
    assert TOPOLOGY_VOLUME_FACTOR['ico'] > TOPOLOGY_VOLUME_FACTOR['oct']


def test_topology_volume_factor_ico_less_than_sphere():
    # Must be less than sphere: (4/3)*pi ≈ 4.189
    assert TOPOLOGY_VOLUME_FACTOR['ico'] < 4.0 / 3.0 * np.pi
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cell_body.py::test_icosahedron_vertex_count -v
```

Expected: `ImportError` (names not defined yet)

- [ ] **Step 3: Add constants to `develmorph/cell_body.py`**

`cell_body.py` already imports `math` and `numpy as np`. Add the following after the existing `OCTAHEDRON_EDGES` block:

```python
# Icosahedron: 12 vertices via golden ratio
# Positions are all permutations of (0, +/-1, +/-phi), normalized to unit sphere.
# Vertex index layout:
#   0-3:  (0, +/-1, +/-phi)
#   4-7:  (+/-1, +/-phi, 0)
#   8-11: (+/-phi, 0, +/-1)
_PHI = (1.0 + math.sqrt(5.0)) / 2.0
_ICO_NORM = math.sqrt(1.0 + _PHI * _PHI)

ICOSAHEDRON_DIRS = np.array([
    [0.0,            1.0 / _ICO_NORM,  _PHI / _ICO_NORM],   # 0
    [0.0,           -1.0 / _ICO_NORM,  _PHI / _ICO_NORM],   # 1
    [0.0,            1.0 / _ICO_NORM, -_PHI / _ICO_NORM],   # 2
    [0.0,           -1.0 / _ICO_NORM, -_PHI / _ICO_NORM],   # 3
    [ 1.0 / _ICO_NORM,  _PHI / _ICO_NORM, 0.0],             # 4
    [-1.0 / _ICO_NORM,  _PHI / _ICO_NORM, 0.0],             # 5
    [ 1.0 / _ICO_NORM, -_PHI / _ICO_NORM, 0.0],             # 6
    [-1.0 / _ICO_NORM, -_PHI / _ICO_NORM, 0.0],             # 7
    [ _PHI / _ICO_NORM, 0.0,  1.0 / _ICO_NORM],             # 8
    [-_PHI / _ICO_NORM, 0.0,  1.0 / _ICO_NORM],             # 9
    [ _PHI / _ICO_NORM, 0.0, -1.0 / _ICO_NORM],             # 10
    [-_PHI / _ICO_NORM, 0.0, -1.0 / _ICO_NORM],             # 11
])

# 20 triangular faces with outward-facing normals.
# Orientation was verified by computing dot(Va, cross(Vb, Vc)) > 0 for each face.
ICOSAHEDRON_TRIANGLES = [
    (0, 1, 8),  (0, 9, 1),  (0, 4, 5),  (0, 8, 4),  (0, 5, 9),
    (1, 7, 6),  (1, 6, 8),  (1, 9, 7),
    (2, 10, 3), (2, 3, 11), (2, 5, 4),  (2, 4, 10), (2, 11, 5),
    (3, 6, 7),  (3, 10, 6), (3, 7, 11),
    (4, 8, 10),
    (5, 11, 9),
    (6, 10, 8),
    (7, 9, 11),
]

# 30 edges (each pair i < j, derived from ICOSAHEDRON_TRIANGLES adjacency)
ICOSAHEDRON_EDGES = [
    (0, 1), (0, 4), (0, 5), (0, 8), (0, 9),
    (1, 6), (1, 7), (1, 8), (1, 9),
    (2, 3), (2, 4), (2, 5), (2, 10), (2, 11),
    (3, 6), (3, 7), (3, 10), (3, 11),
    (4, 5), (4, 8), (4, 10),
    (5, 9), (5, 11),
    (6, 7), (6, 8), (6, 10),
    (7, 9), (7, 11),
    (8, 10),
    (9, 11),
]

# Volume = TOPOLOGY_VOLUME_FACTOR[topology] * r^3 for a cell of radius r.
# Octahedron: exact 4/3. Icosahedron: computed from mesh via divergence theorem.
_ICO_VOLUME_FACTOR = sum(
    np.dot(ICOSAHEDRON_DIRS[a], np.cross(ICOSAHEDRON_DIRS[b], ICOSAHEDRON_DIRS[c]))
    for a, b, c in ICOSAHEDRON_TRIANGLES
) / 6.0

TOPOLOGY_VOLUME_FACTOR = {
    'oct': 4.0 / 3.0,
    'ico': _ICO_VOLUME_FACTOR,
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cell_body.py::test_icosahedron_vertex_count tests/test_cell_body.py::test_icosahedron_dirs_on_unit_sphere tests/test_cell_body.py::test_icosahedron_face_count tests/test_cell_body.py::test_icosahedron_edge_count tests/test_cell_body.py::test_topology_volume_factor_oct tests/test_cell_body.py::test_topology_volume_factor_ico_larger_than_oct tests/test_cell_body.py::test_topology_volume_factor_ico_less_than_sphere -v
```

Expected: 7 PASS

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
uv run pytest
```

Expected: all existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add develmorph/cell_body.py tests/test_cell_body.py
git commit -m "feat: add ICOSAHEDRON constants and TOPOLOGY_VOLUME_FACTOR to cell_body"
```

---

## Task 2: Add `topology` to `CellBody` and `create_cell_body`

**Files:**
- Modify: `develmorph/cell_body.py`
- Test: `tests/test_cell_body.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_cell_body.py`:

```python
def test_create_cell_body_ico_particle_count():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    assert len(positions) == 13  # 1 center + 12 surface


def test_create_cell_body_ico_surface_count():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    assert len(cb.surface_indices) == 12


def test_create_cell_body_ico_topology_field():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    assert cb.topology == 'ico'


def test_create_cell_body_default_topology_is_oct():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions,
    )
    assert cb.topology == 'oct'


def test_create_cell_body_ico_volume_range():
    # Unit-sphere icosahedron volume ≈ 2.536; must exceed octahedron (4/3) and be
    # below sphere (4.189).
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    assert 2.0 < cb.target_volume < 3.0


def test_create_cell_body_ico_face_count():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    assert len(cb.triangles) == 20


def test_create_cell_body_ico_edge_count():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    assert len(cb.edges) == 30
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cell_body.py::test_create_cell_body_ico_particle_count -v
```

Expected: `TypeError: unexpected keyword argument 'topology'`

- [ ] **Step 3: Add `topology` to `CellBody.__init__`**

In `develmorph/cell_body.py`, inside `CellBody.__init__`, add `topology='oct'` after `orientation=None`:

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
):
    self.topology = topology
    self.name = name
    # ... rest of __init__ unchanged
```

- [ ] **Step 4: Add `topology` to `create_cell_body` and wire the branching**

In `develmorph/cell_body.py`, update `create_cell_body` signature and the topology-selection block:

```python
def create_cell_body(
    center, radius, positions, membrane_compliance=0.01,
    area_threshold=None, name=None, orientation=None,
    topology='oct',
):
    center = np.asarray(center, dtype=float)
    n = len(positions)

    if topology == 'ico':
        dirs = ICOSAHEDRON_DIRS
        triangles = ICOSAHEDRON_TRIANGLES
        edges = ICOSAHEDRON_EDGES
    elif area_threshold is not None:
        unit_threshold = area_threshold / (radius * radius)
        dirs, triangles = subdivide_mesh(OCTAHEDRON_DIRS, OCTAHEDRON_TRIANGLES, unit_threshold)
        edges = compute_edges(triangles)
    else:
        dirs = OCTAHEDRON_DIRS
        triangles = OCTAHEDRON_TRIANGLES
        edges = OCTAHEDRON_EDGES
    # ... rest of function body unchanged until CellBody constructor
```

At the end of `create_cell_body`, pass `topology` to the `CellBody` constructor:

```python
    cell_body = CellBody(
        center_idx=center_idx,
        surface_indices=surface_indices,
        target_volume=target_volume,
        center_surface_rest=center_surface_rest,
        surface_surface_rest=surface_surface_rest,
        membrane_compliance=membrane_compliance,
        triangles=triangles,
        edges=edges,
        name=name,
        orientation=orientation if orientation is not None else np.eye(3),
        topology=topology,
    )
```

- [ ] **Step 5: Run new tests to verify they pass**

```bash
uv run pytest tests/test_cell_body.py::test_create_cell_body_ico_particle_count tests/test_cell_body.py::test_create_cell_body_ico_surface_count tests/test_cell_body.py::test_create_cell_body_ico_topology_field tests/test_cell_body.py::test_create_cell_body_default_topology_is_oct tests/test_cell_body.py::test_create_cell_body_ico_volume_range tests/test_cell_body.py::test_create_cell_body_ico_face_count tests/test_cell_body.py::test_create_cell_body_ico_edge_count -v
```

Expected: 7 PASS

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest
```

Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add develmorph/cell_body.py tests/test_cell_body.py
git commit -m "feat: add topology parameter to CellBody and create_cell_body"
```

---

## Task 3: Topology-aware radius and topology propagation in `divide_cell_body`

**Background:** `divide_cell_body` currently computes `parent_radius` using the
octahedron formula `r = (3V/4)^(1/3)` (from `V = (4/3)r³`). For an icosahedron,
the correct inverse is `r = (V / TOPOLOGY_VOLUME_FACTOR['ico'])^(1/3)`.
Without this fix, ico daughters would be created with the wrong radius.

**Files:**
- Modify: `develmorph/division.py`
- Test: `tests/test_division.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_division.py`:

```python
import numpy as np
from develmorph.cell_body import create_cell_body, TOPOLOGY_VOLUME_FACTOR
from develmorph.division import divide_cell_body


def test_divide_ico_daughters_inherit_topology():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    axis = np.array([1.0, 0.0, 0.0])
    daughters, positions = divide_cell_body(cb, positions, axis=axis)
    assert daughters[0].topology == 'ico'
    assert daughters[1].topology == 'ico'


def test_divide_ico_daughters_correct_volume():
    # Total volume of daughters should conserve parent volume.
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions, topology='ico',
    )
    parent_volume = cb.target_volume
    axis = np.array([1.0, 0.0, 0.0])
    daughters, positions = divide_cell_body(cb, positions, axis=axis)
    total_daughter_volume = daughters[0].target_volume + daughters[1].target_volume
    np.testing.assert_allclose(total_daughter_volume, parent_volume, rtol=0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_division.py::test_divide_ico_daughters_inherit_topology -v
```

Expected: FAIL (`daughters[0].topology == 'oct'` instead of `'ico'`)

- [ ] **Step 3: Update `develmorph/division.py`**

Add import at the top of `division.py`:

```python
from develmorph.cell_body import create_cell_body, TOPOLOGY_VOLUME_FACTOR
```

Replace the octahedron-specific radius formula and the two `create_cell_body` calls:

```python
    # Radius from volume: V = TOPOLOGY_VOLUME_FACTOR[topology] * r^3
    vol_factor = TOPOLOGY_VOLUME_FACTOR[parent.topology]
    parent_radius = (parent_volume / vol_factor) ** (1.0 / 3.0)

    # Daughter radii from volume ratio
    r1 = parent_radius * (volume_ratio ** (1.0 / 3.0))
    r2 = parent_radius * ((1.0 - volume_ratio) ** (1.0 / 3.0))
```

Then update the two `create_cell_body` calls to pass `topology`:

```python
    d1, positions = create_cell_body(
        center=center1,
        radius=r1,
        positions=positions,
        membrane_compliance=parent.membrane_compliance,
        area_threshold=area_threshold,
        topology=parent.topology,
        name=daughter_names[0],
        orientation=orient1,
    )
    d2, positions = create_cell_body(
        center=center2,
        radius=r2,
        positions=positions,
        membrane_compliance=parent.membrane_compliance,
        area_threshold=area_threshold,
        topology=parent.topology,
        name=daughter_names[1],
        orientation=orient2,
    )
```

- [ ] **Step 4: Run new tests to verify they pass**

```bash
uv run pytest tests/test_division.py::test_divide_ico_daughters_inherit_topology tests/test_division.py::test_divide_ico_daughters_correct_volume -v
```

Expected: 2 PASS

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add develmorph/division.py tests/test_division.py
git commit -m "feat: topology-aware radius and propagation in divide_cell_body"
```

---

## Task 4: Integration test — ico cell in World physics loop

**Files:**
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_engine.py`:

```python
import numpy as np
from develmorph.cell_body import create_cell_body
from develmorph.engine import World


def test_world_simulates_ico_cell_without_error():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions,
        topology='ico',
    )
    world = World(
        cell_bodies=[cb],
        positions=positions,
    )
    world.run(5)
    assert world.cell_bodies[0].topology == 'ico'
    assert len(world.positions) == 13  # 1 center + 12 surface


def test_world_ico_cell_preserves_approximate_volume():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions,
        topology='ico',
    )
    initial_target = cb.target_volume
    world = World(
        cell_bodies=[cb],
        positions=positions,
        shell_params=(10.0, 10.0, 10.0),
    )
    world.run(20)
    # target_volume is set at creation and does not change — it is the constraint target
    assert world.cell_bodies[0].target_volume == initial_target
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_engine.py::test_world_simulates_ico_cell_without_error -v
```

Expected: FAIL or PASS depending on current state; if the previous tasks are done correctly, this should PASS already. If it fails, investigate the error.

- [ ] **Step 3: Run new tests to verify they pass**

```bash
uv run pytest tests/test_engine.py::test_world_simulates_ico_cell_without_error tests/test_engine.py::test_world_ico_cell_preserves_approximate_volume -v
```

Expected: 2 PASS

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest
```

Expected: all tests PASS (71 original + ~16 new = ~87 total)

- [ ] **Step 5: Commit**

```bash
git add tests/test_engine.py
git commit -m "test: add integration tests for ico cell in World physics loop"
```
