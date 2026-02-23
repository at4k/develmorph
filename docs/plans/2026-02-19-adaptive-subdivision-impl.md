# Adaptive Surface Point Subdivision Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add triangle-area-threshold-based centroid subdivision to CellBody so larger cells get more surface points automatically.

**Architecture:** `subdivide_mesh()` iteratively splits triangles whose area exceeds a threshold by inserting their centroid (projected to unit sphere). `compute_edges()` derives edge lists from the resulting triangle connectivity. `CellBody` and `create_cell_body` become mesh-topology-agnostic by storing edges/triangles as instance data instead of referencing global octahedron constants.

**Tech Stack:** Python 3.13, numpy, uv, pytest

---

### Task 1: `compute_edges` — extract edges from triangles

**Files:**
- Modify: `develmorph/cell_body.py` (add function after line 42, below `OCTAHEDRON_EDGES`)
- Test: `tests/test_cell_body.py`

**Step 1: Write the failing test**

Add to `tests/test_cell_body.py`:

```python
from develmorph.cell_body import compute_edges

def test_compute_edges_octahedron():
    """compute_edges extracts 12 unique edges from 8 octahedron triangles."""
    from develmorph.cell_body import OCTAHEDRON_TRIANGLES
    edges = compute_edges(OCTAHEDRON_TRIANGLES)
    assert len(edges) == 12
    # All edges should be sorted tuples (i < j)
    for a, b in edges:
        assert a < b

def test_compute_edges_single_triangle():
    """Single triangle produces 3 edges."""
    edges = compute_edges([(0, 1, 2)])
    assert len(edges) == 3
    assert set(edges) == {(0, 1), (0, 2), (1, 2)}
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cell_body.py::test_compute_edges_octahedron tests/test_cell_body.py::test_compute_edges_single_triangle -v`
Expected: FAIL with `ImportError: cannot import name 'compute_edges'`

**Step 3: Implement `compute_edges`**

Add to `develmorph/cell_body.py` after `OCTAHEDRON_EDGES`:

```python
def compute_edges(triangles):
    """
    Extract unique edges from a triangle list.

    Each edge is a sorted tuple (i, j) where i < j.
    Returns a sorted list for deterministic ordering.
    """
    edge_set = set()
    for a, b, c in triangles:
        edge_set.add((min(a, b), max(a, b)))
        edge_set.add((min(b, c), max(b, c)))
        edge_set.add((min(a, c), max(a, c)))
    return sorted(edge_set)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cell_body.py::test_compute_edges_octahedron tests/test_cell_body.py::test_compute_edges_single_triangle -v`
Expected: PASS

**Step 5: Run all existing tests to verify no regressions**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add develmorph/cell_body.py tests/test_cell_body.py
git commit -m "feat: add compute_edges utility for extracting edges from triangles"
```

---

### Task 2: `subdivide_mesh` — iterative centroid subdivision

**Files:**
- Modify: `develmorph/cell_body.py` (add function after `compute_edges`)
- Test: `tests/test_cell_body.py`

**Step 1: Write the failing tests**

Add to `tests/test_cell_body.py`:

```python
from develmorph.cell_body import subdivide_mesh, OCTAHEDRON_DIRS, OCTAHEDRON_TRIANGLES

def test_subdivide_mesh_no_subdivision_when_below_threshold():
    """When all triangles are below threshold, mesh is unchanged."""
    dirs, tris = subdivide_mesh(OCTAHEDRON_DIRS, OCTAHEDRON_TRIANGLES, area_threshold=10.0)
    assert len(dirs) == 6
    assert len(tris) == 8

def test_subdivide_mesh_all_subdivided():
    """With a low threshold, all 8 octahedron faces are subdivided."""
    # Octahedron face area on unit sphere: sqrt(3)/2 ~= 0.866
    dirs, tris = subdivide_mesh(OCTAHEDRON_DIRS, OCTAHEDRON_TRIANGLES, area_threshold=0.1)
    # After first pass: 8 faces -> 24 faces, 6 -> 14 points
    # May iterate further if still above threshold
    assert len(dirs) > 6
    assert len(tris) > 8
    # Every triangle should be below threshold
    for a, b, c in tris:
        v0, v1, v2 = dirs[a], dirs[b], dirs[c]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        assert area <= 0.1 + 1e-10

def test_subdivide_mesh_points_on_unit_sphere():
    """All output direction vectors lie on the unit sphere."""
    dirs, tris = subdivide_mesh(OCTAHEDRON_DIRS, OCTAHEDRON_TRIANGLES, area_threshold=0.3)
    for d in dirs:
        assert abs(np.linalg.norm(d) - 1.0) < 1e-10

def test_subdivide_mesh_centroid_split_produces_3_triangles():
    """Subdividing 1 triangle produces exactly 3 sub-triangles."""
    # Use a single large triangle
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    tris = [(0, 1, 2)]
    new_dirs, new_tris = subdivide_mesh(dirs, tris, area_threshold=0.01)
    # At minimum, after first pass: 1 -> 3 triangles, 3 -> 4 points
    assert len(new_dirs) >= 4
    assert len(new_tris) >= 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cell_body.py -k "subdivide_mesh" -v`
Expected: FAIL with `ImportError: cannot import name 'subdivide_mesh'`

**Step 3: Implement `subdivide_mesh`**

Add to `develmorph/cell_body.py` after `compute_edges`:

```python
def subdivide_mesh(directions, triangles, area_threshold, max_iterations=10):
    """
    Iteratively subdivide triangles whose area exceeds the threshold.

    For each oversized triangle, insert its centroid (projected to unit sphere)
    and replace the triangle with 3 sub-triangles.

    Args:
        directions: (N, 3) unit direction vectors on the sphere
        triangles: list of (i, j, k) index tuples
        area_threshold: maximum allowed triangle area
        max_iterations: safety limit on subdivision passes

    Returns:
        (new_directions, new_triangles): refined mesh
    """
    dirs = list(directions)
    tris = list(triangles)

    for _ in range(max_iterations):
        new_tris = []
        any_subdivided = False
        for a, b, c in tris:
            v0 = np.asarray(dirs[a])
            v1 = np.asarray(dirs[b])
            v2 = np.asarray(dirs[c])
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            if area > area_threshold:
                # Add centroid projected to unit sphere
                centroid = (v0 + v1 + v2) / 3.0
                centroid = centroid / np.linalg.norm(centroid)
                ci = len(dirs)
                dirs.append(centroid)
                new_tris.append((a, b, ci))
                new_tris.append((b, c, ci))
                new_tris.append((c, a, ci))
                any_subdivided = True
            else:
                new_tris.append((a, b, c))
        tris = new_tris
        if not any_subdivided:
            break

    return np.array(dirs), tris
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cell_body.py -k "subdivide_mesh" -v`
Expected: PASS

**Step 5: Run all tests**

Run: `uv run pytest -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add develmorph/cell_body.py tests/test_cell_body.py
git commit -m "feat: add subdivide_mesh for iterative centroid subdivision"
```

---

### Task 3: Make `CellBody` mesh-topology-agnostic

**Files:**
- Modify: `develmorph/cell_body.py:56-84` (CellBody.__init__)
- Test: `tests/test_cell_body.py`

**Step 1: Write the failing test**

Add to `tests/test_cell_body.py`:

```python
def test_cell_body_stores_custom_edges():
    """CellBody stores edges passed via constructor, not hardcoded OCTAHEDRON_EDGES."""
    custom_edges = [(0, 1), (1, 2), (2, 0)]
    cb = CellBody(
        center_idx=0,
        surface_indices=[1, 2, 3],
        target_volume=1.0,
        center_surface_rest=[1.0, 1.0, 1.0],
        surface_surface_rest=[1.0, 1.0, 1.0],
        triangles=[(0, 1, 2)],
        edges=custom_edges,
    )
    assert cb.edges == custom_edges
    assert cb.triangles == [(0, 1, 2)]
    # surface_surface_edges should use custom edges mapped to global indices
    assert len(cb.surface_surface_edges) == 3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cell_body.py::test_cell_body_stores_custom_edges -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'triangles'`

**Step 3: Modify `CellBody.__init__`**

Replace `CellBody.__init__` in `develmorph/cell_body.py`:

```python
class CellBody:
    """
    A cell represented as center point + surface points on a triangulated mesh.

    All indices refer to positions in the flat particle array.
    """

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
    ):
        self.center_idx = center_idx
        self.surface_indices = list(surface_indices)
        self.triangles = triangles if triangles is not None else OCTAHEDRON_TRIANGLES
        self.edges = edges if edges is not None else OCTAHEDRON_EDGES
        self.target_volume = target_volume
        self.membrane_compliance = membrane_compliance

        # Edges for distance constraints (with rest lengths)
        self.center_surface_edges = [(center_idx, si) for si in surface_indices]
        self.center_surface_rest = list(center_surface_rest)
        self.surface_surface_edges = [
            (surface_indices[a], surface_indices[b]) for a, b in self.edges
        ]
        self.surface_surface_rest = list(surface_surface_rest)
```

**Key change:** `self.edges` replaces hardcoded `OCTAHEDRON_EDGES`, and `surface_surface_edges` uses `self.edges`.

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cell_body.py::test_cell_body_stores_custom_edges -v`
Expected: PASS

**Step 5: Run all tests to verify backward compatibility**

Run: `uv run pytest -v`
Expected: All PASS (defaults to `OCTAHEDRON_TRIANGLES` / `OCTAHEDRON_EDGES`)

**Step 6: Commit**

```bash
git add develmorph/cell_body.py tests/test_cell_body.py
git commit -m "refactor: make CellBody accept custom triangles and edges"
```

---

### Task 4: Update `create_cell_body` to support subdivision

**Files:**
- Modify: `develmorph/cell_body.py:89-140` (create_cell_body)
- Test: `tests/test_cell_body.py`

**Step 1: Write the failing tests**

Add to `tests/test_cell_body.py`:

```python
def test_create_cell_body_with_subdivision():
    """create_cell_body with area_threshold produces more surface points."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        area_threshold=0.3,
    )
    # Octahedron face area at r=1: sqrt(3)/2 ~= 0.866 > 0.3
    # Should subdivide, producing more than 6 surface points
    assert len(cb.surface_indices) > 6
    assert len(cb.triangles) > 8
    # All surface points should be at distance radius from center
    center = positions[cb.center_idx]
    for si in cb.surface_indices:
        dist = np.linalg.norm(positions[si] - center)
        assert abs(dist - 1.0) < 1e-10

def test_create_cell_body_no_subdivision_default():
    """create_cell_body without area_threshold produces standard 6 surface points."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(cb.surface_indices) == 6
    assert len(cb.triangles) == 8

def test_create_cell_body_subdivision_volume_correct():
    """Subdivided cell has correct target volume from its mesh geometry."""
    positions = np.zeros((0, 3))
    r = 1.0
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=r,
        positions=positions,
        area_threshold=0.3,
    )
    # Volume should be computed from actual mesh, close to (4/3)*r^3 for
    # a mesh approximating an octahedron (with more faces, it gets closer
    # but is still based on surface-to-origin signed volume)
    assert cb.target_volume > 0
    # With more points on unit sphere, volume should be close to but not
    # exactly equal to octahedron volume (subdivision changes mesh slightly)
    octahedron_vol = (4.0 / 3.0) * r ** 3
    # Should be within 20% (centroid-subdivided mesh is slightly different)
    assert abs(cb.target_volume - octahedron_vol) / octahedron_vol < 0.2

def test_create_cell_body_subdivision_edges_match_triangles():
    """Edge count from compute_edges matches stored edges."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        area_threshold=0.3,
    )
    from develmorph.cell_body import compute_edges
    expected_edges = compute_edges(cb.triangles)
    assert len(cb.edges) == len(expected_edges)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cell_body.py -k "subdivision" -v`
Expected: FAIL with `TypeError: create_cell_body() got an unexpected keyword argument 'area_threshold'`

**Step 3: Modify `create_cell_body`**

Replace the function in `develmorph/cell_body.py`:

```python
def create_cell_body(center, radius, positions, membrane_compliance=0.01, area_threshold=None):
    """
    Create a CellBody and append its particles to the positions array.

    Args:
        center: (3,) center position
        radius: cell radius
        positions: existing (N, 3) positions array to append to
        membrane_compliance: membrane softness
        area_threshold: if set, subdivide triangles whose area exceeds this
            value. Area is measured on the actual surface (scaled by radius^2).
            None means no subdivision (standard 6-point octahedron).

    Returns:
        (cell_body, new_positions): CellBody and updated positions array
    """
    center = np.asarray(center, dtype=float)
    n = len(positions)

    # Determine mesh topology
    if area_threshold is not None:
        # Subdivide on unit sphere, then scale by radius
        # Convert area threshold from world-space to unit-sphere-space
        unit_threshold = area_threshold / (radius * radius)
        dirs, triangles = subdivide_mesh(OCTAHEDRON_DIRS, OCTAHEDRON_TRIANGLES, unit_threshold)
        edges = compute_edges(triangles)
    else:
        dirs = OCTAHEDRON_DIRS
        triangles = OCTAHEDRON_TRIANGLES
        edges = OCTAHEDRON_EDGES

    # Center point
    center_idx = n
    new_points = [center]

    # Surface points
    surface_indices = []
    for i, d in enumerate(dirs):
        surface_indices.append(n + 1 + i)
        new_points.append(center + radius * d)

    if len(positions) == 0:
        new_positions = np.array(new_points)
    else:
        new_positions = np.vstack([positions, new_points])

    # Compute target volume from actual mesh geometry
    target_volume = 0.0
    for a, b, c in triangles:
        p0 = new_positions[surface_indices[a]]
        p1 = new_positions[surface_indices[b]]
        p2 = new_positions[surface_indices[c]]
        target_volume += np.dot(p0, np.cross(p1, p2))
    target_volume /= 6.0

    # Compute rest lengths from initial geometry
    center_surface_rest = [
        np.linalg.norm(new_positions[si] - new_positions[center_idx])
        for si in surface_indices
    ]
    surface_surface_rest = [
        np.linalg.norm(
            new_positions[surface_indices[a]] - new_positions[surface_indices[b]]
        )
        for a, b in edges
    ]

    cell_body = CellBody(
        center_idx=center_idx,
        surface_indices=surface_indices,
        target_volume=target_volume,
        center_surface_rest=center_surface_rest,
        surface_surface_rest=surface_surface_rest,
        membrane_compliance=membrane_compliance,
        triangles=triangles,
        edges=edges,
    )

    return cell_body, new_positions
```

**Important detail:** The `area_threshold` is specified in world-space (scaled surface area). Inside the function, we convert it to unit-sphere-space by dividing by `radius^2`, because `subdivide_mesh` operates on unit direction vectors. This means the same `area_threshold` value works correctly for cells of different sizes — larger cells subdivide more.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cell_body.py -k "subdivision" -v`
Expected: PASS

**Step 5: Run all tests**

Run: `uv run pytest -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add develmorph/cell_body.py tests/test_cell_body.py
git commit -m "feat: add area_threshold to create_cell_body for adaptive subdivision"
```

---

### Task 5: Update `build_particle_arrays` to use stored edges

**Files:**
- Modify: `develmorph/cell_body.py:150-190` (build_particle_arrays)
- Test: `tests/test_cell_body.py`

**Step 1: Write the failing test**

Add to `tests/test_cell_body.py`:

```python
def test_build_particle_arrays_with_subdivided_cell():
    """build_particle_arrays works correctly with subdivided cells."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        area_threshold=0.3,
    )
    n_surface = len(cb.surface_indices)
    n_edges = len(cb.edges)

    new_pos, new_inv = build_particle_arrays([cb], positions)

    assert len(new_pos) == 1 + n_surface  # center + surface
    assert len(cb.surface_surface_edges) == n_edges
    # center-surface edges should match surface count
    assert len(cb.center_surface_edges) == n_surface
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cell_body.py::test_build_particle_arrays_with_subdivided_cell -v`
Expected: FAIL (because `build_particle_arrays` uses hardcoded `OCTAHEDRON_EDGES`, causing IndexError for subdivided cells with >6 surface points)

**Step 3: Modify `build_particle_arrays`**

In `develmorph/cell_body.py`, change `build_particle_arrays`:

Replace the line:
```python
        cb.surface_surface_edges = [
            (cb.surface_indices[a], cb.surface_indices[b]) for a, b in OCTAHEDRON_EDGES
        ]
```

With:
```python
        cb.surface_surface_edges = [
            (cb.surface_indices[a], cb.surface_indices[b]) for a, b in cb.edges
        ]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cell_body.py::test_build_particle_arrays_with_subdivided_cell -v`
Expected: PASS

**Step 5: Run all tests**

Run: `uv run pytest -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add develmorph/cell_body.py tests/test_cell_body.py
git commit -m "fix: build_particle_arrays uses CellBody.edges instead of hardcoded OCTAHEDRON_EDGES"
```

---

### Task 6: Pass `area_threshold` through `divide_cell_body`

**Files:**
- Modify: `develmorph/division.py`
- Test: `tests/test_division.py`

**Step 1: Write the failing test**

Add to `tests/test_division.py`:

```python
def test_divide_cell_body_with_area_threshold():
    """Daughter cells inherit area_threshold from division parameters."""
    positions = np.zeros((0, 3))
    parent, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        area_threshold=0.3,
    )
    daughters, new_positions = divide_cell_body(
        parent, positions,
        axis=np.array([1.0, 0.0, 0.0]),
        area_threshold=0.3,
    )
    # Daughters should have more than 6 surface points if their radius
    # is large enough to trigger subdivision at threshold 0.3
    # At minimum, the function should not crash
    assert len(daughters) == 2
    for d in daughters:
        assert len(d.surface_indices) >= 6
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_division.py::test_divide_cell_body_with_area_threshold -v`
Expected: FAIL with `TypeError: divide_cell_body() got an unexpected keyword argument 'area_threshold'`

**Step 3: Modify `divide_cell_body`**

In `develmorph/division.py`, add `area_threshold=None` parameter and pass through to `create_cell_body`:

```python
def divide_cell_body(parent, positions, axis=None, volume_ratio=0.5, area_threshold=None):
    """
    Divide a CellBody into two daughters.

    ...existing docstring...
        area_threshold: if set, passed to create_cell_body for adaptive subdivision

    Returns:
        (daughters, new_positions): list of 2 CellBodies and updated positions
    """
    if axis is None:
        axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    parent_center = positions[parent.center_idx].copy()
    parent_volume = parent.target_volume

    # Octahedron volume: V = (4/3) * r^3, so r = (3V/4)^(1/3)
    parent_radius = (3.0 * parent_volume / 4.0) ** (1.0 / 3.0)

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
        center=center1,
        radius=r1,
        positions=positions,
        membrane_compliance=parent.membrane_compliance,
        area_threshold=area_threshold,
    )
    d2, positions = create_cell_body(
        center=center2,
        radius=r2,
        positions=positions,
        membrane_compliance=parent.membrane_compliance,
        area_threshold=area_threshold,
    )

    return [d1, d2], positions
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_division.py::test_divide_cell_body_with_area_threshold -v`
Expected: PASS

**Step 5: Run all tests**

Run: `uv run pytest -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add develmorph/division.py tests/test_division.py
git commit -m "feat: pass area_threshold through divide_cell_body to create_cell_body"
```

---

### Task 7: Integration test — subdivided cell through XPBD solver

**Files:**
- Test: `tests/test_solver.py`

**Step 1: Write the integration test**

Add to `tests/test_solver.py`:

```python
from develmorph.cell_body import create_cell_body, build_particle_arrays

def test_step_with_subdivided_cell():
    """XPBD step works correctly with a subdivided cell (>6 surface points)."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        area_threshold=0.3,
    )
    n_particles = len(positions)
    velocities = np.zeros((n_particles, 3))
    inv_masses = np.ones(n_particles)

    new_pos, new_vel = step(
        positions, velocities, inv_masses,
        dt=0.01,
        cell_bodies=[cb],
        shell_params=(5.0, 5.0, 5.0),
        n_iter=5,
    )
    assert new_pos.shape == (n_particles, 3)
    # Cell should remain roughly spherical (no NaN, no explosion)
    center = new_pos[cb.center_idx]
    for si in cb.surface_indices:
        dist = np.linalg.norm(new_pos[si] - center)
        assert dist < 3.0  # sanity: not exploded
        assert not np.any(np.isnan(new_pos[si]))
```

**Step 2: Run test**

Run: `uv run pytest tests/test_solver.py::test_step_with_subdivided_cell -v`
Expected: PASS (solver already uses `cb.center_surface_edges`, `cb.surface_surface_edges`, and `cb.surface_global_triangles()`)

**Step 3: Run all tests**

Run: `uv run pytest -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/test_solver.py
git commit -m "test: add integration test for XPBD solver with subdivided cells"
```

---

### Summary

| Task | What | Points after | Triangles after |
|------|------|-------------|----------------|
| 1 | `compute_edges` utility | — | — |
| 2 | `subdivide_mesh` algorithm | — | — |
| 3 | `CellBody` accepts custom topology | — | — |
| 4 | `create_cell_body` with `area_threshold` | 6 → 14+ | 8 → 24+ |
| 5 | `build_particle_arrays` uses stored edges | — | — |
| 6 | `divide_cell_body` passes `area_threshold` | — | — |
| 7 | Integration test with XPBD solver | — | — |
