# Adaptive Surface Point Subdivision Design

## Goal

Increase surface point count for CellBody based on triangle area threshold, improving simulation stability for larger cells while keeping small cells lightweight.

## Problem

Current CellBody uses a fixed 6-point octahedron (8 triangles, 12 edges). For larger cells, this coarse mesh leads to:
- Unstable `_max_surface_dist` values (surface points distorted by competing constraints)
- Poor volume constraint accuracy
- Inaccurate collision/adhesion rest lengths

## Approach: Iterative Centroid Subdivision

Self-contained implementation using only numpy. No external mesh library needed.

### Algorithm

1. Start with octahedron mesh (6 directions, 8 triangles)
2. For each triangle, compute area = 0.5 * |cross(e1, e2)|
3. If area > threshold: add centroid of 3 vertices as new point, project to unit sphere
4. Replace that triangle with 3 sub-triangles sharing the centroid
5. Repeat until no triangle exceeds threshold
6. Derive edges automatically from triangle list

### Centroid Subdivision (1 triangle -> 3)

```
    v0                v0
   / \              / | \
  /   \    -->     / c  \
 /     \          / / \ \
v1------v2      v1---c---v2

c = normalize(v0 + v1 + v2) * radius
Creates: (v0,v1,c), (v1,v2,c), (v2,v0,c)
```

### Point Count Growth

Each subdivision step replaces 1 triangle with 3 (net +2 triangles, +1 point).
- Octahedron: 6 points, 8 triangles
- After 1 pass (all 8 subdivided): 14 points, 24 triangles
- After 2 passes: depends on threshold, typically 20-30 points

Area shrinks by ~1/3 per step, so convergence is fast (2-3 iterations typical).

## Changes

### `develmorph/cell_body.py`

**New functions:**
- `subdivide_mesh(directions, triangles, area_threshold)` - iterative centroid subdivision
  - Input: unit direction vectors (N,3), triangle list [(i,j,k),...], area threshold
  - Output: new directions (M,3), new triangle list
  - Iterates until all triangles are below threshold
- `compute_edges(triangles)` - extract unique edges from triangle list
  - Input: triangle list [(i,j,k),...]
  - Output: sorted list of (i,j) pairs

**Modified: `CellBody.__init__`**
- Accept `triangles` parameter (default: `OCTAHEDRON_TRIANGLES`)
- Accept `edges` parameter for surface-surface edges (default: `OCTAHEDRON_EDGES`)
- Store edges as instance attribute for `build_particle_arrays`

**Modified: `create_cell_body`**
- Add `area_threshold` parameter (default: `None` for backward compatibility)
- When set: call `subdivide_mesh` then `compute_edges` to get refined mesh
- Generate surface points from refined directions instead of `OCTAHEDRON_DIRS`

**Modified: `build_particle_arrays`**
- Use `cb.edges` (stored in CellBody) instead of hardcoded `OCTAHEDRON_EDGES`

### No changes needed

- `develmorph/project.py` - constraints already use `cb.surface_global_triangles()` and `cb.center_surface_edges`/`cb.surface_surface_edges`
- `develmorph/solver.py` - already reads edges from CellBody
- `develmorph/division.py` - uses `create_cell_body` (will pass through `area_threshold`)

### Threshold Guidelines

Octahedron face area for radius r: (sqrt(3)/2) * r^2
- r=0.5: area ~= 0.217 per face
- r=0.8: area ~= 0.554 per face

Suggested default when enabled: `area_threshold=0.2`
- r=0.5 cells: no subdivision (0.217 ~ threshold)
- r=0.8+ cells: subdivision triggered

## Backward Compatibility

- `area_threshold=None` (default) -> no subdivision, identical to current behavior
- All existing tests pass unchanged
- `build_particle_arrays` uses CellBody's stored edges instead of global constant

## Future Work (Not In Scope)

- Division: inherit parent surface points + regular polygon cut face
- Remeshing after constraint deformation
- Switch to trimesh if mesh operations become complex
