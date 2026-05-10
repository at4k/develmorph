# Icosahedron Cell Topology Design

Date: 2026-05-10  
Branch: feat-xpbd-solver  
Status: Approved

## Overview

Add icosahedron (正二十面体) as a selectable cell topology alongside the existing
octahedron, improving sphere approximation quality and membrane deformation resolution.

## Motivation

- Better sphere approximation: more triangular faces (20 vs 8) reduce volume
  discretization error
- More deformation degrees of freedom: 12 surface vertices (vs 6) allow finer
  local membrane deformation
- Selectable: octahedron remains the default; icosahedron is opt-in per cell

## Architecture

### New constants in `cell_body.py`

```python
ICOSAHEDRON_DIRS       # (12, 3) ndarray - unit-sphere vertices via golden ratio
ICOSAHEDRON_TRIANGLES  # list of 20 (i, j, k) tuples
ICOSAHEDRON_EDGES      # list of 30 (i, j) tuples
```

Vertices are derived from the golden ratio phi = (1+sqrt(5))/2:
- (0, ±1, ±phi) normalized  → 4 vertices
- (±1, ±phi, 0) normalized  → 4 vertices
- (±phi, 0, ±1) normalized  → 4 vertices

### `CellBody` changes

Add `topology: str` field (default `'oct'`). No other fields change.

### `create_cell_body()` API

```python
def create_cell_body(
    center, radius, positions,
    topology='oct',        # 'oct' | 'ico'   <-- new
    membrane_compliance=0.01,
    area_threshold=None,
    name=None,
    orientation=None,
):
```

Internal branching selects the appropriate dirs/triangles/edges constant set.
All downstream logic (volume computation, rest lengths, CellBody construction)
is shared and unchanged.

### `division.py`

`divide_cell_body()` reads `cell_body.topology` and passes it to both
`create_cell_body()` calls for daughter cells.

### `engine.py`

`World.add_cell()` accepts an optional `topology='oct'` keyword argument and
forwards it to `create_cell_body()`.

### Files not changed

- `project.py` (constraints are topology-independent)
- `solver.py`
- `viz.py`
- All YAML lineage files

## Particle count comparison

| topology | surface pts | total particles/cell |
|----------|-------------|----------------------|
| `'oct'`  | 6           | 7                    |
| `'ico'`  | 12          | 13                   |

4-cell simulation: 28 → 52 particles (acceptable for performance).

## Data flow

```
World.add_cell(topology='ico')
  → create_cell_body(topology='ico')
      → selects ICOSAHEDRON_DIRS / TRIANGLES / EDGES
      → builds 13 particles
      → computes target_volume from mesh
      → CellBody(topology='ico', ...)
  → appended to world.cell_bodies

divide_cell_body(parent_cb, ...)
  → topology = parent_cb.topology        # inherit
  → create_cell_body(..., topology=topology)  # daughter A
  → create_cell_body(..., topology=topology)  # daughter B
```

## Testing plan

All 71 existing tests pass unchanged (topology defaults to 'oct').

New tests added to `test_cell_body.py`:

| test | assertion |
|------|-----------|
| `test_icosahedron_dirs_unit_sphere` | all 12 vertices have norm == 1.0 |
| `test_icosahedron_face_count` | len(ICOSAHEDRON_TRIANGLES) == 20 |
| `test_icosahedron_edge_count` | len(ICOSAHEDRON_EDGES) == 30 |
| `test_create_cell_body_ico_particle_count` | 13 particles created |
| `test_create_cell_body_ico_volume` | volume in reasonable range vs sphere volume |
| `test_create_cell_body_oct_unchanged` | topology omitted → same as before |

New test added to `test_division.py`:

| test | assertion |
|------|-----------|
| `test_divide_ico_daughters_inherit_topology` | both daughters have topology='ico' |

## Interaction with `area_threshold`

`area_threshold` triggers adaptive subdivision of the octahedron base mesh.
When `topology='ico'`, `area_threshold` is ignored — icosahedron does not support
adaptive subdivision in this design. The two parameters are mutually exclusive;
combining them raises no error but subdivision is silently skipped.

## Backward compatibility

`topology` defaults to `'oct'` in all new function signatures. All existing
call sites require no changes.
