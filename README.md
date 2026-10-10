# DevelMorph

DevelMorph is a developmental morphogenesis simulator for early embryonic cell mechanics in 3D.
The target is the *C. elegans* 1 -> 2 -> 4 cell stage inside an ellipsoidal eggshell: division rules come from a lineage file, and the physics decides where the daughter cells end up (for example, the 4-cell "diamond" arrangement).

## Status: paused

Development in this repository is **paused**. The main obstacle turned out to be the simulation engine itself (an XPBD constraint solver), not the biology layer. The engine is now being studied on its own in a separate repository:

- XPBD engine study: https://github.com/at4k/xpbd

This README records where the code stands and what was tried, so that the work can be resumed later.

## Current state of the code

Branch `feat-xpbd-solver` holds all of the work below. `main` only contains the early base code (point cells + projection).

- Tests: 205 passed, 2 xfailed (`uv run pytest`)
- Lineage layer (stable): `CellLineage` reads YAML division rules (axis in the cell's local frame, daughter orientations, volume ratios) up to the 8-cell stage. `LineageRecorder` stores snapshots.
- Physics layer (experimental): two cell models share one XPBD solver.
  - `World` + `CellBody`: deformable cells made of a center particle and a surface mesh (octahedron / icosahedron, optional adaptive subdivision).
  - `WorldPC` + `PhysicsCell`: center-only cells. The shape is reconstructed afterwards, either as a flattened sphere (`SurfaceShell`) or as a shell-clipped power diagram (Voronoi) cell.
- Visualization: matplotlib, napari time-lapse, and an ASCII TUI (`scripts/tui_cells.py`).

### Main modules

| Module | Role |
|---|---|
| `celllineage.py`, `lineages/*.yaml` | Division rules (how a cell divides, not when) |
| `division.py` | Volume-conserving division of `CellBody` / `PhysicsCell` |
| `cell_body.py` | Center + surface-mesh cell model |
| `physics_cell.py`, `contact.py` | Center-only cell model and analytical contact geometry |
| `project.py` | XPBD constraints: collision, shell, distance, volume, adhesion, triangle area, tetrahedral volume, Voronoi volume |
| `solver.py` | XPBD step functions (`step`, `step_pc`) and adhesion pair management |
| `engine.py` | `World` / `WorldPC` simulation loops |
| `power_diagram.py` | Shell-clipped power diagram (Voronoi) cells |
| `spatial_hash.py` | O(N) neighbor search |

## What was tried

Each row is one attempt, in order. Detailed specs and plans are in `docs/plans/` and `docs/superpowers/`.

| # | Approach | Result |
|---|---|---|
| 1 | Point cells with radius, overdamped motion, projection-based overlap/shell correction | Worked for simple division. Replaced by XPBD to get stiffness independent of time step and iteration count, and to express membrane, volume and adhesion as one kind of constraint (`docs/xpbd_roadmap.md`). |
| 2 | XPBD Phase A: rigid collision + shell constraints | Stable, but cells are rigid spheres. |
| 3 | XPBD Phase B: `CellBody` with surface points, distance (membrane) + volume constraints | Diverged when both constraints were stiff. Needed a soft volume constraint, a stiff membrane, per-iteration clamping and an inversion guard. |
| 4 | XPBD Phase C: cell adhesion | A two-way adhesion constraint was too soft to matter. Split into a hard collision (repulsion) and a soft attraction-only adhesion with attach/detach hysteresis. |
| 5 | Finer membranes: adaptive subdivision, icosahedron topology, triangle area and tetrahedral volume constraints | Better shapes, but many interacting compliance parameters. |
| 6 | 4-cell stage with `CellBody` | ABp collapsed; fixed with a zero center-shell margin and a center-relative inversion guard. A soft shell let surface points leak out; the shell had to be rigid. Physics alone gave a T-shape, not a diamond. |
| 7 | Wnt-like dorsal bias on EMS to force the diamond | Produced a diamond, but it was a hand-added force. Reverted. |
| 8 | Diamond from division geometry: P1 and AB divide along the D-V axis, settling gaps between divisions, P0 radius 1.3, `center_shell_margin_factor=0.4` | Diamond forms (EMS D-V offset ~0.6-0.75). Required careful tuning of timing, radii and margins. |
| 9 | Kinematic shell model (`WorldPC`): center-only cells, contact-area adhesion, reconstructed surfaces | Faster and simpler, and the diamond still forms. **KI-001:** rigid spheres cannot form the diamond and stay inside the shell at the same time, so 27-88% of cells end up partly outside the shell. |
| 10 | Voronoi / power diagram cells to resolve KI-001 (no overlap by construction, shell-clipped volumes) | Tessellation volume error ~1.2%. In the dynamic simulation, the Voronoi volume, collision and soft shell constraints work against each other: cells drift out of the shell and volume conservation fails (the 2 xfail tests). |

## Why the engine is studied separately

Most of the time went into getting XPBD constraints to agree with each other, not into the biology:

- the balance of compliance values (hard vs. soft) decided whether the result was stable, diverged, or leaked out of the shell;
- a representation that fixed one problem (rigid spheres, surface meshes, Voronoi cells) brought new conflicts between constraints;
- the fixes piled up as special-case parameters (margins, clamps, guards, timings).

Instead of adding more patches here, the XPBD engine is being studied on its own first, in the separate repository above. DevelMorph can then be rebuilt on a better-understood engine.

## Known issues

- **KI-001** (`docs/superpowers/specs/2026-05-23-kinematic-shell-design.md`): with rigid-sphere cells, the shell constraint and the diamond configuration cannot both be satisfied.
- **Voronoi mode**: volume conservation fails when the collision constraint and a soft shell are both active (`tests/test_voronoi_ki001.py`, `tests/test_worldpc.py`, marked xfail).

## Usage

```bash
uv sync                                   # install dependencies
uv run pytest                             # run tests
uv run python scripts/demo_4cell_stage.py # 4-cell stage with CellBody (XPBD)
uv run python scripts/demo_diamond_pc.py  # diamond formation with WorldPC
uv run python scripts/tui_cells.py        # ASCII view of a WorldPC run
```
