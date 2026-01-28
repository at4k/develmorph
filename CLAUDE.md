# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DevelMorph is a developmental morphogenesis simulator for modeling early embryonic cell mechanics in 3D. This project uses **uv** for Python package management. The project simulates cell division with projection-based constraint correction, focusing on how division rules shape 3D structure when lineage, space, and time are coupled.

## Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_celllineage_phase2a.py

# Run a specific test
uv run pytest tests/test_celllineage_phase2a.py::test_celllineage_basic_division_axis

# Run demo scripts
uv run python scripts/demo_lineage_division_3d.py

# Add a new dependency
uv add <package-name>
```

## Architecture

### Core Components

**CellLineage** (`develmorph/celllineage.py`): Encapsulates division rules loaded from YAML configuration. Defines *how* a cell divides (axis, daughter orientations) but not *when*. Division rules are interpreted relative to a cell's local orientation frame (e1, e2, e3).

**LineageRecorder** (`develmorph/lineage_recorder.py`): Captures snapshot-based system state at each division step. Outputs a DataFrame with columns: step, cell_id, lineage_id, x, y, z, divided.

**Division** (`develmorph/division.py`): Performs the geometric operation of splitting one cell into two daughters along a specified axis, adjusting radii to conserve volume.

**Solver** (`develmorph/solver.py`): Advances physical state by one time step using overdamped dynamics with projection-based constraints.

**Projection** (`develmorph/project.py`): Two constraint functions:
- `project_overlap`: Iteratively removes cell-cell overlaps
- `project_shell`: Projects cells inside an ellipsoidal boundary

**PolarityField** (`develmorph/polarity.py`): Defines global polarity axis (e.g., anterior-posterior) for position-dependent polarity values.

### Division Rule System

Division rules are defined in YAML files (see `develmorph/lineages/celegans_early.yaml`). Each rule specifies:
- Which cells it applies to (based on orientation)
- Division axis (e1, e2, or e3 in local frame)
- Daughter orientation transformations (inherit, invert_e1, rotate_e1_e2)

### Phase Structure

The project follows a phased development approach:
- **Phase2-A**: Deterministic cell division with explicit lineage tracking (current focus)
- **Phase2-B**: Will add polarity fields and context-dependent division
- **Phase2-C**: Will add fate differentiation

## Key Design Decisions

- Time is event-based (division steps), not continuous physics
- State is captured as snapshots, not deltas
- Cells do not decide when/how to divide; CellLineage provides rules when queried
- Division axes are interpreted relative to local polarity, not absolute coordinates
- Demo scripts are experimental scaffolding that may bypass physics layers
