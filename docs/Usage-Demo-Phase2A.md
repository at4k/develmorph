# Usage Guide: Phase2-A Demos

This document explains how to run and interpret Phase2-A demo scripts.

---

## Primary Demo

Script location:

    scripts/demo_lineage_division_3d.py

---

## Purpose

This demo verifies that:

- division direction rules affect 3D structure
- lineage-driven behavior is respected
- time-series snapshots are interpretable

The demo intentionally bypasses:
- physics solvers
- force relaxation
- continuous-time dynamics

---

## How to Run

    uv run scripts/demo_lineage_division_3d.py

---

## Output Data

The demo records cell states into a DataFrame following
the schema defined in API-Contract.md.

Example rows:

- step=0, cell_id=0, lineage_id=0, divided=False
- step=1, cell_id=0, lineage_id=0, divided=True
- step=2, cell_id=1, lineage_id=0, divided=True
- step=2, cell_id=2, lineage_id=0, divided=True

---

## Interpreting step

- step is a division event index
- records correspond to pre-division snapshots
- newly created cells appear at the next step

This is intentional and consistent with Phase2-A design.

---

## Visualization

Visualization is performed via a helper such as:

    visualize_lineage_3d(df)

---

## What to Look For

- symmetry breaking across steps
- relative orientation of daughter cells
- structural changes when rules are modified

---

## Common Misinterpretations

- Only three cells appear
  → the final snapshot may not yet be recorded

- Cells do not move
  → Phase2-A has no physics; movement is geometric only

---

## Safe Modifications

Allowed:
- changing division rules
- changing number of steps
- changing initial polarity

Not allowed:
- changing CellLineage internals
- recording partial state
- implicit physics assumptions