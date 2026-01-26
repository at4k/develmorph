# Phase2-A Overview

## Purpose

Phase2-A establishes the minimal **developmental kernel** of DevelMorph.

The goal is **not biological realism**, but to verify that **division rules shape 3D structure** when lineage, space, and time are coupled.

In short:

> Phase2-A answers the question:  
> *If I control how cells divide, can I reliably control what structure emerges?*

---

## Scope of Phase2-A

Phase2-A deliberately limits complexity.

### Included
- Deterministic cell division
- Explicit lineage tracking
- Discrete division steps
- 3D spatial embedding
- Time-series recording and visualization

### Excluded
- Continuous-time physics
- Mechanical relaxation
- Fate commitment
- Tissue-level constraints

This separation is intentional: **structure before realism**.

---

## Conceptual Model

### Time

Time is **event-based**, not continuous.

- `step` represents a **division event index**
- Each step corresponds to one round of division decisions

There is no notion of velocity, force, or numerical integration.

---

### State

At each step, the system state is defined by:

- The set of existing cells
- Their positions in 3D space
- Their lineage identity

State is captured as a **snapshot**, not a delta.

---

## Core Components

### Cell

A `Cell` is a minimal state container.

It holds:
- `cell_id`
- `lineage_id`
- `position`

A Cell:
- Does **not** decide when to divide
- Does **not** decide how to divide

This keeps the model explicit and debuggable.

---

### CellLineage

`CellLineage` defines **how division decisions are made — but not when**.

Responsibilities:
- Hold `lineage_id`
- Store a division axis rule (callable)
- Provide a division axis when queried

Critically:

> CellLineage does **not** hard-code division behavior.

Rules are injected, making lineage behavior configurable and testable.

---

### Division Rule

A division rule is a pure function:

- Input: `step` index, `polarity` vector
- Output: a 3D direction vector

Rules are interpreted **relative to polarity**, not absolute space.

This design anticipates Phase2-B, where polarity will be sampled from a field.

---

### Recorder

The recorder captures the full system state at each step.

- All cells are recorded
- Each record is a snapshot
- Output is a structured table (e.g. DataFrame)

The recorder:
- Does **not** interpret lineage
- Does **not** enforce semantics

It exists purely for inspection and visualization.

---

### Demo Scripts

Demo scripts are **experimental scaffolding**, not production code.

Their purpose:
- Make structure visible
- Expose conceptual bugs
- Validate design assumptions

In Phase2-A, demos are treated as **experiments**, not tests.

---

## Success Criteria

Phase2-A is considered complete when:

- Division behavior is rule-driven, not hard-coded
- Lineage identity is preserved across divisions
- 3D structure reflects division rules
- Time-series data can be visualized meaningfully

Cell count is intentionally small.

---

## Transition to Phase2-B

Phase2-A establishes the structural skeleton.

Phase2-B will extend it by:
- Introducing polarity fields
- Making division context-dependent
- Beginning fate differentiation

Phase2-A exists to ensure these extensions have a solid foundation.