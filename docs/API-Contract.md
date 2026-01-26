# API Contract (Phase2)

This document defines the stable public interfaces for Phase2.
Any change to these interfaces must be accompanied by updates to:
- tests
- demo scripts
- this document

This file acts as a contract between design, implementation, and experiments.

---

## CellLineage

### Responsibility

CellLineage encapsulates division rules, not division execution.
It defines how a cell divides, but never when.

---

### Public Interface

Class signature:

    class CellLineage:
        def __init__(
            self,
            lineage_id: int,
            division_axis_rule: Callable | None = None,
        )

        def decide_division_axis(
            self,
            step: int,
            polarity: np.ndarray | None = None,
        ) -> np.ndarray

---

### Guarantees

- Returned axis is a normalized 3D vector
- Deterministic for identical inputs
- No mutation of internal state
- Independent of global state

---

### Non-Responsibilities

- Does not perform division
- Does not manage time or steps
- Does not know about Recorder or visualization

---

## Division Axis Rule

### Definition

A division axis rule is a pure callable.

Signature:

    def division_axis_rule(
        step: int,
        polarity: np.ndarray | None,
    ) -> np.ndarray

---

### Requirements

- Must return a non-zero vector
- Deterministic for identical inputs
- Interpreted relative to polarity, not absolute coordinates

---

## LineageRecorder

### Responsibility

Capture snapshot-based system state at each division step.

---

### Public Interface

    class LineageRecorder:
        def record(
            self,
            step: int,
            cell_id: int,
            lineage_id: int,
            position: np.ndarray,
            divided: bool,
        )

        def to_dataframe(self) -> pandas.DataFrame

---

### DataFrame Schema

- step (int): division event index
- cell_id (int): unique cell identifier
- lineage_id (int): lineage identifier
- x, y, z (float): 3D position
- divided (bool): whether the cell divided at this step

---

### Guarantees

- One row per cell per step
- Records are immutable
- No implicit aggregation or filtering

---

## Solver Compatibility (Phase2)

Phase2 tests and demos assume the following solver signature:

    def step(
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        dt: float,
    ) -> np.ndarray

Any change requires explicit test updates.

---

## Demo Scripts

Demo scripts may:
- bypass physics
- use simplified cell proxies
- hardcode parameters

Demo scripts must not:
- modify core APIs
- depend on private/internal attributes