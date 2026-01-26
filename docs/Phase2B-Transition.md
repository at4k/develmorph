# Transition from Phase2-A to Phase2-B

This document clarifies what Phase2-A establishes
and how Phase2-B builds upon it.

---

## Phase2-A Invariants

The following concepts are fixed after Phase2-A:

- event-based time (step)
- snapshot-based recording
- separation of division rule, execution, and recording
- lineage as a rule container, not a controller

These assumptions must remain valid in Phase2-B.

---

## What Phase2-B Adds

Phase2-B introduces environmental context.

New elements include:
- polarity fields
- spatial sampling
- context-dependent division
- early fate bias (not commitment)

---

## Interface Evolution

Phase2-A logic:

    division_axis = rule(step, polarity)

Phase2-B logic:

    polarity = field.sample(position)
    division_axis = rule(step, polarity)

No change is required to:
- Recorder
- CellLineage public API
- demo structure

---

## Design Constraints

Phase2-B must:

- preserve Phase2-A demos without modification
- allow rule injection without refactoring
- keep polarity external to Cell objects

---

## Fate Signals (Preview)

Phase2-B may introduce:
- polarity history
- division history
- spatial context

These are signals, not fate commitments.

Fate commitment is deferred to Phase3.

---

## Summary

Phase2-A provides the structural skeleton.
Phase2-B adds environment-aware behavior.

The transition must be additive, not disruptive.