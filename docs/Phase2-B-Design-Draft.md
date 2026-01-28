Phase2-B Design Draft

Purpose

Phase2-B extends Phase2-A by introducing cell-intrinsic orientation and division timing heterogeneity, while still avoiding full physical realism.

The central question of Phase2-B is:

If cells carry their own orientation and divide at different times, can we reproduce early embryonic arrangements (e.g. C. elegans 4-cell stage) in a controlled and inspectable way?

Phase2-B is about causal ordering and geometry, not forces.

⸻

Positioning Relative to Phase2-A

Phase2-A established:
	•	Discrete division steps
	•	Lineage-driven division direction
	•	Explicit state recording
	•	3D structural emergence

Phase2-B adds:
	•	Cell-level orientation (cell orientation)
	•	Per-cell division timing (asynchrony)
	•	Orientation inheritance and update on division

What remains out of scope:
	•	Continuous mechanics
	•	Collision resolution by forces
	•	Tissue-scale constraints

⸻

Core Concept: Cell Orientation

Each cell carries an orientation frame, not a global polarity field.

This reflects biological reality:
	•	Early embryonic cells are enclosed (e.g. eggshell)
	•	No reliable external gradient exists
	•	Division orientation is defined relative to the cell itself

Orientation Representation

Each cell holds an orthonormal basis:
	•	e1: primary axis (division-relevant)
	•	e2: secondary axis
	•	e3: normal (e1 × e2)

This frame:
	•	Moves with the cell
	•	Is inherited by daughter cells
	•	Can be rotated during division events

⸻

Cell Responsibilities (Phase2-B)

A Cell is no longer a pure container.

It now holds:
	•	cell_id
	•	lineage_id
	•	position
	•	orientation (e1, e2, e3)
	•	next_division_step

A Cell:
	•	Does not decide how to divide
	•	Does not decide why to divide
	•	Does know when it is scheduled to divide

This keeps causality explicit.

⸻

CellLineage Responsibilities

CellLineage remains a rule provider.

In Phase2-B it provides:
	•	Division axis rule (relative to cell orientation)
	•	Orientation update rule (how daughters reorient)

Critically:
	•	Rules are data-driven (YAML)
	•	No hard-coded biological cases

CellLineage answers:

“Given this cell at this step, how should division be oriented relative to itself?”

⸻

Division Timing Model

Division is asynchronous.

Each cell carries next_division_step:
	•	Set at birth
	•	Possibly inherited or overridden by lineage rules

At each global step:
	•	Only cells whose next_division_step == step divide
	•	Others persist unchanged

This enables:
	•	AB-first, P-later patterns
	•	Controlled replay of known embryonic sequences

⸻

YAML-Driven Lineage Definition

Lineage behavior is defined externally.

YAML specifies:
	•	Division orientation rules per step or per lineage state
	•	Orientation rotation (e.g. parallel, orthogonal, inherited)
	•	Division delay rules

This ensures:
	•	Reproducibility
	•	Inspectability
	•	Zero code edits for new hypotheses

⸻

Recording and Visualization

Recorder behavior is unchanged:
	•	Full snapshot per step
	•	All cells recorded

Visualization goals for Phase2-B:
	•	See orientation frames
	•	See asymmetric timing effects
	•	Verify geometric outcomes (e.g. T-shape, diamond)

Visualization remains diagnostic, not aesthetic.

⸻

Success Criteria

Phase2-B is complete when:
	•	Division timing differs between sibling lineages
	•	Orientation inheritance is visible and testable
	•	4-cell arrangements matching known embryos can be reproduced
	•	All behavior is explainable from rules + initial state

If it looks right but cannot be explained, Phase2-B has failed.

⸻

Transition to Phase2-C

Phase2-B establishes structured causality.

Phase2-C may introduce:
	•	Mechanical constraints
	•	Eggshell boundaries
	•	Limited relaxation dynamics

But only after orientation and timing are proven correct.