# Design: 4-Cell Stage Reproduction via Lineage-Driven Division

Date: 2026-05-04
Branch: feat-xpbd-solver

## Goal

Reproduce C. elegans 4-cell stage (ABa, ABp, EMS, P2) using the XPBD physical
simulation. Division axis and volume ratio must match `celegans_early.yaml`.

## Success Criteria

- XPBD simulation runs with lineage-driven division for steps 1-3 (P0->2-cell->3-cell->4-cell)
- Each CellBody knows its biological name and orientation frame
- Division axis (major_axis / orthogonal_axis) resolves correctly per DivisionRule
- Cells physically settle into a recognizable 4-cell configuration
- All existing tests continue to pass (backward compat)
- New tests cover: name assignment, orientation inheritance, axis resolution, volume ratio

## Architecture

### Component Changes

#### `cell_body.py` - CellBody

Add two fields:

- `name: str | None = None` - biological name (e.g. "P0", "AB", "P1")
- `orientation: np.ndarray` - 3x3 orthonormal matrix [e1, e2, e3], default `np.eye(3)`

`create_cell_body` gains optional `name` and `orientation` parameters.

#### `division.py` - divide_cell_body

Add `daughter_names` and `daughter_orientations` parameters (both optional):

```python
def divide_cell_body(
    parent, positions,
    axis=None, volume_ratio=0.5,
    area_threshold=None,
    daughter_names=(None, None),
    daughter_orientations=(None, None),
)
```

Daughters receive their name and orientation from these parameters.
If not provided, daughters inherit parent name=None and orientation=parent.orientation.

#### `engine.py` - World

`World.__init__` gains an optional `lineage: CellLineage | None = None` parameter.

`World.step()` division logic becomes:

```
if self.lineage is not None:
    program = self.lineage.get_step_program(self.step_count)
    for rule in program.divisions:
        find CellBody by name == rule.parent
        axis = self.lineage.resolve_division_axis(rule, parent.orientation)
        daughter_names = (rule.daughters[0], rule.daughters[1])
        daughter_orientations = (parent.orientation.copy(), parent.orientation.copy())
        divide with axis, rule.volume_ratio, daughter_names, daughter_orientations
else:
    # existing behavior: divide largest cell
```

### Data Flow

```
celegans_early.yaml
    -> CellLineage (loaded at World init)
    -> World.step() queries get_step_program(step_count)
    -> DivisionRule: parent name, axis type, daughters, volume_ratio
    -> find CellBody by .name
    -> resolve_division_axis(rule, parent.orientation) -> 3D vector
    -> divide_cell_body(axis, volume_ratio, daughter_names, daughter_orientations)
    -> daughters appended to self.cell_bodies with name + orientation set
```

### Orientation Management

- Initial P0: `orientation = np.eye(3)`, so e1=x, e2=y, e3=z
- A-P axis = e1 (x-axis) by convention, matching `major_axis` -> column 0
- All daughters inherit parent orientation unchanged in this phase
- `CellLineage.AXIS_MAP` already maps major_axis->0, minor_axis->1, orthogonal_axis->2

### Backward Compatibility

- `CellBody.name` defaults to `None`; existing code that creates CellBody without names is unaffected
- `World(lineage=None)` falls back to current largest-cell-first logic
- `divide_cell_body` without daughter_names/orientations behaves as before

## Testing Plan

Tests written before implementation (TDD):

1. `test_cell_body_has_name_and_orientation` - CellBody stores name and orientation
2. `test_create_cell_body_with_name` - create_cell_body passes name through
3. `test_divide_cell_body_daughter_names` - daughters get correct names
4. `test_divide_cell_body_inherits_orientation` - daughters inherit parent orientation
5. `test_world_lineage_step1_p0_divides` - step 1: P0 -> AB + P1 with correct axis/volume
6. `test_world_lineage_step2_ab_divides` - step 2: AB -> ABa + ABp (orthogonal axis)
7. `test_world_lineage_step3_p1_divides` - step 3: P1 -> EMS + P2
8. `test_world_lineage_4cell_stage_physics` - full run, 4 cell bodies settled

## Files Modified

| File | Change |
|------|--------|
| `develmorph/cell_body.py` | Add name, orientation to CellBody and create_cell_body |
| `develmorph/division.py` | Add daughter_names, daughter_orientations params |
| `develmorph/engine.py` | Add lineage param, lineage-driven step() logic |
| `tests/test_cell_body.py` | New tests for name/orientation fields |
| `tests/test_division.py` | New tests for daughter name/orientation |
| `tests/test_engine.py` | New file for World+CellLineage integration tests |
| `scripts/demo_4cell_stage.py` | New demo script for visual confirmation |
