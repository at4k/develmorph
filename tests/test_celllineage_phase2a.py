# tests/test_celllineage_phase2a.py

import numpy as np
import pytest

from develmorph.celllineage import CellLineage, DivisionRule
from develmorph.lineage_recorder import LineageRecorder
from develmorph.cell import Cell


YAML_PATH = "develmorph/lineages/celegans_early.yaml"


def test_celllineage_loads_yaml():
    """
    Verify CellLineage correctly loads YAML configuration.
    """
    lineage = CellLineage(YAML_PATH)

    assert lineage.lineage_id == "celegans_early"
    assert lineage.get_step_program(1) is not None
    assert lineage.get_step_program(999) is None


def test_celllineage_get_division_for_cell():
    """
    Verify division rules are retrieved correctly by cell name.
    """
    lineage = CellLineage(YAML_PATH)

    # Step 1: P0 divides
    rule = lineage.get_division_for_cell(step=1, cell_name="P0")
    assert rule is not None
    assert rule.parent == "P0"
    assert rule.axis == "major_axis"
    assert rule.daughters == ["AB", "P1"]

    # Step 2: AB divides
    rule = lineage.get_division_for_cell(step=2, cell_name="AB")
    assert rule is not None
    assert rule.axis == "orthogonal_axis"

    # Step 2: P1 does not divide
    rule = lineage.get_division_for_cell(step=2, cell_name="P1")
    assert rule is None


def test_celllineage_resolve_division_axis():
    """
    Verify division axis is correctly resolved from orientation.
    """
    lineage = CellLineage(YAML_PATH)

    # Identity orientation
    orientation = np.eye(3)

    # major_axis -> e1 = [1, 0, 0]
    rule = DivisionRule(parent="P0", axis="major_axis", daughters=["A", "B"])
    axis = lineage.resolve_division_axis(rule, orientation)
    assert axis.shape == (3,)
    np.testing.assert_array_almost_equal(axis, [1, 0, 0])

    # minor_axis -> e2 = [0, 1, 0]
    rule = DivisionRule(parent="P0", axis="minor_axis", daughters=["A", "B"])
    axis = lineage.resolve_division_axis(rule, orientation)
    np.testing.assert_array_almost_equal(axis, [0, 1, 0])

    # orthogonal_axis -> e3 = [0, 0, 1]
    rule = DivisionRule(parent="P0", axis="orthogonal_axis", daughters=["A", "B"])
    axis = lineage.resolve_division_axis(rule, orientation)
    np.testing.assert_array_almost_equal(axis, [0, 0, 1])


def test_celllineage_resolve_axis_with_rotated_orientation():
    """
    Verify axis resolution works with non-identity orientation.
    """
    lineage = CellLineage(YAML_PATH)

    # Rotate 90 degrees around z-axis: e1 points to y, e2 points to -x
    orientation = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1],
    ], dtype=float)

    rule = DivisionRule(parent="P0", axis="major_axis", daughters=["A", "B"])
    axis = lineage.resolve_division_axis(rule, orientation)
    np.testing.assert_array_almost_equal(axis, [0, 1, 0])


def test_lineage_recorder_records_time_series():
    """
    Verify that LineageRecorder correctly stores
    time-series cellular states.
    """
    recorder = LineageRecorder()

    recorder.record(
        step=0,
        cell_id=0,
        lineage_id=0,
        position=np.array([0.0, 0.0, 0.0]),
        divided=False,
    )

    recorder.record(
        step=1,
        cell_id=1,
        lineage_id=0,
        position=np.array([1.0, 0.0, 0.0]),
        divided=True,
    )

    data = recorder.to_dataframe()

    assert len(data) == 2
    assert set(data.columns) >= {
        "step",
        "cell_id",
        "lineage_id",
        "x",
        "y",
        "z",
        "divided",
    }


def test_cell_orientation_inheritance():
    """
    Verify Cell correctly stores and inherits orientation.
    """
    parent = Cell(cell_id=0, name="P0", position=[0, 0, 0])

    # Default orientation is identity
    np.testing.assert_array_equal(parent.orientation, np.eye(3))

    # Create daughter with inherited orientation
    daughter = Cell(
        cell_id=1,
        name="AB",
        position=[0.5, 0, 0],
        orientation=parent.orientation.copy(),
    )

    np.testing.assert_array_equal(daughter.orientation, parent.orientation)


def test_full_division_simulation_with_recorder():
    """
    Integration test: run full 4-cell simulation with LineageRecorder.

    Pipeline:
        CellLineage -> division axis -> Cell creation -> LineageRecorder
    """
    lineage = CellLineage(YAML_PATH)
    recorder = LineageRecorder()

    # Initial cell
    cells = {"P0": Cell(cell_id=0, name="P0", position=[0, 0, 0])}
    next_id = 1

    # Record initial state
    for cell in cells.values():
        recorder.record(
            step=0,
            cell_id=cell.cell_id,
            lineage_id=0,
            position=cell.position,
            divided=False,
        )

    step = 1
    while True:
        program = lineage.get_step_program(step)
        if program is None:
            break

        divisions = [r for r in program.divisions if r.parent in cells]
        if not divisions:
            break

        for rule in divisions:
            parent = cells[rule.parent]
            axis = lineage.resolve_division_axis(rule, parent.orientation)

            # Create daughters
            a = Cell(
                cell_id=next_id,
                name=rule.daughters[0],
                position=parent.position + 0.5 * axis,
                orientation=parent.orientation.copy(),
            )
            b = Cell(
                cell_id=next_id + 1,
                name=rule.daughters[1],
                position=parent.position - 0.5 * axis,
                orientation=parent.orientation.copy(),
            )
            next_id += 2

            del cells[rule.parent]
            cells[a.name] = a
            cells[b.name] = b

        # Record state after divisions
        for cell in cells.values():
            recorder.record(
                step=step,
                cell_id=cell.cell_id,
                lineage_id=0,
                position=cell.position,
                divided=True,
            )

        step += 1

    # Verify final cell count
    assert len(cells) == 4
    assert set(cells.keys()) == {"ABa", "ABp", "EMS", "P2"}

    # Verify recorder captured history
    df = recorder.to_dataframe()
    assert len(df) > 0
    assert set(df.columns) >= {"step", "cell_id", "x", "y", "z", "divided"}

    # Check steps are recorded
    steps_recorded = sorted(df["step"].unique())
    assert steps_recorded == [0, 1, 2, 3]
