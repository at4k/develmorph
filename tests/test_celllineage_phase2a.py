# tests/test_celllineage_phase2a.py

import numpy as np
import pytest

from develmorph.celllineage import CellLineage
from develmorph.lineage_recorder import LineageRecorder


def test_celllineage_basic_division_axis():
    """
    Minimal sanity check for CellLineage.

    Ensures that decide_division_axis() always returns
    a normalized 3D vector.

    This test freezes the Phase2-A assumption:
    division direction is predefined and deterministic
    at early developmental stages.
    """
    lineage = CellLineage(lineage_id=0)

    axis = lineage.decide_division_axis(step=0)

    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)

    assert axis.shape == (3,)
    assert pytest.approx(norm, rel=1e-6) == 1.0


def test_lineage_recorder_records_time_series():
    """
    Verify that LineageRecorder correctly stores
    time-series cellular states.

    This test validates the integrity of developmental
    history logging prior to visualization.
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


def test_celllineage_and_recorder_integration():
    """
    Integration test for Phase2-A.

    Pipeline:
        CellLineage -> division axis -> positional update -> LineageRecorder

    Confirms that lineage decisions can be propagated
    into spatial trajectories and recorded without
    interface mismatch.
    """
    lineage = CellLineage(lineage_id=1)
    recorder = LineageRecorder()

    pos = np.zeros(3)

    for step in range(3):
        axis = lineage.decide_division_axis(step)
        pos = pos + axis  # temporary kinematic proxy

        recorder.record(
            step=step,
            cell_id=step,
            lineage_id=lineage.lineage_id,
            position=pos.copy(),
            divided=(step > 0),
        )

    df = recorder.to_dataframe()

    assert df["step"].tolist() == [0, 1, 2]
    assert np.all(np.isfinite(df[["x", "y", "z"]].values))
