# Phase2-A: CellLineage implementation
# Minimal, deterministic lineage-controlled division axis

from __future__ import annotations
import numpy as np


import numpy as np


class CellLineage:
    """
    Represents a developmental lineage that prescribes
    division behavior independently of physical simulation.

    Phase2-A assumption:
    Division axes are predefined and deterministic.
    """

    def __init__(self, lineage_id: int):
        self.lineage_id = lineage_id

    def decide_division_axis(self, step: int) -> np.ndarray:
        """
        Decide the division axis for a given developmental step.

        Parameters
        ----------
        step : int
            Discrete developmental time step.

        Returns
        -------
        np.ndarray, shape (3,)
            Normalized 3D division axis.
        """
        # Temporary deterministic rule for Phase2-A
        axis = np.array([1.0, 0.0, 0.0])

        # Normalize defensively
        axis = axis / np.linalg.norm(axis)
        return axis


# --- Example integration sketch (engine-side) ---


class DivisionEngine:
    def __init__(self, lineage: CellLineage):
        self.lineage = lineage

    def divide_cell(self, cell, world):
        if not self.lineage.should_divide(cell, world):
            return None

        axis = self.lineage.decide_division_axis(cell, world)
        eps = self.lineage.eps

        pos_a = cell.pos + eps * axis
        pos_b = cell.pos - eps * axis

        # create daughter cells (details depend on your existing Cell class)
        cell_a = cell.clone(pos=pos_a)
        cell_b = cell.clone(pos=pos_b)

        return cell_a, cell_b
