# Phase2-A: CellLineage implementation
# Minimal, deterministic lineage-controlled division axis

from __future__ import annotations
import numpy as np
from typing import Callable


def default_division_axis_rule(step: int, polarity: np.ndarray) -> np.ndarray:
    """
    Default rule: always divide along polarity axis.
    """
    return polarity


class CellLineage:
    """
    Represents a developmental lineage that prescribes
    division behavior independently of physical simulation.
    Controls lineage-level developmental rules such as division orientation.

    Phase2-A assumption:
    Division axes are predefined and deterministic.
    """

    def __init__(
        self,
        lineage_id: int,
        polarity: np.ndarray = np.array([1.0, 0.0, 0.0]),
        division_axis_rule: Callable[[int, np.ndarray], np.ndarray] | None = None,
    ):
        self.lineage_id = lineage_id
        self.polarity = polarity / np.linalg.norm(polarity)

        self.division_axis_rule = division_axis_rule or default_division_axis_rule

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
        axis = self.division_axis_rule(step, self.polarity)

        # Normalize defensively
        axis = axis / np.linalg.norm(axis)
        return axis
