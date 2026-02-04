from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


@dataclass
class DivisionRule:
    """A single division instruction."""

    parent: str
    axis: str  # major_axis, minor_axis, orthogonal_axis
    daughters: List[str]
    volume_ratio: float = 0.5  # Fraction of volume going to first daughter (0.0-1.0)


@dataclass
class StepProgram:
    """All events happening at a given step."""

    step: int
    divisions: List[DivisionRule]
    fate: List[Dict[str, Any]]  # For future expansion


class CellLineage:
    """
    Lineage-defined division program loaded from YAML.

    Provides division rules based on step number and cell name.
    Does not execute division; only provides the rules.
    """

    AXIS_MAP = {
        "major_axis": 0,  # e1
        "minor_axis": 1,  # e2
        "orthogonal_axis": 2,  # e3
    }

    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        self.lineage_id = data.get("lineage_id", "unknown")
        self._steps: Dict[int, StepProgram] = {}
        self._parse_steps(data.get("steps", []))

    def _parse_steps(self, steps_data: list):
        for step_data in steps_data:
            step_num = step_data["step"]

            divisions = []
            for div in step_data.get("divisions", []):
                divisions.append(
                    DivisionRule(
                        parent=div["parent"],
                        axis=div["axis"],
                        daughters=div["daughters"],
                        volume_ratio=div.get("volume_ratio", 0.5),
                    )
                )

            fate = step_data.get("fate", [])

            self._steps[step_num] = StepProgram(
                step=step_num,
                divisions=divisions,
                fate=fate,
            )

    def get_step_program(self, step: int) -> Optional[StepProgram]:
        """Get the full program for a step."""
        return self._steps.get(step)

    def get_division_for_cell(
        self, step: int, cell_name: str
    ) -> Optional[DivisionRule]:
        """Get division rule for a specific cell at a step."""
        program = self._steps.get(step)
        if program is None:
            return None

        for div in program.divisions:
            if div.parent == cell_name:
                return div
        return None

    def resolve_division_axis(
        self, rule: DivisionRule, parent_orientation: np.ndarray
    ) -> np.ndarray:
        """
        Resolve division axis from rule and parent orientation.

        Args:
            rule: DivisionRule specifying axis type
            parent_orientation: 3x3 orthonormal matrix [e1, e2, e3]

        Returns:
            3D unit vector for division axis
        """
        axis_idx = self.AXIS_MAP[rule.axis]
        axis = parent_orientation[:, axis_idx]
        return axis / np.linalg.norm(axis)
