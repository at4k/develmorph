# scripts/demo_lineage_division_3d.py
"""
Phase2-A demonstration script.

Simulates a minimal lineage-driven cell division process,
records the developmental history, and visualizes it in 3D.

This demo intentionally bypasses full physics/solver layers.
The goal is to verify that:
- CellLineage controls division direction
- LineageRecorder captures time-series structure
- viz.py can faithfully render developmental trajectories
"""

import numpy as np

from develmorph.celllineage import CellLineage
from develmorph.lineage_recorder import LineageRecorder
from develmorph.viz import visualize_lineage_3d


class SimpleCell:
    """
    Minimal cell proxy used only for Phase2-A demonstration.
    """

    def __init__(self, cell_id: int, lineage: CellLineage, position):
        self.cell_id = cell_id
        self.lineage = lineage
        self.position = np.asarray(position, dtype=float)

    def divide(self, step: int, factory):
        """
        Perform a deterministic division along the lineage axis.

        Returns two daughter SimpleCell instances.
        """
        axis = self.lineage.decide_division_axis(step)
        offset = 0.5 * axis

        c1 = factory.create(self.lineage, self.position + offset)
        c2 = factory.create(self.lineage, self.position - offset)
        print(f"step={step}: {c1},{c2}")
        return c1, c2


class CellFactory:
    def __init__(self):
        self.next_id = 0

    def create(self, lineage, position):
        cid = self.next_id
        self.next_id += 1
        return SimpleCell(cid, lineage, position)


def alternating_division_rule(step: int, polarity: np.ndarray):
    """
    Step 1: along polarity (x-axis)
    Step 2: orthogonal direction (y-axis)
    """
    if step == 1:
        return polarity
    elif step == 2:
        return np.array([0.0, 1.0, 0.0])
    else:
        return polarity


def run_demo():
    lineage = CellLineage(
        lineage_id=0,
        polarity=np.array([1.0, 0.0, 0.0]),
        division_axis_rule=alternating_division_rule,
    )
    recorder = LineageRecorder()

    # initial cell
    factory = CellFactory()
    cells = [factory.create(lineage, [0, 0, 0])]

    n_steps = 3

    for step in range(n_steps):

        next_cells = []
        for cell in cells:
            cell.position += 0.1 * lineage.decide_division_axis(step)

            recorder.record(
                step=step,
                cell_id=cell.cell_id,
                lineage_id=lineage.lineage_id,
                position=cell.position,
                divided=(step > 0),
            )

            # divide after step 0
            if step >= 1:
                daughters = cell.divide(step, factory)
                next_cells.extend(daughters)
            else:
                next_cells.append(cell)

        cells = next_cells

    for cell in cells:
        recorder.record(
            step=n_steps,
            cell_id=cell.cell_id,
            lineage_id=lineage.lineage_id,
            position=cell.position,
            divided=False,
        )
    print(cells)

    df = recorder.to_dataframe()
    print(df)

    visualize_lineage_3d(df)


if __name__ == "__main__":
    run_demo()
