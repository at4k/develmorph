import pandas as pd


class LineageRecorder:
    """
    Records developmental history of cells over time.

    Designed for Phase2-A to Phase2-C visualization
    and analysis.
    """

    def __init__(self):
        self._records = []

    def record(
        self,
        step: int,
        cell_id: int,
        lineage_id: int,
        position,
        divided: bool,
    ):
        """
        Record a single cellular state at a given time step.
        """
        x, y, z = position

        self._records.append(
            {
                "step": step,
                "cell_id": cell_id,
                "lineage_id": lineage_id,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "divided": bool(divided),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert recorded history to a pandas DataFrame.
        """
        return pd.DataFrame(self._records)
