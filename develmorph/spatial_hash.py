# develmorph/spatial_hash.py
from __future__ import annotations

from collections import defaultdict

import numpy as np


class SpatialHash:
    """
    Grid-based spatial hash for O(N) neighbor search.

    Assigns each cell center to a grid voxel of size `cell_size`.
    Candidate collision pairs are cells in the same or adjacent voxels.
    Set `cell_size` >= max(r_i + r_j) across all pairs so that every
    overlapping pair is guaranteed to appear as a candidate.
    """

    def __init__(self, cell_size: float) -> None:
        self.cell_size = float(cell_size)
        self._grid: dict = defaultdict(list)
        self._physics_cells: list = []

    def build(self, physics_cells: list, positions: np.ndarray) -> None:
        """Populate the hash grid from current positions."""
        self._grid.clear()
        self._physics_cells = physics_cells

        for idx, pc in enumerate(physics_cells):
            pos = positions[pc.center_idx]
            voxel = self._voxel(pos)
            self._grid[voxel].append(idx)

    def candidate_pairs(self) -> set:
        """
        Return set of (i, j) pairs with i < j that are in adjacent voxels.

        Checks all 27 neighboring voxels for each occupied voxel.
        Only forward neighbors are checked to avoid duplicate pairs.
        """
        pairs: set = set()

        for voxel, indices in self._grid.items():
            # Self-pairs within same voxel
            for a in range(len(indices)):
                for b in range(a + 1, len(indices)):
                    i, j = indices[a], indices[b]
                    pairs.add((min(i, j), max(i, j)))

            # Pairs with neighboring voxels (forward only to avoid duplicates)
            vx, vy, vz = voxel
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = (vx + dx, vy + dy, vz + dz)
                        if neighbor <= voxel:
                            continue
                        for i in indices:
                            for j in self._grid.get(neighbor, []):
                                pairs.add((min(i, j), max(i, j)))

        return pairs

    def _voxel(self, pos: np.ndarray) -> tuple:
        """Map a 3D position to integer voxel coordinates."""
        return (
            int(np.floor(pos[0] / self.cell_size)),
            int(np.floor(pos[1] / self.cell_size)),
            int(np.floor(pos[2] / self.cell_size)),
        )
