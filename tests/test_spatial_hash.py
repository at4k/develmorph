# tests/test_spatial_hash.py
import numpy as np
from develmorph.physics_cell import create_physics_cell
from develmorph.spatial_hash import SpatialHash


def test_spatial_hash_finds_overlapping_pair():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)
    c3, positions = create_physics_cell(np.array([10.0, 0.0, 0.0]), 1.0, positions)

    sh = SpatialHash(cell_size=2.5)
    sh.build([c1, c2, c3], positions)
    pairs = sh.candidate_pairs()

    assert (0, 1) in pairs or (1, 0) in pairs
    assert (0, 2) not in pairs and (2, 0) not in pairs


def test_spatial_hash_empty_cells():
    sh = SpatialHash(cell_size=2.0)
    sh.build([], np.zeros((0, 3)))
    assert sh.candidate_pairs() == set()


def test_spatial_hash_single_cell():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    sh = SpatialHash(cell_size=2.0)
    sh.build([c1], positions)
    assert sh.candidate_pairs() == set()


def test_spatial_hash_finds_all_overlapping_pairs():
    """Chain of 4 cells; each overlaps only its neighbor."""
    positions = np.zeros((0, 3))
    cells = []
    for i in range(4):
        c, positions = create_physics_cell(np.array([float(i) * 1.8, 0.0, 0.0]),
                                           1.0, positions)
        cells.append(c)

    sh = SpatialHash(cell_size=1.5)
    sh.build(cells, positions)
    pairs = sh.candidate_pairs()

    # Neighbors overlap (dist=1.8 < r1+r2=2.0)
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (2, 3) in pairs
    # Non-neighbors do not (dist=3.6, in non-adjacent voxels with cell_size=1.5)
    assert (0, 2) not in pairs
