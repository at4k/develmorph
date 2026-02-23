import math

import numpy as np

from develmorph.cell_body import (
    CellBody,
    build_particle_arrays,
    compute_edges,
    create_cell_body,
)


def test_create_cell_body_particle_count():
    """CellBody has 1 center + 6 surface = 7 particles."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(cb.surface_indices) == 6
    assert len(positions) == 7


def test_create_cell_body_octahedron_positions():
    """Surface points form an octahedron at the given radius."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([1.0, 2.0, 3.0]),
        radius=0.5,
        positions=positions,
    )
    center = positions[cb.center_idx]
    for si in cb.surface_indices:
        dist = np.linalg.norm(positions[si] - center)
        assert abs(dist - 0.5) < 1e-10


def test_create_cell_body_triangle_count():
    """Octahedron has 8 triangular faces."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(cb.triangles) == 8


def test_create_cell_body_target_volume():
    """Target volume equals octahedron mesh volume (4/3 * r^3)."""
    positions = np.zeros((0, 3))
    r = 1.5
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=r,
        positions=positions,
    )
    # Regular octahedron with vertices at distance r: V = (4/3) * r^3
    expected = (4.0 / 3.0) * r**3
    assert abs(cb.target_volume - expected) < 1e-10


def test_create_multiple_cells_indices():
    """Multiple cells get non-overlapping indices."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    cb2, positions = create_cell_body(
        center=np.array([3.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    all_indices_1 = {cb1.center_idx} | set(cb1.surface_indices)
    all_indices_2 = {cb2.center_idx} | set(cb2.surface_indices)
    assert len(all_indices_1 & all_indices_2) == 0
    assert len(positions) == 14


def test_build_particle_arrays():
    """build_particle_arrays reconstructs flat arrays and reassigns indices."""
    positions = np.zeros((0, 3))
    cb1, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    cb2, positions = create_cell_body(
        center=np.array([3.0, 0.0, 0.0]),
        radius=0.5,
        positions=positions,
    )
    cell_bodies = [cb1, cb2]
    new_positions, new_inv_masses = build_particle_arrays(cell_bodies, positions)
    assert len(new_positions) == 14
    assert len(new_inv_masses) == 14
    assert np.allclose(new_positions[cb1.center_idx], [0.0, 0.0, 0.0])
    assert np.allclose(new_positions[cb2.center_idx], [3.0, 0.0, 0.0])


def test_cell_body_edge_pairs():
    """CellBody has 6 center-surface edges and 12 surface-surface edges."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(cb.center_surface_edges) == 6
    assert len(cb.surface_surface_edges) == 12


def test_compute_edges_octahedron():
    """compute_edges extracts 12 unique edges from 8 octahedron triangles."""
    from develmorph.cell_body import OCTAHEDRON_TRIANGLES

    edges = compute_edges(OCTAHEDRON_TRIANGLES)
    assert len(edges) == 12
    # All edges should be sorted tuples (i < j)
    for a, b in edges:
        assert a < b


def test_compute_edges_single_triangle():
    """Single triangle produces 3 edges."""
    edges = compute_edges([(0, 1, 2)])
    assert len(edges) == 3
    assert set(edges) == {(0, 1), (0, 2), (1, 2)}
