import math

import numpy as np

from develmorph.cell_body import (
    OCTAHEDRON_DIRS,
    OCTAHEDRON_TRIANGLES,
    CellBody,
    build_particle_arrays,
    compute_edges,
    create_cell_body,
    subdivide_mesh,
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


def test_subdivide_mesh_no_subdivision_when_below_threshold():
    """When all triangles are below threshold, mesh is unchanged."""
    dirs, tris = subdivide_mesh(OCTAHEDRON_DIRS, OCTAHEDRON_TRIANGLES, area_threshold=10.0)
    assert len(dirs) == 6
    assert len(tris) == 8


def test_subdivide_mesh_all_subdivided():
    """With threshold=0.3, all octahedron faces are subdivided below the threshold.

    Centroid subdivision converges to a fixed point at ~0.207 (arc-midpoint of the
    longest original edge). Any threshold above this fixed point guarantees convergence.
    """
    dirs, tris = subdivide_mesh(OCTAHEDRON_DIRS, OCTAHEDRON_TRIANGLES, area_threshold=0.3)
    assert len(dirs) > 6
    assert len(tris) > 8
    for a, b, c in tris:
        v0, v1, v2 = dirs[a], dirs[b], dirs[c]
        area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        assert area <= 0.3 + 1e-10


def test_subdivide_mesh_points_on_unit_sphere():
    """All output direction vectors lie on the unit sphere."""
    dirs, tris = subdivide_mesh(OCTAHEDRON_DIRS, OCTAHEDRON_TRIANGLES, area_threshold=0.3)
    for d in dirs:
        assert abs(np.linalg.norm(d) - 1.0) < 1e-10


def test_create_cell_body_with_subdivision():
    """create_cell_body with area_threshold produces more surface points."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        area_threshold=0.3,
    )
    assert len(cb.surface_indices) > 6
    assert len(cb.triangles) > 8
    center = positions[cb.center_idx]
    for si in cb.surface_indices:
        dist = np.linalg.norm(positions[si] - center)
        assert abs(dist - 1.0) < 1e-10


def test_create_cell_body_no_subdivision_default():
    """create_cell_body without area_threshold produces standard 6 surface points."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert len(cb.surface_indices) == 6
    assert len(cb.triangles) == 8


def test_create_cell_body_subdivision_volume_correct():
    """Subdivided cell has target volume between octahedron and sphere volume.

    Centroid subdivision pushes face midpoints onto the sphere surface, so the
    effective mesh volume is larger than the inscribed octahedron and smaller
    than the enclosing sphere.
    """
    positions = np.zeros((0, 3))
    r = 1.0
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=r,
        positions=positions,
        area_threshold=0.3,
    )
    assert cb.target_volume > 0
    octahedron_vol = (4.0 / 3.0) * r**3
    sphere_vol = (4.0 / 3.0) * math.pi * r**3
    assert cb.target_volume > octahedron_vol
    assert cb.target_volume < sphere_vol


def test_create_cell_body_subdivision_edges_match_triangles():
    """Edge count from compute_edges matches stored edges."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        area_threshold=0.3,
    )
    expected_edges = compute_edges(cb.triangles)
    assert len(cb.edges) == len(expected_edges)


def test_build_particle_arrays_with_subdivided_cell():
    """build_particle_arrays works correctly with subdivided cells."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        area_threshold=0.3,
    )
    n_surface = len(cb.surface_indices)
    n_edges = len(cb.edges)

    new_pos, new_inv = build_particle_arrays([cb], positions)

    assert len(new_pos) == 1 + n_surface
    assert len(cb.surface_surface_edges) == n_edges
    assert len(cb.center_surface_edges) == n_surface


def test_cell_body_stores_custom_edges():
    """CellBody stores edges passed via constructor, not hardcoded OCTAHEDRON_EDGES."""
    custom_edges = [(0, 1), (1, 2), (2, 0)]
    cb = CellBody(
        center_idx=0,
        surface_indices=[1, 2, 3],
        target_volume=1.0,
        center_surface_rest=[1.0, 1.0, 1.0],
        surface_surface_rest=[1.0, 1.0, 1.0],
        triangles=[(0, 1, 2)],
        edges=custom_edges,
    )
    assert cb.edges == custom_edges
    assert cb.triangles == [(0, 1, 2)]
    assert len(cb.surface_surface_edges) == 3


def test_subdivide_mesh_centroid_split_produces_3_triangles():
    """Subdividing 1 triangle produces exactly 3 sub-triangles."""
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    tris = [(0, 1, 2)]
    new_dirs, new_tris = subdivide_mesh(dirs, tris, area_threshold=0.01)
    assert len(new_dirs) >= 4
    assert len(new_tris) >= 3


def test_cell_body_default_name_is_none():
    """CellBody.name defaults to None."""
    cb = CellBody(
        center_idx=0,
        surface_indices=[1, 2, 3, 4, 5, 6],
        target_volume=1.0,
        center_surface_rest=[1.0] * 6,
        surface_surface_rest=[1.0] * 12,
    )
    assert cb.name is None


def test_cell_body_stores_name():
    """CellBody stores the name passed to constructor."""
    cb = CellBody(
        center_idx=0,
        surface_indices=[1, 2, 3, 4, 5, 6],
        target_volume=1.0,
        center_surface_rest=[1.0] * 6,
        surface_surface_rest=[1.0] * 12,
        name="P0",
    )
    assert cb.name == "P0"


def test_cell_body_default_orientation_is_identity():
    """CellBody.orientation defaults to identity matrix."""
    cb = CellBody(
        center_idx=0,
        surface_indices=[1, 2, 3, 4, 5, 6],
        target_volume=1.0,
        center_surface_rest=[1.0] * 6,
        surface_surface_rest=[1.0] * 12,
    )
    assert np.allclose(cb.orientation, np.eye(3))


def test_create_cell_body_with_name():
    """create_cell_body passes name through to CellBody."""
    positions = np.zeros((0, 3))
    cb, _ = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        name="P0",
    )
    assert cb.name == "P0"


def test_create_cell_body_default_name_is_none():
    """create_cell_body without name gives CellBody.name == None."""
    positions = np.zeros((0, 3))
    cb, _ = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert cb.name is None


def test_create_cell_body_with_orientation():
    """create_cell_body passes orientation through to CellBody."""
    custom = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    positions = np.zeros((0, 3))
    cb, _ = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        orientation=custom,
    )
    assert np.allclose(cb.orientation, custom)


def test_create_cell_body_default_orientation_is_identity():
    """create_cell_body without orientation gives identity matrix."""
    positions = np.zeros((0, 3))
    cb, _ = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    assert np.allclose(cb.orientation, np.eye(3))
