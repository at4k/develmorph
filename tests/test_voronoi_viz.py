# tests/test_voronoi_viz.py
from __future__ import annotations
import numpy as np
from develmorph.physics_cell import create_physics_cell
from develmorph.power_diagram import compute_voronoi_cells
from develmorph.viz import VoronoiMesh, reconstruct_voronoi_meshes


def _make_2cell():
    positions = np.zeros((0, 3))
    c0, positions = create_physics_cell(np.array([1.0, 0.0, 0.0]), 0.5, positions)
    c1, positions = create_physics_cell(np.array([-1.0, 0.0, 0.0]), 0.5, positions)
    return [c0, c1], positions


def test_voronoi_mesh_points_returns_vertices():
    cells, positions = _make_2cell()
    shell = (5.0, 5.0, 5.0)
    voronoi = compute_voronoi_cells(cells, positions, shell, n_ghost=100)
    mesh = VoronoiMesh(voronoi[0])
    pts = mesh.mesh_points()
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    assert len(pts) >= 4  # at least a tetrahedron's worth of vertices


def test_reconstruct_voronoi_meshes_count():
    cells, positions = _make_2cell()
    shell = (5.0, 5.0, 5.0)
    voronoi = compute_voronoi_cells(cells, positions, shell, n_ghost=100)
    meshes = reconstruct_voronoi_meshes(voronoi)
    assert len(meshes) == 2


def test_reconstruct_voronoi_meshes_vertices_finite():
    cells, positions = _make_2cell()
    shell = (5.0, 5.0, 5.0)
    voronoi = compute_voronoi_cells(cells, positions, shell, n_ghost=100)
    meshes = reconstruct_voronoi_meshes(voronoi)
    for mesh in meshes:
        pts = mesh.mesh_points()
        assert np.all(np.isfinite(pts)), "Non-finite vertices in VoronoiMesh"
