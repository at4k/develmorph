# tests/test_power_diagram.py
from __future__ import annotations
import numpy as np
from develmorph.power_diagram import make_ghost_points, VoronoiCell


def test_make_ghost_points_shape():
    pts = make_ghost_points(1.84, 1.26, 1.04, n=50)
    assert pts.shape == (50, 3)


def test_make_ghost_points_on_ellipsoid_surface():
    a, b, c = 1.84, 1.26, 1.04
    margin = 0.05
    pts = make_ghost_points(a, b, c, n=50, margin=margin)
    # Each point should be on the surface of the expanded ellipsoid
    ellipsoid_val = (
        (pts[:, 0] / (a + margin)) ** 2
        + (pts[:, 1] / (b + margin)) ** 2
        + (pts[:, 2] / (c + margin)) ** 2
    )
    np.testing.assert_allclose(ellipsoid_val, 1.0, atol=1e-6)


def test_make_ghost_points_cover_all_axes():
    """Points should cover +/-x, +/-y, +/-z directions."""
    pts = make_ghost_points(1.0, 1.0, 1.0, n=100)
    # At least one point in each of the 6 principal directions
    assert pts[:, 0].max() > 0.9, "Missing +x coverage"
    assert pts[:, 0].min() < -0.9, "Missing -x coverage"
    assert pts[:, 1].max() > 0.9, "Missing +y coverage"
    assert pts[:, 1].min() < -0.9, "Missing -y coverage"
    assert pts[:, 2].max() > 0.9, "Missing +z coverage"
    assert pts[:, 2].min() < -0.9, "Missing -z coverage"


def test_voronoi_cell_volume_gradient_single_face():
    """volume_gradient = 0.5 * sum(area * normal) for all faces."""
    normal = np.array([1.0, 0.0, 0.0])
    vc = VoronoiCell(
        index=0,
        vertices=np.zeros((1, 3)),
        volume=0.0,
        face_areas=[2.0],
        face_normals=[normal],
        face_neighbors=[None],
        contact_areas={},
        contact_normals={},
    )
    grad = vc.volume_gradient()
    np.testing.assert_allclose(grad, np.array([1.0, 0.0, 0.0]))


def test_voronoi_cell_volume_gradient_empty():
    """volume_gradient returns zero vector when no faces."""
    vc = VoronoiCell(
        index=0, vertices=np.zeros((0, 3)), volume=0.0,
        face_areas=[], face_normals=[], face_neighbors=[],
        contact_areas={}, contact_normals={},
    )
    np.testing.assert_array_equal(vc.volume_gradient(), np.zeros(3))
