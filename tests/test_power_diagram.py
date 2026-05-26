# tests/test_power_diagram.py
from __future__ import annotations
import numpy as np
import pytest
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


def test_voronoi_cell_is_dataclass():
    import numpy as np
    vc = VoronoiCell(
        index=0,
        vertices=np.zeros((4, 3)),
        volume=1.0,
        face_areas=[1.0, 1.0, 1.0, 1.0],
        face_normals=[np.array([1.,0.,0.])] * 4,
        face_neighbors=[1, None, None, 2],
        contact_areas={1: 1.0, 2: 1.0},
        contact_normals={1: np.array([1.,0.,0.]), 2: np.array([0.,1.,0.])},
    )
    assert vc.index == 0
    assert vc.volume == 1.0
    assert len(vc.face_areas) == 4
