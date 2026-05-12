import math

import numpy as np

from develmorph.cell_body import create_cell_body
from develmorph.project import (
    AdhesionConstraint,
    CollisionConstraint,
    DistanceConstraint,
    ShellConstraint,
    TriangleAreaConstraint,
    VolumeConstraint,
)


def test_collision_resolves_overlap():
    """Two overlapping particles are pushed apart."""
    x = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    radii = np.array([0.5, 0.5])
    dt = 0.01

    c = CollisionConstraint(i=0, j=1, rest_length=radii[0] + radii[1], compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    dist = np.linalg.norm(x[0] - x[1])
    assert dist >= (radii[0] + radii[1]) * 0.99


def test_collision_no_correction_when_separated():
    """No correction when particles are already separated."""
    x = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    assert np.allclose(x, x_before)


def test_collision_symmetric_correction():
    """Equal-mass particles receive equal and opposite corrections."""
    x = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c.project(x, inv_masses, dt, 0.0)

    # Midpoint should remain at original midpoint
    midpoint = (x[0] + x[1]) / 2
    assert np.allclose(midpoint, [0.25, 0.0, 0.0])


def test_collision_respects_mass_ratio():
    """Heavier particle moves less."""
    x = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    inv_masses = np.array([0.5, 1.0])  # particle 0 is heavier
    dt = 0.01

    x_before = x.copy()
    c = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c.project(x, inv_masses, dt, 0.0)

    move_0 = np.linalg.norm(x[0] - x_before[0])
    move_1 = np.linalg.norm(x[1] - x_before[1])
    assert move_1 > move_0  # lighter moves more


def test_collision_compliance_softens():
    """Higher compliance produces smaller correction per iteration."""
    x_hard = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    x_soft = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c_hard = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c_soft = CollisionConstraint(i=0, j=1, rest_length=1.0, compliance=1.0)

    c_hard.project(x_hard, inv_masses, dt, 0.0)
    c_soft.project(x_soft, inv_masses, dt, 0.0)

    dist_hard = np.linalg.norm(x_hard[0] - x_hard[1])
    dist_soft = np.linalg.norm(x_soft[0] - x_soft[1])

    # Hard constraint corrects more in one step
    assert dist_hard > dist_soft


def test_shell_pushes_inside():
    """Particle outside ellipsoid is pushed back inside."""
    x = np.array([[10.0, 0.0, 0.0]])
    inv_masses = np.array([1.0])
    dt = 0.01

    c = ShellConstraint(i=0, a=5.0, b=5.0, c=5.0, radius=0.5, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    # Effective shell: a - radius = 4.5
    val = (x[0, 0] / 4.5) ** 2 + (x[0, 1] / 4.5) ** 2 + (x[0, 2] / 4.5) ** 2
    assert val <= 1.01  # inside with small tolerance


def test_shell_no_correction_when_inside():
    """No correction when particle is already inside."""
    x = np.array([[1.0, 0.0, 0.0]])
    inv_masses = np.array([1.0])
    dt = 0.01

    c = ShellConstraint(i=0, a=5.0, b=5.0, c=5.0, radius=0.5, compliance=0.0)
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    assert np.allclose(x, x_before)


def test_shell_respects_cell_radius():
    """Cell surface (not just center) stays inside shell."""
    # Place center at exactly (a - radius) on x-axis: should be on boundary
    a, r = 5.0, 0.5
    x = np.array([[a - r, 0.0, 0.0]])
    inv_masses = np.array([1.0])
    dt = 0.01

    c = ShellConstraint(i=0, a=a, b=5.0, c=5.0, radius=r, compliance=0.0)
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    # Should not move (exactly on boundary)
    assert np.allclose(x, x_before, atol=1e-6)


def test_shell_ellipsoid_axes():
    """Asymmetric ellipsoid constrains correctly on short axis."""
    # a=5, b=2, c=2; place at y=3 (outside on short axis)
    x = np.array([[0.0, 3.0, 0.0]])
    inv_masses = np.array([1.0])
    dt = 0.01

    c = ShellConstraint(i=0, a=5.0, b=2.0, c=2.0, radius=0.0, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    val = (x[0, 1] / 2.0) ** 2
    assert val <= 1.01


def test_distance_contracts_when_stretched():
    """Stretched pair is pulled back together."""
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    dist = np.linalg.norm(x[0] - x[1])
    assert abs(dist - 1.0) < 0.05


def test_distance_expands_when_compressed():
    """Compressed pair is pushed apart."""
    x = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    dist = np.linalg.norm(x[0] - x[1])
    assert abs(dist - 1.0) < 0.05


def test_distance_preserves_midpoint():
    """Equal-mass correction preserves midpoint."""
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c.project(x, inv_masses, dt, 0.0)

    midpoint = (x[0] + x[1]) / 2
    assert np.allclose(midpoint, [1.0, 0.0, 0.0])


def test_distance_compliance_allows_deviation():
    """Higher compliance allows more deviation from rest length."""
    x_hard = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    x_soft = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c_hard = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c_soft = DistanceConstraint(i=0, j=1, rest_length=1.0, compliance=1.0)

    c_hard.project(x_hard, inv_masses, dt, 0.0)
    c_soft.project(x_soft, inv_masses, dt, 0.0)

    err_hard = abs(np.linalg.norm(x_hard[0] - x_hard[1]) - 1.0)
    err_soft = abs(np.linalg.norm(x_soft[0] - x_soft[1]) - 1.0)

    assert err_hard < err_soft


def _make_test_cell(center, radius):
    """Helper: create a single CellBody and return (cell_body, positions, inv_masses)."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array(center), radius=radius, positions=positions
    )
    inv_masses = np.ones(len(positions))
    return cb, positions, inv_masses


def test_volume_no_correction_at_target():
    """No correction when volume equals target."""
    cb, x, inv_masses = _make_test_cell([0.0, 0.0, 0.0], 1.0)
    dt = 0.01

    c = VolumeConstraint(
        center_idx=cb.center_idx,
        surface_indices=cb.surface_indices,
        triangles=cb.surface_global_triangles(),
        target_volume=cb.target_volume,
        compliance=0.0,
    )
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    assert np.allclose(x, x_before, atol=0.05)


def test_volume_inflates_when_deflated():
    """Deflated cell inflates back toward target volume."""
    cb, x, inv_masses = _make_test_cell([0.0, 0.0, 0.0], 1.0)
    dt = 0.01

    center = x[cb.center_idx]
    for si in cb.surface_indices:
        x[si] = center + (x[si] - center) * 0.5

    c = VolumeConstraint(
        center_idx=cb.center_idx,
        surface_indices=cb.surface_indices,
        triangles=cb.surface_global_triangles(),
        target_volume=cb.target_volume,
        compliance=0.0,
    )

    lambda_acc = 0.0
    for _ in range(20):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    for si in cb.surface_indices:
        dist = np.linalg.norm(x[si] - center)
        assert dist > 0.5


def test_volume_deflates_when_inflated():
    """Inflated cell deflates back toward target volume."""
    cb, x, inv_masses = _make_test_cell([0.0, 0.0, 0.0], 1.0)
    dt = 0.01

    center = x[cb.center_idx]
    for si in cb.surface_indices:
        x[si] = center + (x[si] - center) * 2.0

    c = VolumeConstraint(
        center_idx=cb.center_idx,
        surface_indices=cb.surface_indices,
        triangles=cb.surface_global_triangles(),
        target_volume=cb.target_volume,
        compliance=0.0,
    )

    lambda_acc = 0.0
    for _ in range(20):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    for si in cb.surface_indices:
        dist = np.linalg.norm(x[si] - center)
        assert dist < 2.0


def test_volume_bulge_when_pushed():
    """When pushed from one side, cell bulges on the other side to conserve volume."""
    cb, x, inv_masses = _make_test_cell([0.0, 0.0, 0.0], 1.0)
    dt = 0.01

    # Push +x surface point inward
    x[cb.surface_indices[0]] = np.array([0.3, 0.0, 0.0])

    c = VolumeConstraint(
        center_idx=cb.center_idx,
        surface_indices=cb.surface_indices,
        triangles=cb.surface_global_triangles(),
        target_volume=cb.target_volume,
        compliance=0.0,
    )

    lambda_acc = 0.0
    for _ in range(20):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    # -x point should have moved outward to compensate
    dist_neg_x = np.linalg.norm(x[cb.surface_indices[1]] - x[cb.center_idx])
    assert dist_neg_x > 1.0


# --- AdhesionConstraint tests ---


def test_adhesion_pulls_back_when_separated():
    """Separated pair is pulled back to rest length (attraction)."""
    x = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    lambda_acc = 0.0

    for _ in range(10):
        lambda_acc = c.project(x, inv_masses, dt, lambda_acc)

    dist = np.linalg.norm(x[0] - x[1])
    assert abs(dist - 1.0) < 0.05


def test_adhesion_no_correction_when_overlapping():
    """No correction when overlapping (C < 0); collision handles repulsion."""
    x = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    assert np.allclose(x, x_before)


def test_adhesion_no_correction_at_rest():
    """No correction when at rest length."""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    x_before = x.copy()
    c.project(x, inv_masses, dt, 0.0)

    assert np.allclose(x, x_before, atol=1e-6)


def test_adhesion_compliance_softens():
    """Higher compliance allows more deviation from rest length."""
    x_hard = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    x_soft = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    inv_masses = np.array([1.0, 1.0])
    dt = 0.01

    c_hard = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=0.0)
    c_soft = AdhesionConstraint(i=0, j=1, rest_length=1.0, compliance=1.0)

    c_hard.project(x_hard, inv_masses, dt, 0.0)
    c_soft.project(x_soft, inv_masses, dt, 0.0)

    err_hard = abs(np.linalg.norm(x_hard[0] - x_hard[1]) - 1.0)
    err_soft = abs(np.linalg.norm(x_soft[0] - x_soft[1]) - 1.0)

    assert err_hard < err_soft


def test_triangle_area_no_correction_at_rest():
    """No correction when area equals target."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=float)
    x_before = x.copy()
    inv_masses = np.ones(3)
    c = TriangleAreaConstraint(0, 1, 2, target_area=0.5, compliance=0.0)
    c.project(x, inv_masses, dt=0.01, lambda_acc=0.0)
    np.testing.assert_array_almost_equal(x, x_before)


def test_triangle_area_corrects_stretch():
    """Correction fires and restores area when triangle is stretched."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],  # area = 1.0, target = 0.5
    ], dtype=float)
    inv_masses = np.ones(3)
    c = TriangleAreaConstraint(0, 1, 2, target_area=0.5, compliance=0.0)
    lam = 0.0
    for _ in range(20):
        lam = c.project(x, inv_masses, dt=0.01, lambda_acc=lam)
    n = np.cross(x[1] - x[0], x[2] - x[0])
    area = 0.5 * np.linalg.norm(n)
    assert area <= 0.5 * 1.01


def test_triangle_area_no_correction_on_compression():
    """No correction when area is below target (inequality: stretch only)."""
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],  # area = 0.25 < target 0.5
    ], dtype=float)
    x_before = x.copy()
    inv_masses = np.ones(3)
    c = TriangleAreaConstraint(0, 1, 2, target_area=0.5, compliance=0.0)
    c.project(x, inv_masses, dt=0.01, lambda_acc=0.0)
    np.testing.assert_array_equal(x, x_before)
