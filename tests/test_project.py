import math

import numpy as np

from develmorph.cell_body import create_cell_body
from develmorph.project import (
    AdhesionConstraint,
    CollisionConstraint,
    DistanceConstraint,
    ShellConstraint,
    TetrahedralVolumeConstraint,
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


def test_tet_volume_constraint_basic():
    # Tetrahedron: center at origin, ideal triangle at unit axes.
    # V_ideal = dot((1,0,0), cross((0,1,0),(0,0,1))) / 6 = 1/6
    x = np.array([
        [0.0, 0.0, 0.0],  # center (idx 0)
        [1.0, 0.0, 0.0],  # p0    (idx 1)
        [0.0, 1.0, 0.0],  # p1    (idx 2)
        [0.0, 0.0, 1.0],  # p2    (idx 3)
    ], dtype=float)
    inv_masses = np.ones(4)
    target_v = 1.0 / 6.0

    # Compress: move p0 inward so V < target
    x[1] = [0.5, 0.0, 0.0]
    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_before = np.dot(a, np.cross(b, c)) / 6.0
    assert v_before < target_v

    constraint = TetrahedralVolumeConstraint(0, 1, 2, 3, target_v, compliance=0.0)
    lambda_acc = 0.0
    for _ in range(20):
        lambda_acc = constraint.project(x, inv_masses, 0.01, lambda_acc)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_after = np.dot(a, np.cross(b, c)) / 6.0
    assert abs(v_after - target_v) < abs(v_before - target_v) * 0.1


def test_tet_volume_constraint_gradient():
    # Verify dV/dpc = -(g0 + g1 + g2) via numerical differentiation.
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    def vol(pos):
        a = pos[1] - pos[0]; b = pos[2] - pos[0]; c = pos[3] - pos[0]
        return np.dot(a, np.cross(b, c)) / 6.0

    eps = 1e-6
    num_grad_c = np.zeros(3)
    for d in range(3):
        xp = x.copy(); xp[0, d] += eps
        xm = x.copy(); xm[0, d] -= eps
        num_grad_c[d] = (vol(xp) - vol(xm)) / (2 * eps)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    g0 = np.cross(b, c) / 6.0
    g1 = np.cross(c, a) / 6.0
    g2 = np.cross(a, b) / 6.0
    gc_analytic = -(g0 + g1 + g2)

    np.testing.assert_allclose(gc_analytic, num_grad_c, atol=1e-5)


def test_tet_volume_constraint_equality():
    # With compliance=0, many iterations converge volume to target.
    x = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],  # compressed; ideal position is (1,0,0)
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    inv_masses = np.ones(4)
    target_v = 1.0 / 6.0

    constraint = TetrahedralVolumeConstraint(0, 1, 2, 3, target_v, compliance=0.0)
    lambda_acc = 0.0
    for _ in range(100):
        lambda_acc = constraint.project(x, inv_masses, 0.01, lambda_acc)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    V = np.dot(a, np.cross(b, c)) / 6.0
    assert abs(V - target_v) < 1e-6


def test_tet_volume_sign_below_target():
    # When volume is below target, a single project() call increases it.
    x = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],  # compressed
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    inv_masses = np.ones(4)
    target_v = 1.0 / 6.0

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_before = np.dot(a, np.cross(b, c)) / 6.0
    assert v_before < target_v

    constraint = TetrahedralVolumeConstraint(0, 1, 2, 3, target_v, compliance=0.0)
    constraint.project(x, inv_masses, 0.01, 0.0)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_after = np.dot(a, np.cross(b, c)) / 6.0
    assert v_after > v_before


def test_tet_volume_sign_above_target():
    # When volume exceeds target, a single project() call reduces it.
    x = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],  # expanded
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    inv_masses = np.ones(4)
    target_v = 1.0 / 6.0

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_before = np.dot(a, np.cross(b, c)) / 6.0
    assert v_before > target_v

    constraint = TetrahedralVolumeConstraint(0, 1, 2, 3, target_v, compliance=0.0)
    constraint.project(x, inv_masses, 0.01, 0.0)

    a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
    v_after = np.dot(a, np.cross(b, c)) / 6.0
    assert v_after < v_before


def test_tet_volume_constraint_surface_gradients():
    # Verify g0, g1, g2 match numerical differentiation.
    x = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    def vol(pos):
        a = pos[1] - pos[0]; b = pos[2] - pos[0]; c = pos[3] - pos[0]
        return np.dot(a, np.cross(b, c)) / 6.0

    eps = 1e-6
    for particle_idx in [1, 2, 3]:
        num_grad = np.zeros(3)
        for d in range(3):
            xp = x.copy(); xp[particle_idx, d] += eps
            xm = x.copy(); xm[particle_idx, d] -= eps
            num_grad[d] = (vol(xp) - vol(xm)) / (2 * eps)

        a = x[1] - x[0]; b = x[2] - x[0]; c = x[3] - x[0]
        if particle_idx == 1:
            analytic_grad = np.cross(b, c) / 6.0
        elif particle_idx == 2:
            analytic_grad = np.cross(c, a) / 6.0
        else:
            analytic_grad = np.cross(a, b) / 6.0

        np.testing.assert_allclose(analytic_grad, num_grad, atol=1e-5,
                                   err_msg=f"gradient mismatch for particle {particle_idx}")


# Append at end of tests/test_project.py
from develmorph.project import VoronoiVolumeConstraint


class _FakeVoronoiCell:
    """Minimal stand-in for VoronoiCell used in constraint tests."""
    def __init__(self, volume, gradient):
        self.volume = volume
        self._gradient = np.array(gradient, dtype=float)

    def volume_gradient(self):
        return self._gradient.copy()


def test_voronoi_volume_constraint_no_correction_when_target_met():
    """If current volume equals target, no position change."""
    positions = np.array([[0.0, 0.0, 0.0]])
    inv_masses = np.ones(1)
    vc = VoronoiVolumeConstraint(center_idx=0, target_volume=5.0, compliance=0.0)
    fake_cell = _FakeVoronoiCell(volume=5.0, gradient=[1.0, 0.0, 0.0])
    vc.set_voronoi_cell(fake_cell)

    pos_before = positions.copy()
    vc.project(positions, inv_masses, dt=0.01, lam=0.0)
    np.testing.assert_array_almost_equal(positions, pos_before)


def test_voronoi_volume_constraint_corrects_volume_deficit():
    """If volume < target, center is pushed along gradient to increase volume."""
    positions = np.array([[0.0, 0.0, 0.0]])
    inv_masses = np.ones(1)
    # Target 5.0, current 4.0 -> C = 4.0 - 5.0 = -1.0 -> push in +gradient
    vc = VoronoiVolumeConstraint(center_idx=0, target_volume=5.0, compliance=0.0)
    fake_cell = _FakeVoronoiCell(volume=4.0, gradient=[1.0, 0.0, 0.0])
    vc.set_voronoi_cell(fake_cell)

    vc.project(positions, inv_masses, dt=0.01, lam=0.0)
    # Position should increase in +x (gradient direction, volume deficit corrected)
    assert positions[0, 0] > 0.0, "Expected position to move in +x to restore volume"


def test_voronoi_volume_constraint_no_action_when_no_cell_set():
    """project() does nothing if set_voronoi_cell() was never called."""
    positions = np.array([[1.0, 2.0, 3.0]])
    inv_masses = np.ones(1)
    vc = VoronoiVolumeConstraint(center_idx=0, target_volume=1.0, compliance=0.01)
    pos_before = positions.copy()
    vc.project(positions, inv_masses, dt=0.01, lam=0.0)
    np.testing.assert_array_almost_equal(positions, pos_before)


def test_voronoi_volume_constraint_soft_compliance_reduces_correction():
    """Higher compliance should produce a smaller position correction."""
    inv_masses = np.ones(1)
    dt = 0.01

    def correction_magnitude(compliance):
        pos = np.array([[0.0, 0.0, 0.0]])
        con = VoronoiVolumeConstraint(center_idx=0, target_volume=5.0, compliance=compliance)
        fake_cell = _FakeVoronoiCell(volume=4.0, gradient=[1.0, 0.0, 0.0])
        con.set_voronoi_cell(fake_cell)
        con.project(pos, inv_masses, dt=dt, lam=0.0)
        return abs(pos[0, 0])

    rigid_correction = correction_magnitude(0.0)
    soft_correction = correction_magnitude(0.1)
    assert soft_correction < rigid_correction, (
        "Soft compliance should produce smaller correction than rigid"
    )
