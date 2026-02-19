import numpy as np

from develmorph.project import CollisionConstraint, ShellConstraint


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
