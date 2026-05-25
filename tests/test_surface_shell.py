# tests/test_surface_shell.py
import numpy as np
from develmorph.physics_cell import create_physics_cell
from develmorph.contact import ContactRecord, compute_contacts
from develmorph.viz import SurfaceShell, reconstruct_surface_shells


def test_surface_shell_sphere_points_on_radius():
    """Default shell (no contacts) gives points at distance r from center."""
    shell = SurfaceShell(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.5,
        contacts=[],
    )
    points = shell.mesh_points(n_subdivisions=0)
    dists = np.linalg.norm(points, axis=1)
    np.testing.assert_allclose(dists, 1.5, atol=1e-10)


def test_reconstruct_surface_shells_one_cell():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([1.0, 2.0, 3.0]), 1.0, positions)
    shells = reconstruct_surface_shells([cell], positions, [])
    assert len(shells) == 1
    np.testing.assert_array_almost_equal(shells[0].center, [1.0, 2.0, 3.0])
    assert shells[0].radius == 1.0


def test_surface_shell_contact_flattens_toward_contact_plane():
    """
    With a contact where normal=[-1,0,0] (cell i is LEFT of cell j, n points left),
    the contact face of cell i is its +x side (facing toward j).

    - The +x point (contact-facing) must be pulled INWARD: x < radius = 1.0
    - The -x point (back side) must NOT be moved: x == -radius = -1.0

    Bug 1: flattening proj > 0 on n (not -n) targets the BACK side, not contact side.
    Bug 2: re-normalizing dirs after flattening puts points back on the sphere surface.
    Both bugs mean the contact-facing point is never actually flattened.
    """
    contact = ContactRecord(
        i=0, j=1, overlap=0.3, area=0.5,
        normal=np.array([-1.0, 0.0, 0.0]),  # unit vector from j to i (i is left of j)
    )
    shell = SurfaceShell(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        contacts=[contact],
    )
    points = shell.mesh_points(n_subdivisions=0)

    x_vals = points[:, 0]
    max_x = x_vals.max()
    min_x = x_vals.min()

    # Contact-facing point (+x side) must be pulled inward: strictly less than radius
    assert max_x < 1.0 - 1e-6, (
        f"Contact-facing point (+x) not flattened: max_x={max_x:.6f}, expected < {1.0 - 1e-6:.6f}"
    )

    # Back point (-x side) must remain untouched at -radius
    assert abs(min_x - (-1.0)) < 1e-6, (
        f"Back point (-x) incorrectly moved: min_x={min_x:.6f}, expected -1.0"
    )
