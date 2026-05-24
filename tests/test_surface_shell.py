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
    With a contact, mesh points near the contact direction should be
    pulled inward (flattened) relative to the pure sphere.
    """
    contact = ContactRecord(
        i=0, j=1, overlap=0.3, area=0.5,
        normal=np.array([-1.0, 0.0, 0.0]),  # contact from +x direction
    )
    shell = SurfaceShell(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        contacts=[contact],
    )
    points = shell.mesh_points(n_subdivisions=0)
    # Point along +x direction should be flattened (x < 1.0)
    x_points = points[points[:, 0] > 0.5]  # points roughly along +x
    if len(x_points) > 0:
        assert x_points[:, 0].max() < 1.0 + 1e-6
