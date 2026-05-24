# tests/test_contact.py
import numpy as np
from develmorph.physics_cell import create_physics_cell
from develmorph.contact import ContactRecord, compute_contacts


def test_compute_contacts_overlapping_pair():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)
    # overlap = 2.0 - 1.5 = 0.5

    contacts = compute_contacts([c1, c2], positions)

    assert len(contacts) == 1
    c = contacts[0]
    assert c.i == 0 and c.j == 1
    assert abs(c.overlap - 0.5) < 1e-10
    assert c.area > 0.0


def test_compute_contacts_separated_pair():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([3.0, 0.0, 0.0]), 1.0, positions)
    # dist=3, r1+r2=2: no overlap

    contacts = compute_contacts([c1, c2], positions)
    assert len(contacts) == 0


def test_contact_area_formula():
    """Contact area = pi * delta * r_harmonic."""
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)

    contacts = compute_contacts([c1, c2], positions)
    record = contacts[0]

    delta = 0.5
    r_harm = 2 * 1.0 * 1.0 / (1.0 + 1.0)  # = 1.0
    expected_area = np.pi * delta * r_harm
    assert abs(record.area - expected_area) < 1e-10


def test_contact_normal_points_from_j_to_i():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)

    contacts = compute_contacts([c1, c2], positions)
    n = contacts[0].normal
    # normal should point from j (right) to i (left), i.e. [-1, 0, 0]
    np.testing.assert_array_almost_equal(n, [-1.0, 0.0, 0.0])


def test_compute_contacts_three_cells():
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)
    c3, positions = create_physics_cell(np.array([3.4, 0.0, 0.0]), 1.0, positions)
    # c1-c2 overlap, c2-c3 overlap, c1-c3 separated

    contacts = compute_contacts([c1, c2, c3], positions)
    pairs = {(c.i, c.j) for c in contacts}
    assert (0, 1) in pairs
    assert (1, 2) in pairs
    assert (0, 2) not in pairs
