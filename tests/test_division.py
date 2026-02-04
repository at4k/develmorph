import numpy as np

from develmorph.division import divide_cell


def test_division_increases_cell_number():

    pos = np.array([[0, 0, 0]])
    r = np.array([1.0])

    new_pos, new_r = divide_cell(pos, r, 0)

    assert len(new_pos) == 2
    assert len(new_r) == 2


def test_symmetric_division_conserves_volume():
    """Verify symmetric division (volume_ratio=0.5) conserves total volume."""
    pos = np.array([[0.0, 0.0, 0.0]])
    r = np.array([1.0])

    new_pos, new_r = divide_cell(pos, r, 0, volume_ratio=0.5)

    # Volume of sphere: V = (4/3) * pi * r^3
    parent_volume = (4 / 3) * np.pi * r[0] ** 3
    daughter_volumes = sum((4 / 3) * np.pi * ri**3 for ri in new_r)

    np.testing.assert_almost_equal(daughter_volumes, parent_volume)

    # Both daughters should have equal radius
    np.testing.assert_almost_equal(new_r[0], new_r[1])


def test_asymmetric_division_conserves_volume():
    """Verify asymmetric division conserves total volume."""
    pos = np.array([[0.0, 0.0, 0.0]])
    r = np.array([1.0])

    # 60:40 ratio like P0 -> AB + P1
    new_pos, new_r = divide_cell(pos, r, 0, volume_ratio=0.6)

    parent_volume = (4 / 3) * np.pi * r[0] ** 3
    daughter_volumes = sum((4 / 3) * np.pi * ri**3 for ri in new_r)

    np.testing.assert_almost_equal(daughter_volumes, parent_volume)


def test_asymmetric_division_volume_ratio():
    """Verify volume ratio is correctly applied to daughters."""
    pos = np.array([[0.0, 0.0, 0.0]])
    r = np.array([1.0])

    volume_ratio = 0.6
    new_pos, new_r = divide_cell(pos, r, 0, volume_ratio=volume_ratio)

    # Calculate actual volume fractions
    v1 = (4 / 3) * np.pi * new_r[0] ** 3
    v2 = (4 / 3) * np.pi * new_r[1] ** 3
    total = v1 + v2

    actual_ratio = v1 / total

    np.testing.assert_almost_equal(actual_ratio, volume_ratio)


def test_asymmetric_division_produces_different_radii():
    """Verify asymmetric division produces daughters with different radii."""
    pos = np.array([[0.0, 0.0, 0.0]])
    r = np.array([1.0])

    new_pos, new_r = divide_cell(pos, r, 0, volume_ratio=0.6)

    # First daughter (60% volume) should be larger
    assert new_r[0] > new_r[1]


def test_division_along_specified_axis():
    """Verify daughters are positioned along the specified axis."""
    pos = np.array([[0.0, 0.0, 0.0]])
    r = np.array([1.0])
    axis = np.array([1.0, 0.0, 0.0])  # x-axis

    new_pos, new_r = divide_cell(pos, r, 0, axis=axis, volume_ratio=0.5)

    # Daughters should be separated along x-axis
    diff = new_pos[0] - new_pos[1]
    diff_normalized = diff / np.linalg.norm(diff)

    np.testing.assert_array_almost_equal(np.abs(diff_normalized), axis)
