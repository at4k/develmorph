import numpy as np

from develmorph.cell_body import create_cell_body


def divide_cell_body(parent, positions, axis=None, volume_ratio=0.5, area_threshold=None):
    """
    Divide a CellBody into two daughters.

    Removes parent particles and creates two new CellBodies with
    octahedron surface points. Volume is distributed according to
    volume_ratio.

    Args:
        parent: CellBody to divide
        positions: (N, 3) particle positions array
        axis: division axis (unit vector). Random if None.
        volume_ratio: fraction of volume for first daughter (0.0-1.0)
        area_threshold: if set, passed to create_cell_body for adaptive subdivision

    Returns:
        (daughters, new_positions): list of 2 CellBodies and updated positions
    """
    if axis is None:
        axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    parent_center = positions[parent.center_idx].copy()
    parent_volume = parent.target_volume

    # Octahedron volume: V = (4/3) * r^3, so r = (3V/4)^(1/3)
    parent_radius = (3.0 * parent_volume / 4.0) ** (1.0 / 3.0)

    # Daughter radii from volume ratio
    r1 = parent_radius * (volume_ratio ** (1.0 / 3.0))
    r2 = parent_radius * ((1.0 - volume_ratio) ** (1.0 / 3.0))

    # Daughter centers along axis
    offset1 = 0.6 * r1
    offset2 = 0.6 * r2
    center1 = parent_center + offset1 * axis
    center2 = parent_center - offset2 * axis

    # Remove parent particles from positions array
    parent_indices = sorted(parent.all_indices(), reverse=True)
    positions = np.delete(positions, parent_indices, axis=0)

    # Create daughters (appending to cleaned positions)
    d1, positions = create_cell_body(
        center=center1,
        radius=r1,
        positions=positions,
        membrane_compliance=parent.membrane_compliance,
        area_threshold=area_threshold,
    )
    d2, positions = create_cell_body(
        center=center2,
        radius=r2,
        positions=positions,
        membrane_compliance=parent.membrane_compliance,
        area_threshold=area_threshold,
    )

    return [d1, d2], positions
