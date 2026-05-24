import numpy as np

from develmorph.cell_body import create_cell_body, TOPOLOGY_VOLUME_FACTOR
from develmorph.physics_cell import PhysicsCell


def divide_cell_body(
    parent,
    positions,
    axis=None,
    volume_ratio=0.5,
    area_threshold=None,
    daughter_names=(None, None),
    daughter_orientations=(None, None),
):
    """
    Divide a CellBody into two daughters.

    Removes parent particles and creates two new CellBodies with
    CellBody surface points. Volume is distributed according to
    volume_ratio.

    Args:
        parent: CellBody to divide
        positions: (N, 3) particle positions array
        axis: division axis (unit vector). Random if None.
        volume_ratio: fraction of volume for first daughter (0.0-1.0)
        area_threshold: if set, passed to create_cell_body for adaptive subdivision
        daughter_names: (name1, name2) biological names for daughters
        daughter_orientations: (orient1, orient2) 3x3 orientation matrices.
            If None for either, inherits parent orientation.

    Returns:
        (daughters, new_positions): list of 2 CellBodies and updated positions
    """
    if axis is None:
        axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    parent_center = positions[parent.center_idx].copy()
    parent_volume = parent.target_volume

    # Radius from volume: V = TOPOLOGY_VOLUME_FACTOR[topology] * r^3
    vol_factor = TOPOLOGY_VOLUME_FACTOR[parent.topology]
    parent_radius = (parent_volume / vol_factor) ** (1.0 / 3.0)

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

    orient1 = (
        daughter_orientations[0].copy()
        if daughter_orientations[0] is not None
        else parent.orientation.copy()
    )
    orient2 = (
        daughter_orientations[1].copy()
        if daughter_orientations[1] is not None
        else parent.orientation.copy()
    )

    d1, positions = create_cell_body(
        center=center1,
        radius=r1,
        positions=positions,
        membrane_compliance=parent.membrane_compliance,
        area_threshold=area_threshold,
        topology=parent.topology,
        name=daughter_names[0],
        orientation=orient1,
    )
    d2, positions = create_cell_body(
        center=center2,
        radius=r2,
        positions=positions,
        membrane_compliance=parent.membrane_compliance,
        area_threshold=area_threshold,
        topology=parent.topology,
        name=daughter_names[1],
        orientation=orient2,
    )

    return [d1, d2], positions


def divide_physics_cell(
    parent,
    positions,
    axis=None,
    volume_ratio=0.5,
    daughter_names=(None, None),
    daughter_orientations=(None, None),
):
    """
    Divide a PhysicsCell into two daughters.

    Removes the parent center from positions and appends two daughter centers.
    Volume is distributed according to volume_ratio; radii are scaled so that
    (4/3)*pi*r^3 sums to the parent volume.

    The caller (WorldPC) is responsible for shifting center_idx of all
    surviving cells whose index was above parent.center_idx.

    Args:
        parent: PhysicsCell to divide
        positions: (N, 3) positions array
        axis: division axis unit vector; random if None
        volume_ratio: fraction of volume for first daughter (default 0.5)
        daughter_names: (name1, name2) biological names
        daughter_orientations: (orient1, orient2) 3x3 matrices; inherits if None

    Returns:
        (daughters, new_positions): list of 2 PhysicsCells and updated positions
    """
    if axis is None:
        axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)

    r = parent.radius
    r1 = r * (volume_ratio ** (1.0 / 3.0))
    r2 = r * ((1.0 - volume_ratio) ** (1.0 / 3.0))

    c = positions[parent.center_idx].copy()
    c1 = c + 0.6 * r1 * axis
    c2 = c - 0.6 * r2 * axis

    # Remove parent center; surviving cells at indices > parent.center_idx shift down
    parent_idx = parent.center_idx
    positions = np.delete(positions, parent_idx, axis=0)

    # Append daughters at the end
    idx_d1 = len(positions)
    idx_d2 = len(positions) + 1
    positions = np.vstack([positions, c1.reshape(1, 3), c2.reshape(1, 3)])

    orient1 = (
        daughter_orientations[0].copy()
        if daughter_orientations[0] is not None
        else parent.orientation.copy()
    )
    orient2 = (
        daughter_orientations[1].copy()
        if daughter_orientations[1] is not None
        else parent.orientation.copy()
    )

    d1 = PhysicsCell(center_idx=idx_d1, radius=r1, name=daughter_names[0], orientation=orient1)
    d2 = PhysicsCell(center_idx=idx_d2, radius=r2, name=daughter_names[1], orientation=orient2)

    return [d1, d2], positions
