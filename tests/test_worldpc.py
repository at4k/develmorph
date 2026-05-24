import numpy as np
import pytest
from develmorph.celllineage import CellLineage
from develmorph.engine import WorldPC
from develmorph.physics_cell import create_physics_cell
from develmorph.solver import step_pc


def test_step_pc_returns_finite_positions():
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, new_vel = step_pc(
        positions, velocities, inv_masses,
        dt=0.01,
        physics_cells=[cell],
        shell_params=(5.0, 5.0, 5.0),
        n_iter=10,
        damping=1.0,
    )
    assert np.all(np.isfinite(new_pos))


def test_step_pc_pushes_center_inside_shell():
    """Center outside effective ellipsoid (shell - radius) is pushed inside."""
    positions = np.zeros((0, 3))
    # Shell (2.0, 2.0, 2.0), r=0.5 -> effective (1.5, 1.5, 1.5)
    # Place center at x=1.8, outside effective shell
    cell, positions = create_physics_cell(np.array([1.8, 0.0, 0.0]), 0.5, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    for _ in range(20):
        positions, velocities = step_pc(
            positions, velocities, inv_masses,
            dt=0.01,
            physics_cells=[cell],
            shell_params=(2.0, 2.0, 2.0),
            n_iter=10,
            damping=1.0,
        )

    x, y, z = positions[cell.center_idx]
    a_eff, b_eff, c_eff = 1.5, 1.5, 1.5
    C = (x / a_eff) ** 2 + (y / b_eff) ** 2 + (z / c_eff) ** 2
    assert C <= 1.0 + 1e-6


def test_step_pc_two_cells_separated_after_settling():
    """Two overlapping cells are pushed apart by CollisionConstraint."""
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([1.5, 0.0, 0.0]), 1.0, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    for _ in range(50):
        positions, velocities = step_pc(
            positions, velocities, inv_masses,
            dt=0.01,
            physics_cells=[c1, c2],
            shell_params=(10.0, 10.0, 10.0),
            n_iter=10,
            damping=1.0,
        )

    dist = np.linalg.norm(positions[c1.center_idx] - positions[c2.center_idx])
    assert dist >= c1.radius + c2.radius - 1e-6


def test_step_pc_damping_one_zeroes_velocity():
    """damping=1.0 means velocity is fully zeroed after each step."""
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    velocities = np.ones_like(positions) * 5.0
    inv_masses = np.ones(len(positions))

    _, new_vel = step_pc(
        positions, velocities, inv_masses,
        dt=0.01,
        physics_cells=[cell],
        shell_params=(10.0, 10.0, 10.0),
        n_iter=1,
        damping=1.0,
    )
    assert np.allclose(new_vel, 0.0)


def test_step_pc_asymmetric_shell_confinement():
    """
    Narrow shell axis confines cell more than wide axis.
    Shell (3.0, 3.0, 1.5), r=1.0 -> effective (2.0, 2.0, 0.5).
    Center at (0, 0, 0.8): outside c-axis limit, inside a,b limits.
    """
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.8]), 1.0, positions)
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    for _ in range(30):
        positions, velocities = step_pc(
            positions, velocities, inv_masses,
            dt=0.01,
            physics_cells=[cell],
            shell_params=(3.0, 3.0, 1.5),
            n_iter=10,
            damping=1.0,
        )

    x, y, z = positions[cell.center_idx]
    assert abs(z) <= 0.5 + 1e-6
    assert abs(x) < 0.1
    assert abs(y) < 0.1


YAML_PATH = "develmorph/lineages/celegans_early.yaml"


def make_p0_worldpc():
    lineage = CellLineage(YAML_PATH)
    positions = np.zeros((0, 3))
    p0, positions = create_physics_cell(
        np.array([0.0, 0.0, 0.0]), radius=1.0, positions=positions, name="P0"
    )
    return WorldPC(
        physics_cells=[p0],
        positions=positions,
        lineage=lineage,
        shell_params=(5.0, 5.0, 5.0),
        damping=1.0,
    )


def test_worldpc_step_increments_count():
    world = make_p0_worldpc()
    world.step()
    assert world.step_count == 1


def test_worldpc_positions_finite_after_step():
    world = make_p0_worldpc()
    world.step()
    assert np.all(np.isfinite(world.positions))


def test_worldpc_step0_no_division():
    """Step at count=0 does not trigger division (YAML first step is 1)."""
    world = make_p0_worldpc()
    world.step()  # step_count=0 during execution
    assert len(world.physics_cells) == 1
    assert world.physics_cells[0].name == "P0"


def test_worldpc_step1_p0_divides_into_ab_p1():
    world = make_p0_worldpc()
    world.step()  # step_count=0
    world.step()  # step_count=1: P0 -> AB + P1
    names = {pc.name for pc in world.physics_cells}
    assert names == {"AB", "P1"}


def test_worldpc_four_cell_stage():
    world = make_p0_worldpc()
    for _ in range(4):
        world.step()  # step_counts 0,1,2,3
    names = {pc.name for pc in world.physics_cells}
    assert names == {"ABa", "ABp", "EMS", "P2"}


def test_worldpc_positions_length_equals_cell_count():
    """positions array has exactly one row per PhysicsCell."""
    world = make_p0_worldpc()
    for _ in range(4):
        world.step()
    assert len(world.positions) == len(world.physics_cells)


def test_worldpc_division_volume_ratio():
    """P0 -> AB (60%) + P1 (40%)."""
    world = make_p0_worldpc()
    world.step()  # step_count=0
    world.step()  # step_count=1
    ab = next(pc for pc in world.physics_cells if pc.name == "AB")
    p1 = next(pc for pc in world.physics_cells if pc.name == "P1")
    r1, r2 = ab.radius, p1.radius
    total = r1 ** 3 + r2 ** 3
    assert abs(r1 ** 3 / total - 0.6) < 1e-6


def test_worldpc_no_lineage_no_division():
    """WorldPC without lineage runs without dividing."""
    positions = np.zeros((0, 3))
    cell, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    world = WorldPC(
        physics_cells=[cell],
        positions=positions,
        shell_params=(5.0, 5.0, 5.0),
        damping=1.0,
    )
    world.step()
    assert len(world.physics_cells) == 1


def test_step_pc_area_adhesion_attracts_cells():
    """
    With use_area_adhesion=True and an active adhesion pair beyond rest_length,
    cells should move toward each other (toward rest_length = r_i + r_j).
    """
    positions = np.zeros((0, 3))
    c1, positions = create_physics_cell(np.array([0.0, 0.0, 0.0]), 1.0, positions)
    c2, positions = create_physics_cell(np.array([2.1, 0.0, 0.0]), 1.0, positions)
    # rest_length = 2.0, current dist = 2.1 → adhesion should pull them together

    active_adhesions = {(0, 1)}
    velocities = np.zeros_like(positions)
    inv_masses = np.ones(len(positions))

    new_pos, _ = step_pc(
        positions, velocities, inv_masses,
        dt=0.01,
        physics_cells=[c1, c2],
        shell_params=(10.0, 10.0, 10.0),
        n_iter=10,
        active_adhesions=active_adhesions,
        adhesion_compliance=0.1,
        damping=1.0,
        use_area_adhesion=True,
    )

    dist_after = np.linalg.norm(new_pos[c1.center_idx] - new_pos[c2.center_idx])
    assert dist_after < 2.1  # moved toward rest_length
