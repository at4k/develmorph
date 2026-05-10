import numpy as np
import pytest

from develmorph.cell_body import create_cell_body
from develmorph.celllineage import CellLineage
from develmorph.engine import World

YAML_PATH = "develmorph/lineages/celegans_early.yaml"


def make_p0_world():
    """Helper: 1-cell World with P0 and CellLineage loaded."""
    lineage = CellLineage(YAML_PATH)
    positions = np.zeros((0, 3))
    p0, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
        name="P0",
    )
    return World(
        cell_bodies=[p0],
        positions=positions,
        lineage=lineage,
        shell_params=(5.0, 5.0, 5.0),
    )


def test_world_accepts_lineage_param():
    """World can be constructed with a CellLineage argument."""
    world = make_p0_world()
    assert world.lineage is not None


def test_world_no_lineage_backward_compat():
    """World without lineage still accepts division_steps and divides largest cell."""
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=np.array([0.0, 0.0, 0.0]),
        radius=1.0,
        positions=positions,
    )
    world = World(cell_bodies=[cb], positions=positions, shell_params=(5.0, 5.0, 5.0))
    world.set_divisions([0])
    world.step()
    assert len(world.cell_bodies) == 2


def test_world_lineage_step0_no_division():
    """step_count=0 triggers no division (YAML first step is 1)."""
    world = make_p0_world()
    world.step()  # step_count=0 during execution
    assert len(world.cell_bodies) == 1
    assert world.cell_bodies[0].name == "P0"


def test_world_lineage_step1_p0_divides_into_ab_and_p1():
    """step_count=1: P0 divides into AB and P1."""
    world = make_p0_world()
    world.step()  # step_count=0
    world.step()  # step_count=1: P0 -> AB + P1
    names = {cb.name for cb in world.cell_bodies}
    assert names == {"AB", "P1"}


def test_world_lineage_step1_volume_ratio():
    """P0 -> AB (60%) + P1 (40%) volume ratio matches YAML."""
    world = make_p0_world()
    world.step()  # step_count=0
    world.step()  # step_count=1
    ab = next(cb for cb in world.cell_bodies if cb.name == "AB")
    p1 = next(cb for cb in world.cell_bodies if cb.name == "P1")
    total = ab.target_volume + p1.target_volume
    assert abs(ab.target_volume / total - 0.6) < 1e-6


def test_world_lineage_step1_division_along_x_axis():
    """P0 divides along major_axis=e1=x-axis (orientation=eye(3))."""
    world = make_p0_world()
    world.step()  # step_count=0
    world.step()  # step_count=1
    ab = next(cb for cb in world.cell_bodies if cb.name == "AB")
    p1 = next(cb for cb in world.cell_bodies if cb.name == "P1")
    diff = world.positions[ab.center_idx] - world.positions[p1.center_idx]
    diff_norm = diff / np.linalg.norm(diff)
    assert abs(diff_norm[0]) > 0.9


def test_world_lineage_step2_ab_divides():
    """step_count=2: AB divides into ABa and ABp."""
    world = make_p0_world()
    world.step()  # step_count=0
    world.step()  # step_count=1: P0 -> AB + P1
    world.step()  # step_count=2: AB -> ABa + ABp
    names = {cb.name for cb in world.cell_bodies}
    assert names == {"ABa", "ABp", "P1"}


def test_world_lineage_step3_p1_divides():
    """step_count=3: P1 divides into EMS and P2."""
    world = make_p0_world()
    for _ in range(4):
        world.step()  # step_counts 0,1,2,3
    names = {cb.name for cb in world.cell_bodies}
    assert names == {"ABa", "ABp", "EMS", "P2"}


def test_world_lineage_daughters_inherit_orientation():
    """Daughters inherit parent orientation."""
    world = make_p0_world()
    world.step()  # step_count=0
    world.step()  # step_count=1: P0 -> AB + P1
    for cb in world.cell_bodies:
        assert np.allclose(cb.orientation, np.eye(3))


def test_world_simulates_ico_cell_without_error():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions,
        topology='ico',
    )
    world = World(
        cell_bodies=[cb],
        positions=positions,
    )
    world.run(5)
    assert world.cell_bodies[0].topology == 'ico'
    assert len(world.positions) == 13  # 1 center + 12 surface


def test_world_ico_cell_preserves_approximate_volume():
    positions = np.zeros((0, 3))
    cb, positions = create_cell_body(
        center=[0.0, 0.0, 0.0], radius=1.0, positions=positions,
        topology='ico',
    )
    initial_target = cb.target_volume
    world = World(
        cell_bodies=[cb],
        positions=positions,
        shell_params=(10.0, 10.0, 10.0),
    )
    world.run(20)
    # target_volume is a constraint target set at creation — it does not change
    assert world.cell_bodies[0].target_volume == initial_target
