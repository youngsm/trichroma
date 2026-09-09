"""Independent checks that acceleration cells cannot hide physical interfaces."""

from dataclasses import replace
import numpy as np
import pytest

from chroma.triton.primitives import Bounds, BoxVolume, BoundedObstacle, PrimitiveScene, WireArray
from chroma.triton.regions import compile_regions, subtract_box


def box(name="medium", lower=(-10, -10, -10), upper=(10, 10, 10), inside=3, outside=0):
    return BoxVolume(name, Bounds(lower, upper), inside, outside)


def test_nested_materials_and_surface_exclusion():
    outer = box()
    inner = box("insert", (-1, -2, -3), (1, 2, 3), inside=7, outside=3)
    result = compile_regions(PrimitiveScene((outer, inner)))
    assert result.select([[0, 0, 0]]).material == 7
    assert result.select([[5, 0, 0]]).material == 3
    for region in result.regions:
        if region.material == 3:
            assert not region.bounds.overlaps(inner.bounds)
        else:
            assert inner.bounds.encloses(region.bounds)
        # Physical surfaces themselves never belong to acceleration cells.
        assert not region.bounds.contains([1.0, 0, 0])
    assert not result.rejected_domains


def test_material_ambiguity_disables_only_affected_domain():
    outer = box()
    child = box("insert", (-1, -1, -1), (1, 1, 1), inside=7, outside=9)
    result = compile_regions(PrimitiveScene((outer, child)))
    assert set(dict(result.rejected_domains)) == {"medium"}
    assert {r.material for r in result.regions} == {7}
    unknown = BoundedObstacle("unclassified_mesh", Bounds((-2, -2, -2), (2, 2, 2)), None)
    assert not compile_regions(PrimitiveScene((outer,), (unknown,))).regions


def test_partial_overlap_is_not_given_a_guessed_material():
    a, b = box(), box("second", (0, 0, 0), (20, 20, 20))
    result = compile_regions(PrimitiveScene((a, b)))
    assert not result.regions and len(result.rejected_domains) == 2


def test_subtraction_preserves_volume_without_overlapping_fragments():
    rng = np.random.default_rng(251)
    whole = Bounds((-4, -3, -2), (4, 3, 2))
    for _ in range(100):
        corners = rng.uniform(-6, 6, (2, 3))
        cut = Bounds(corners.min(0), corners.max(0))
        cells = subtract_box(whole, cut)
        intersection = np.prod(
            np.maximum(0, np.minimum(whole.upper, cut.upper) - np.maximum(whole.lower, cut.lower))
        )
        assert sum(c.volume for c in cells) == pytest.approx(whole.volume - intersection)
        for i, cell in enumerate(cells):
            assert whole.encloses(cell) and not cell.overlaps(cut)
            assert all(not cell.overlaps(other) for other in cells[i + 1 :])


def test_random_obstacles_never_occur_inside_certified_regions():
    rng = np.random.default_rng(765)
    obstacles = []
    for i in range(20):
        center = rng.uniform(-9, 9, 3)
        half = rng.uniform(0.1, 2, 3)
        obstacles.append(BoundedObstacle(str(i), Bounds(center - half, center + half), 3))
    scene = PrimitiveScene((box(),), tuple(obstacles))
    result = compile_regions(scene, max_regions=4096)
    points = rng.uniform(-10, 10, (100000, 3))
    blocked = np.zeros(len(points), bool)
    covered = np.zeros(len(points), bool)
    for obstacle in obstacles:
        blocked |= obstacle.bounds.contains(points)
    for region in result.regions:
        covered |= region.bounds.contains(points)
    assert not np.any(covered & blocked)
    assert covered.sum() > 0.99 * np.count_nonzero(~blocked)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_wire_array_discovery_has_no_preferred_world_axis(axis):
    axes = np.roll(np.eye(3), axis, axis=1)
    array = WireArray(
        "array",
        np.array([1.0, -2.0, 0.5]),
        axes[1],
        axes[2],
        pitch=1.0,
        radius=0.1,
        axial_limits=(-10.0, 10.0),
        offset=0.0,
        first=-10,
        last=10,
        exterior_material=3,
    )
    result = compile_regions(PrimitiveScene((box(),), wires=(array,)))
    n = np.cross(array.u, array.v)
    samples = []
    for k in (-10, -3, 0, 4, 10):
        for a in np.linspace(-10, 10, 9):
            for angle in np.linspace(0, 2 * np.pi, 41):
                samples.append(
                    array.origin
                    + a * array.u
                    + k * array.pitch * array.v
                    + array.radius * (np.cos(angle) * array.v + np.sin(angle) * n)
                )
    samples = np.asarray(samples)
    bound = array.obstacle().bounds
    assert np.all(samples >= bound.lower) and np.all(samples <= bound.upper)
    assert result.regions
    for region in result.regions:
        assert not region.bounds.contains(samples).any()


def test_oblique_wire_bounds_and_translated_detector():
    rng = np.random.default_rng(713)
    frame, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    origin = np.array([3500.0, -2700.0, 990.0])
    wire = WireArray(
        "rotated", origin, frame[:, 0], frame[:, 1], 2.0, 0.075, (-2.0, 2.0), 0.3, -2, 3, 3
    )
    theta = rng.uniform(0, 2 * np.pi, 10000)
    k = rng.integers(-2, 4, 10000)
    a = rng.uniform(-2, 2, 10000)
    points = (
        origin
        + a[:, None] * wire.u
        + (0.3 + 2 * k[:, None]) * wire.v
        + 0.075
        * (np.cos(theta)[:, None] * wire.v + np.sin(theta)[:, None] * np.cross(wire.u, wire.v))
    )
    obstacle = wire.obstacle()
    assert np.all(points >= obstacle.bounds.lower) and np.all(points <= obstacle.bounds.upper)
    scene = PrimitiveScene((box(lower=origin - 10, upper=origin + 10),), wires=(wire,))
    result = compile_regions(scene)
    assert result.regions
    assert all(not r.bounds.contains(points).any() for r in result.regions)


def test_renaming_reordering_and_limits_do_not_change_safety():
    scene = PrimitiveScene(
        (box(),),
        (
            BoundedObstacle("a", Bounds((-2, -2, -2), (2, 2, 2)), 3),
            BoundedObstacle("b", Bounds((3, 3, 3), (7, 7, 7)), 3),
        ),
    )
    full = compile_regions(scene)
    renamed = compile_regions(
        replace(
            scene, volumes=(replace(scene.volumes[0], name="x"),), obstacles=scene.obstacles[::-1]
        )
    )
    np.testing.assert_array_equal(
        [r.bounds.lower for r in full.regions], [r.bounds.lower for r in renamed.regions]
    )
    limited = compile_regions(scene, max_regions=1, max_fragments=2)
    assert len(limited.regions) == 1 and limited.dropped_cells > 0
    assert not any(limited.regions[0].bounds.overlaps(o.bounds) for o in scene.obstacles)


def test_thin_and_degenerate_obstacles_and_float32_rounding():
    domain = box(lower=(2000, -10, -10), upper=(2010, 10, 10))
    plane = BoundedObstacle("plane", Bounds((2005, -10, -10), (2005, 10, 10)), 3)
    result = compile_regions(PrimitiveScene((domain,), (plane,)))
    assert len(result.regions) == 2
    assert all(not r.bounds.contains([2005, 0, 0]) for r in result.regions)
    small = Bounds((1, 1, 1), (1 + 1.0e-10, 2, 2))
    assert small.float32_interior() is None
    exact = Bounds((-3, -2160, -2160), (3, 2160, 2160))
    np.testing.assert_array_equal(exact.float32_interior().lower, exact.lower)
    np.testing.assert_array_equal(exact.float32_interior().upper, exact.upper)
    assert not exact.float32_interior().contains([3, 0, 0])


def test_snapshot_ownership_and_invalid_geometry():
    lower = np.array([-1.0, -1.0, -1.0])
    bounds = Bounds(lower, [1, 1, 1])
    lower[:] = 7
    np.testing.assert_array_equal(bounds.lower, [-1, -1, -1])
    with pytest.raises(ValueError):
        bounds.lower[0] = 3
    with pytest.raises(ValueError, match="coincident"):
        PrimitiveScene((box(), replace(box(), name="other")))
    with pytest.raises(ValueError):
        Bounds([0, 0, np.nan], [1, 1, 1])
    with pytest.raises(ValueError):
        compile_regions(PrimitiveScene((box(),)), max_regions=1.5)


def test_shared_rotated_mesh_instances_are_excluded_from_regions():
    from chroma.triton.primitives import Mesh, MeshInstance

    mesh = Mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 2]], [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    )
    angle = 0.731
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    instances = tuple(
        MeshInstance(f"sensor:{i}", mesh, rotation, [x, 0, 0], 3) for i, x in enumerate([-5, 5])
    )
    scene = PrimitiveScene((box(),), instances=instances)
    result = compile_regions(scene)
    assert result.regions
    rng = np.random.default_rng(711)
    barycentric = rng.dirichlet([1, 1, 1], 10000)
    faces = mesh.vertices[mesh.triangles[rng.integers(0, 4, 10000)]]
    surface_points = np.einsum("ni,nij->nj", barycentric, faces)
    for instance in instances:
        assert instance.mesh is mesh
        world = surface_points @ instance.rotation.T + instance.translation
        assert all(not r.bounds.contains(world).any() for r in result.regions)


def test_dense_array_budget_checks_every_obstacle_and_reports_coarse_cover():
    rng = np.random.default_rng(814)
    obstacles = []
    for i in range(5000):
        center = rng.uniform(-8.0, 8.0, 3)
        center[0] = (1 if i % 2 else -1) * rng.uniform(4.0, 9.0)
        obstacles.append(BoundedObstacle(str(i), Bounds(center - 0.1, center + 0.1), 3))
    result = compile_regions(PrimitiveScene((box(),), obstacles))
    assert result.coarse_domains == ("medium",)
    assert len(result.regions) == 1
    region = result.regions[0]
    assert region.bounds.contains([0, 0, 0])
    assert all(not region.bounds.overlaps(o.bounds) for o in obstacles)
    # Even an obstacle beyond the subdivision budget can forbid certification.
    obstacles[-1] = replace(obstacles[-1], exterior_material=None)
    assert not compile_regions(PrimitiveScene((box(),), obstacles)).regions
    obstacles[-1] = BoundedObstacle("center", Bounds([-1] * 3, [1] * 3), 3)
    assert not compile_regions(PrimitiveScene((box(),), obstacles)).regions
