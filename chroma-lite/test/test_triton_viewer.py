"""Camera conventions, retained geometry, and end-to-end rendering checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from chroma.geometry import Geometry, Solid
from chroma.make import box
from chroma.triton.viewer import Camera, DetectorViewer, _frame_dimensions, _geometry_groups


def test_camera_basis_and_validation():
    camera = Camera((0, 0, 3), up=(0, 1, 0))
    packed = camera.packed()
    np.testing.assert_array_equal(packed[3:6], [0, 0, -1])
    np.testing.assert_array_equal(packed[6:9], [1, 0, 0])
    np.testing.assert_array_equal(packed[9:12], [0, 1, 0])
    orbit = Camera.orbit((2, 3, 4), 5, azimuth=0, elevation=0)
    np.testing.assert_allclose(orbit.eye, [7, 3, 4])
    for camera in (
        Camera((0, 0, 0)),
        Camera((0, 0, 3)),
        Camera((np.nan, 0, 1)),
        Camera((0, 0, 3), up=(0, 1, 0), fov=180),
    ):
        with pytest.raises(ValueError):
            camera.packed()
    with pytest.raises(ValueError):
        Camera.orbit((0, 0, 0), 1, elevation=90)


def test_geometry_groups_keep_instances_and_colors_without_mutation():
    detector = Geometry()
    mesh = box(2, 2, 2)
    shared = Solid(mesh, None, None, color=0xFFCC9900)
    for displacement in ((1, 0, 0), (2, 0, 0)):
        detector.add_solid(shared, displacement=displacement)
    detector.add_solid(Solid(mesh, None, None, color=0xFF0000FF))
    groups = _geometry_groups(detector)
    assert len(groups) == 2
    assert groups[0].rotations.shape == (2, 3, 3)
    assert not hasattr(detector, "mesh")
    recolored = _geometry_groups(detector, solid_colors={i: 0xFFFF0000 for i in range(3)})
    assert len(recolored) == 1 and len(recolored[0].rotations) == 3
    assert np.all(shared.color == 0xFFCC9900)
    groups[0].vertices[:] = 0
    assert np.any(mesh.vertices != 0)
    assert len(_geometry_groups(detector, hidden_solids=(0, 1))) == 1
    with pytest.raises(ValueError, match="no visible"):
        _geometry_groups(detector, hidden_solids=(0, 1, 2))


def test_budget_dimensions_keep_every_pixel_sampled():
    for rays in (1, 2, 101, 100_000, 625_000, 2_500_003):
        width, height = _frame_dimensions(1000, 625, rays)
        assert 0 < width * height <= rays
        assert width <= 1000 and height <= 625
    for invalid in (True, 0, -1, 2.5):
        with pytest.raises(ValueError):
            _frame_dimensions(1000, 625, invalid)


def _require_gpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")


def test_widget_ray_count_controls_requested_budget(monkeypatch):
    pytest.importorskip("ipywidgets")
    from chroma.triton.viewer import RenderFrame
    import IPython.display

    monkeypatch.setattr(IPython.display, "display", lambda _: None)
    viewer = object.__new__(DetectorViewer)
    viewer.width, viewer.height, viewer.rays, viewer.radius = 8, 5, 2_500_000, 10.0
    viewer.camera = Camera.orbit((0, 0, 0), 5)
    viewer._widget = viewer._pending = viewer._detail_callback = None
    requested = []

    def fake_render(camera, *, rays):
        requested.append(rays)
        return RenderFrame(np.zeros((5, 8, 3), np.uint8), rays, 0.001, camera)

    viewer.render_frame = fake_render
    controls = viewer.show()
    assert requested[-1] == 2_500_000
    controls.children[0].children[3].value = 125_003
    assert viewer.rays == requested[-1] == 125_003
    controls.children[0].children[3].value = 17
    controls.children[0].children[0].value = 50
    assert requested[-1] == 17
    assert "17 camera rays" in controls.children[2].value
    viewer.close()


def test_gpu_camera_shading_matches_independent_pixel_reference():
    _require_gpu()
    mesh = SimpleNamespace(
        vertices=np.array([[-10, -10, 0], [10, -10, 0], [0, 10, 0]], np.float32),
        triangles=np.array([[0, 1, 2]], np.int32),
        colors=np.array([0xFFCC8844]),
    )
    viewer = DetectorViewer(mesh, width=32, height=20, rays=640)
    camera = Camera((0, 0, 2), up=(0, 1, 0), fov=60)
    frame = viewer.render_frame(camera, jitter=False)
    yy, xx = np.indices((20, 32))
    u = (2 * (xx + 0.5) / 32 - 1) * (32 / 20) * np.tan(np.pi / 6)
    v = (1 - 2 * (yy + 0.5) / 20) * np.tan(np.pi / 6)
    headlight = 0.22 + 0.78 / np.sqrt(1 + u * u + v * v)
    expected = np.floor(headlight[:, :, None] * np.array([204, 136, 68]) + 0.5).astype(np.uint8)
    np.testing.assert_allclose(frame.image, expected, atol=1)
    assert frame.rays == 640 and frame.seconds > 0
    assert frame.png().startswith(b"\x89PNG")
    # A budget not divisible by the image size still traces exactly that count.
    frame2 = viewer.render_frame(camera, rays=641, jitter=False)
    np.testing.assert_array_equal(frame2.image, frame.image)


def test_gpu_instance_render_agrees_with_flattened_geometry():
    _require_gpu()
    detector = Geometry()
    solid = Solid(box(1, 1, 1), None, None, color=0xFF80C080)
    for x, angle in ((-1, 0.2), (1, -0.4)):
        c, s = np.cos(angle), np.sin(angle)
        detector.add_solid(
            solid, displacement=(x, 0, 0), rotation=((c, -s, 0), (s, c, 0), (0, 0, 1))
        )
    camera = Camera((0, -4, 2), up=(0, 0, 1))
    instanced = DetectorViewer(detector, width=64, height=40, rays=2560, min_instances=2)
    flattened = DetectorViewer(detector, width=64, height=40, rays=2560, use_instances=False)
    first = instanced.render_frame(camera, jitter=False)
    second = flattened.render_frame(camera, jitter=False)
    np.testing.assert_allclose(first.image, second.image, atol=1)
    assert sum(group["instanced"] for group in instanced._groups) == 1
    assert not hasattr(detector, "mesh")


def test_analytic_wire_metadata_cannot_be_silently_omitted():
    geometry = Geometry()
    geometry.add_solid(Solid(box(2, 2, 2), None, None))
    geometry.wireplanes = [object()]
    with pytest.raises(ValueError, match="analytic_wires=False"):
        _geometry_groups(geometry)
