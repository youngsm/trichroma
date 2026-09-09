"""Interactive, GPU ray-traced geometry inspection in Jupyter.

Camera rays show opaque triangle colors under a headlight. They are not
scintillation/Cherenkov photons and do not run optical transport. Importing
this module needs only NumPy; CUDA and widgets are loaded when used.
"""

from dataclasses import dataclass
import hashlib
import io
import math
import time

import numpy as np


@dataclass(frozen=True)
class Camera:
    eye: tuple
    target: tuple = (0.0, 0.0, 0.0)
    up: tuple = (0.0, 0.0, 1.0)
    fov: float = 65.0

    def packed(self):
        eye, target, up = (np.asarray(v, dtype=float) for v in (self.eye, self.target, self.up))
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (eye, target, up)):
            raise ValueError("camera vectors must be finite three-vectors")
        forward = target - eye
        if np.linalg.norm(forward) == 0 or np.linalg.norm(up) == 0:
            raise ValueError("eye and target must differ and up must be nonzero")
        if not np.isfinite(self.fov) or not 0 < self.fov < 179:
            raise ValueError("fov must be between 0 and 179 degrees")
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-12 * np.linalg.norm(up):
            raise ValueError("up must not be parallel to the viewing direction")
        right /= np.linalg.norm(right)
        vertical = np.cross(right, forward)
        return np.asarray(
            [*eye, *forward, *right, *vertical, np.tan(np.deg2rad(self.fov) / 2)], np.float32
        )

    @classmethod
    def orbit(cls, target, distance, azimuth=35.0, elevation=15.0, fov=65.0):
        if not np.isfinite([distance, azimuth, elevation]).all() or distance <= 0:
            raise ValueError("orbit values must be finite and distance positive")
        if not -89.9 <= elevation <= 89.9:
            raise ValueError("elevation must lie in [-89.9, 89.9]")
        target = np.asarray(target, float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("target must be a finite three-vector")
        azimuth, elevation = np.deg2rad([azimuth, elevation])
        offset = distance * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )
        return cls(tuple(target + offset), tuple(target), fov=fov)


@dataclass(frozen=True)
class RenderFrame:
    image: np.ndarray
    rays: int
    seconds: float
    camera: Camera

    @property
    def rays_per_second(self):
        return self.rays / self.seconds

    @property
    def frames_per_second(self):
        return 1 / self.seconds

    def png(self, *, compress_level=1):
        from PIL import Image

        output = io.BytesIO()
        Image.fromarray(self.image).save(output, format="PNG", compress_level=compress_level)
        return output.getvalue()

    def _repr_png_(self):
        return self.png()


@dataclass(frozen=True)
class BoundaryRenderHit:
    """GPU boundary-layer output; color IDs are -1 on a miss.

    Layer providers implement validate_geometry(), prepare(device), and then
    reserve(count)/trace(origins, directions, tmax) on the prepared object.
    RGB colors use packed uint32 words, normals are normalized world vectors,
    and distances use the same ray parameter as the triangle query.
    """

    color_ids: object
    distances: object
    world_normals: object
    colors: object
    overflow: object = None


@dataclass(frozen=True)
class _MeshGroup:
    vertices: np.ndarray
    triangles: np.ndarray
    colors: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    bounds_rotation: object = None


def _colors(value, count):
    array = np.asarray(value, dtype=np.uint32)
    if array.ndim == 0:
        array = np.full(count, array, np.uint32)
    if array.shape != (count,):
        raise ValueError("triangle colors must be scalar or have shape (triangle_count,)")
    return array.copy()


def _geometry_groups(geometry, hidden_solids=(), solid_colors=None, handled_features=()):
    """Extract without flattening/mutating the caller's Geometry object."""
    if len(getattr(geometry, "wireplanes", ())) and "wireplanes" not in handled_features:
        raise ValueError(
            "Analytic wire metadata needs a matching boundary layer, or build with analytic_wires=False"
        )
    hidden = set(hidden_solids)
    solid_colors = {} if solid_colors is None else dict(solid_colors)
    if getattr(geometry, "solids", None):
        grouped = {}
        for index, (solid, rotation, translation) in enumerate(
            zip(geometry.solids, geometry.solid_rotations, geometry.solid_displacements)
        ):
            if index in hidden:
                continue
            colors = _colors(solid_colors.get(index, solid.color), len(solid.mesh.triangles))
            key = (id(solid.mesh), hashlib.sha256(colors.tobytes()).digest())
            if key not in grouped:
                grouped[key] = [solid.mesh, colors, [], []]
            grouped[key][2].append(rotation)
            grouped[key][3].append(translation)
        groups = [
            _MeshGroup(
                np.asarray(mesh.vertices, np.float32).copy(),
                np.asarray(mesh.triangles, np.int32).copy(),
                colors,
                np.asarray(rotations, np.float32),
                np.asarray(translations, np.float32),
            )
            for mesh, colors, rotations, translations in grouped.values()
        ]
    else:
        if hidden or solid_colors:
            raise ValueError("hidden_solids/solid_colors requires Geometry with solid records")
        mesh = getattr(geometry, "mesh", geometry)
        groups = [
            _MeshGroup(
                np.asarray(mesh.vertices, np.float32).copy(),
                np.asarray(mesh.triangles, np.int32).copy(),
                _colors(getattr(geometry, "colors", 0xFFA0ACBC), len(mesh.triangles)),
                np.eye(3, dtype=np.float32)[None],
                np.zeros((1, 3), np.float32),
            )
        ]
    if not groups:
        raise ValueError("no visible geometry")
    for group in groups:
        if not (
            np.isfinite(group.vertices).all()
            and np.isfinite(group.rotations).all()
            and np.isfinite(group.translations).all()
        ):
            raise ValueError("geometry and transforms must be finite")
    return groups


def _frame_dimensions(width, height, rays):
    if isinstance(rays, bool) or not isinstance(rays, (int, np.integer)) or rays <= 0:
        raise ValueError("rays must be a positive integer")
    if rays < width * height:
        scale = math.sqrt(rays / (width * height))
        return max(1, int(width * scale)), max(1, int(height * scale))
    return width, height


class DetectorViewer:
    """Reusable GPU renderer for Chroma Geometry/Solid meshes.

    Repeated mesh objects retain a single BLAS and per-instance transforms;
    this optional path uses the chroma_lar instance traversal service. A
    flattened mesh needs only chroma-lite. Enclosures and glass are opaque;
    start inside the detector or use ``hidden_solids=(0,)`` for a cutaway.
    Optional boundary_layers add exact non-mesh queries through BoundaryRenderHit.
    Instances are sequential resources; create one viewer per CUDA stream.
    """

    def __init__(
        self,
        geometry,
        *,
        width=1000,
        height=625,
        rays=2_500_000,
        device="cuda",
        hidden_solids=(),
        use_instances=True,
        solid_colors=None,
        min_instances=4,
        boundary_layers=(),
    ):
        import torch
        from .bvh import build_packed_bvh

        for name, value in (("width", width), ("height", height), ("min_instances", min_instances)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _frame_dimensions(width, height, rays)
        if not torch.cuda.is_available():
            raise RuntimeError("DetectorViewer requires a local CUDA GPU and Triton")
        self.device = torch.device(device)
        self.width, self.height, self.rays = width, height, int(rays)
        self._groups = []
        boundary_layers = tuple(boundary_layers)
        handled_features = set()
        for layer in boundary_layers:
            layer.validate_geometry(geometry)
            handled_features.update(layer.features)
        groups = _geometry_groups(geometry, hidden_solids, solid_colors, handled_features)
        self._boundary_layers = [layer.prepare(self.device) for layer in boundary_layers]
        bounds = []
        static_vertices, static_triangles, static_colors = [], [], []
        static_count = 0
        for group in groups:
            # Transform only eight mesh-box corners for aggregate camera bounds.
            lo, hi = group.vertices.min(0), group.vertices.max(0)
            corners = np.array(
                [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
            )
            world = np.einsum("nij,kj->nki", group.rotations, corners) + group.translations[:, None]
            bounds.extend((world.min((0, 1)), world.max((0, 1))))
            rigid = np.allclose(
                group.rotations @ group.rotations.transpose(0, 2, 1), np.eye(3), atol=2e-6, rtol=0
            )
            if use_instances and len(group.rotations) >= max(2, min_instances) and rigid:
                from chroma_lar.triton_scene.primitive_adapter import instance_traversal_view
                from chroma_lar.triton_scene.instances import build_pmt_instance_accelerator
                from .primitives import Mesh, MeshInstance

                mesh = Mesh(group.vertices, group.triangles)
                instances = tuple(
                    MeshInstance(str(i), mesh, rotation, translation, 0)
                    for i, (rotation, translation) in enumerate(
                        zip(group.rotations, group.translations)
                    )
                )
                accelerator = build_pmt_instance_accelerator(
                    instance_traversal_view(instances), device=self.device
                )
                self._groups.append(
                    dict(
                        instanced=True,
                        accelerator=accelerator,
                        colors=torch.as_tensor(group.colors.view(np.int32), device=self.device),
                        workspace=None,
                    )
                )
            else:
                for rotation, translation in zip(group.rotations, group.translations):
                    static_vertices.append(group.vertices @ rotation.T + translation)
                    static_triangles.append(group.triangles + static_count)
                    static_colors.append(group.colors)
                    static_count += len(group.vertices)
        if static_vertices:
            host = build_packed_bvh(
                np.concatenate(static_vertices), np.concatenate(static_triangles)
            )
            triangles = host.triangle_vertices
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
            self._groups.insert(
                0,
                dict(
                    instanced=False,
                    accelerator=host.to_triton(self.device),
                    normals=torch.as_tensor(normals, device=self.device),
                    colors=torch.as_tensor(
                        np.concatenate(static_colors).view(np.int32), device=self.device
                    ),
                    workspace=None,
                ),
            )
        bounds = np.asarray(bounds)
        self.center = (bounds.min(0) + bounds.max(0)) / 2
        self.radius = float(np.min(bounds.max(0) - bounds.min(0)) / 2)
        if self.radius <= 0:
            self.radius = max(float(np.max(bounds.max(0) - bounds.min(0)) / 2), 1.0)
        self.camera = Camera.orbit(self.center, self.radius * 0.65)
        self._capacity = 0
        self._widget = None
        self._pending = None
        self._detail_callback = None

    def _reserve(self, count):
        import torch

        if count <= self._capacity:
            return
        self._origins = torch.empty((count, 3), dtype=torch.float32, device=self.device)
        self._directions = torch.empty_like(self._origins)
        self._nearest = torch.empty(count, dtype=torch.float32, device=self.device)
        self._shade = torch.empty_like(self._origins)
        self._last_hit = torch.full((count,), -1, dtype=torch.int32, device=self.device)
        self._camera = torch.empty(13, dtype=torch.float32, device=self.device)
        for group in self._groups:
            accelerator = group["accelerator"]
            group["workspace"] = (
                accelerator.allocate_workspace(count, result_capacity=count)
                if group["instanced"]
                else accelerator.allocate_workspace(count)
            )
        for layer in self._boundary_layers:
            layer.reserve(count)
        self._capacity = count

    def render_frame(self, camera=None, *, rays=None, seed=0, jitter=True):
        """Return uint8 RGB plus wall time including tracing, shading and download.

        The first call includes allocation/JIT. Call once before measuring warm
        performance. PNG encoding, widget communication and browser painting
        are outside ``seconds``. Every frame traces exactly ``rays`` camera rays.
        """
        import torch
        import triton
        from .bvh_kernels import nearest_hit_local
        from .viewer_kernels import camera_rays, shade_hits, resolve_image

        count = self.rays if rays is None else rays
        width, height = _frame_dimensions(self.width, self.height, count)
        camera = self.camera if camera is None else camera
        packed = camera.packed()
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        self._reserve(count)
        self._camera.copy_(torch.from_numpy(packed))
        origin, direction = self._origins[:count], self._directions[:count]
        nearest, shade = self._nearest[:count], self._shade[:count]
        camera_rays[(triton.cdiv(count, 256),)](
            origin,
            direction,
            nearest,
            shade,
            self._camera,
            count,
            width,
            height,
            seed,
            jitter,
            BLOCK=256,
        )
        overflow = []
        for group in self._groups:
            if group["instanced"]:
                from chroma_lar.triton_scene.instances import nearest_pmt_hit

                result = nearest_pmt_hit(
                    group["accelerator"],
                    origin,
                    direction,
                    tmax=nearest,
                    last_instance=self._last_hit[:count],
                    last_triangle=self._last_hit[:count],
                    workspace=group["workspace"],
                    out=group["workspace"].outputs(count),
                    ray_tile=None,
                    use_tlas=True,
                    check_overflow=False,
                )
                normals = result.world_normals
            else:
                result = nearest_hit_local(
                    group["accelerator"],
                    origin,
                    direction,
                    tmax=nearest,
                    last_hit=self._last_hit[:count],
                    workspace=group["workspace"],
                )
                normals = group["normals"]
            overflow.append(result.overflow.any())
            shade_hits[(triton.cdiv(count, 256),)](
                result.triangle_ids,
                result.distances,
                normals,
                group["colors"],
                direction,
                nearest,
                shade,
                count,
                PER_RAY_NORMALS=group["instanced"],
                BLOCK=256,
            )
        for layer in self._boundary_layers:
            result = layer.trace(origin, direction, nearest)
            if result.overflow is not None:
                overflow.append(result.overflow.any())
            shade_hits[(triton.cdiv(count, 256),)](
                result.color_ids,
                result.distances,
                result.world_normals,
                result.colors,
                direction,
                nearest,
                shade,
                count,
                PER_RAY_NORMALS=True,
                BLOCK=256,
            )
        image = torch.empty((height, width, 3), dtype=torch.uint8, device=self.device)
        resolve_image[(triton.cdiv(width * height, 256),)](
            shade,
            image,
            count,
            width * height,
            SAMPLES=math.ceil(count / (width * height)),
            BLOCK=256,
        )
        host = image.cpu().numpy()
        if bool(torch.stack(overflow).any().item()):
            raise RuntimeError("geometry traversal overflow; image is incomplete")
        elapsed = time.perf_counter() - start
        self.camera = camera
        return RenderFrame(host, count, elapsed, camera)

    def show(self, *, preview_rays=100_000, settle_seconds=0.25):
        """Display orbit/zoom controls and render a full frame after motion settles."""
        import asyncio
        import ipywidgets as widgets
        from IPython.display import display

        _frame_dimensions(self.width, self.height, preview_rays)
        if not np.isfinite(settle_seconds) or settle_seconds < 0:
            raise ValueError("settle_seconds must be finite and nonnegative")
        if self._widget is not None:
            display(self._widget)
            return self._widget
        image = widgets.Image(format="png", layout=widgets.Layout(width=f"{self.width}px"))
        status = widgets.HTML()
        target = np.asarray(self.camera.target, float)
        offset = np.asarray(self.camera.eye, float) - target
        initial_distance = float(np.linalg.norm(offset) / self.radius)
        initial_azimuth = float(np.rad2deg(np.arctan2(offset[1], offset[0])))
        initial_elevation = float(np.rad2deg(np.arctan2(offset[2], np.linalg.norm(offset[:2]))))
        azimuth = widgets.FloatSlider(
            description="Orbit", min=-180, max=180, value=initial_azimuth, continuous_update=True
        )
        elevation = widgets.FloatSlider(
            description="Elevation",
            min=-89,
            max=89,
            value=initial_elevation,
            continuous_update=True,
        )
        distance = widgets.FloatLogSlider(
            description="Distance",
            min=min(-2, math.floor(math.log10(initial_distance))),
            max=max(0.5, math.ceil(math.log10(initial_distance))),
            value=initial_distance,
            base=10,
            continuous_update=True,
        )
        detail = widgets.Button(description=f"Render {self.rays/1e6:g}M rays")
        ray_count = widgets.BoundedIntText(
            value=self.rays,
            min=1,
            max=max(100_000_000, self.rays),
            step=100_000,
            description="Camera rays",
            layout=widgets.Layout(width="245px"),
            style={"description_width": "90px"},
        )
        controls = widgets.HBox(
            [azimuth, elevation, distance, ray_count, detail],
            layout=widgets.Layout(flex_flow="row wrap"),
        )

        def render(full):
            started = time.perf_counter()
            try:
                camera = Camera.orbit(
                    target, self.radius * distance.value, azimuth.value, elevation.value
                )
                frame = self.render_frame(
                    camera, rays=self.rays if full else min(preview_rays, self.rays)
                )
                image.value = frame.png()
                total = time.perf_counter() - started
                status.value = (
                    f"<b>{frame.rays:,} camera rays</b> · "
                    f"{frame.seconds*1000:.1f} ms render + download · "
                    f"{total*1000:.1f} ms including PNG · "
                    f"{frame.rays_per_second/1e6:.1f}M rays/s · opaque geometry view"
                )
            except Exception as error:
                import html

                status.value = "<b>Render failed:</b> " + html.escape(str(error))
                raise

        def update(change):
            if self._pending is not None:
                self._pending.cancel()
                self._pending = None
            render(False)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Environments without a running widget event loop can still
                # explicitly request full detail with the render button.
                return
            self._pending = loop.call_later(settle_seconds, render, True)

        for control in (azimuth, elevation, distance):
            control.observe(update, names="value")

        def change_ray_count(change):
            if self._pending is not None:
                self._pending.cancel()
                self._pending = None
            self.rays = int(change["new"])
            detail.description = f"Render {self.rays/1e6:g}M rays"
            render(True)

        ray_count.observe(change_ray_count, names="value")

        def on_detail(_):
            render(True)

        detail.on_click(on_detail)
        self._detail_callback = (detail, on_detail)
        self._widget = widgets.VBox([controls, image, status])
        display(self._widget)
        render(True)
        return self._widget

    def close(self):
        """Cancel pending renders and close notebook widgets."""
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        if self._detail_callback is not None:
            button, callback = self._detail_callback
            button.on_click(callback, remove=True)
            self._detail_callback = None
        if self._widget is not None:
            # Detach observers before closing children: widget registries and
            # callbacks must not retain old GPU viewers after detector switches.
            widgets = [self._widget]
            for widget in widgets:
                widgets.extend(getattr(widget, "children", ()))
            for widget in reversed(widgets):
                widget.unobserve_all()
                widget.close()
            self._widget = None
