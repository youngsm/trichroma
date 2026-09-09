"""Asynchronous production boundary geometry driven by device queue counts.

These primitives deliberately separate allocation capacity from live length.
They launch from a host-known upper bound while reading the one-word live
count on the GPU, allowing scheduler epochs to be enqueued without a host
``Tensor.item()``.  The strict Chroma replay path does not import or call this
module.

Dynamic queue counts are a producer/consumer contract: callers keep
``0 <= count <= input_capacity <= launch_capacity`` until the stream reaches
the final consumer.  Kernels defensively clamp the gather count to the input
capacity, and every wholly inactive CTA exits before doing substantive work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


_IMPORT_ERROR = None
try:  # Optional so CPU-only scene compilation remains importable.
    import torch
    import triton
    import triton.language as tl
except Exception as error:  # pragma: no cover - environment dependent
    torch = None
    triton = None
    tl = None
    _IMPORT_ERROR = error


@dataclass
class DeviceBoundaryRayWorkspace:
    """Capacity-owned dense ray storage populated from a sparse DeviceQueue."""

    origins: Any
    directions: Any
    last_instances: Any
    last_triangles: Any
    pmt_tmax: Any
    capacity: int

    @classmethod
    def allocate(cls, capacity: int, device: Any = "cuda"):
        _require_backend()
        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("ray workspace capacity cannot be negative")
        return cls(
            origins=torch.empty(
                (capacity, 3), dtype=torch.float32, device=device
            ),
            directions=torch.empty(
                (capacity, 3), dtype=torch.float32, device=device
            ),
            last_instances=torch.empty(
                capacity, dtype=torch.int32, device=device
            ),
            last_triangles=torch.empty(
                capacity, dtype=torch.int32, device=device
            ),
            pmt_tmax=torch.empty(
                capacity, dtype=torch.float32, device=device
            ),
            capacity=capacity,
        )

    def prefix(self, capacity: int):
        capacity = int(capacity)
        if capacity < 0 or capacity > self.capacity:
            raise ValueError("requested prefix exceeds ray workspace capacity")
        return DeviceBoundaryRayWorkspace(
            origins=self.origins[:capacity],
            directions=self.directions[:capacity],
            last_instances=self.last_instances[:capacity],
            last_triangles=self.last_triangles[:capacity],
            pmt_tmax=self.pmt_tmax[:capacity],
            capacity=capacity,
        )


@dataclass
class DeviceBoundaryMergeWorkspace:
    """Capacity-owned SoA outputs matching ``boundary_step`` argument order."""

    distance: Any
    normal: Any
    material_from: Any
    material_to: Any
    surface: Any
    instance: Any
    triangle: Any
    channel: Any
    capacity: int

    @classmethod
    def allocate(cls, capacity: int, device: Any = "cuda"):
        _require_backend()
        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("merge workspace capacity cannot be negative")
        return cls(
            distance=torch.empty(capacity, dtype=torch.float32, device=device),
            normal=torch.empty(
                (capacity, 3), dtype=torch.float32, device=device
            ),
            material_from=torch.empty(
                capacity, dtype=torch.int32, device=device
            ),
            material_to=torch.empty(
                capacity, dtype=torch.int32, device=device
            ),
            surface=torch.empty(capacity, dtype=torch.int32, device=device),
            instance=torch.empty(capacity, dtype=torch.int32, device=device),
            triangle=torch.empty(capacity, dtype=torch.int32, device=device),
            channel=torch.empty(capacity, dtype=torch.int32, device=device),
            capacity=capacity,
        )

    def outputs(self, capacity: Optional[int] = None):
        capacity = self.capacity if capacity is None else int(capacity)
        if capacity < 0 or capacity > self.capacity:
            raise ValueError("requested outputs exceed merge workspace capacity")
        return (
            self.distance[:capacity],
            self.normal[:capacity],
            self.material_from[:capacity],
            self.material_to[:capacity],
            self.surface[:capacity],
            self.instance[:capacity],
            self.triangle[:capacity],
            self.channel[:capacity],
        )


def _require_backend() -> None:
    if torch is None or triton is None:
        detail = "" if _IMPORT_ERROR is None else f": {_IMPORT_ERROR}"
        raise RuntimeError("PyTorch and Triton are required" + detail)
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device visible to PyTorch is required")


def _validate_count(count: Any, device: Any) -> None:
    if (
        not isinstance(count, torch.Tensor)
        or count.shape != (1,)
        or count.dtype != torch.int32
        or count.device != device
        or not count.is_contiguous()
    ):
        raise ValueError(
            "queue count must be contiguous CUDA int32 [1] on the data device"
        )


if triton is not None and torch is not None:

    @triton.jit(do_not_specialize=[10])
    def _gather_boundary_rays_device_count_kernel(
        positions,
        directions,
        last_instances,
        last_triangles,
        queue,
        queue_count,
        out_origins,
        out_directions,
        out_last_instances,
        out_last_triangles,
        input_capacity,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_start = tl.program_id(0) * BLOCK_SIZE
        live_count = tl.load(queue_count).to(tl.int32)
        # Clamp only for memory safety.  A valid producer never exercises it;
        # the scheduler contract remains count <= input_capacity.
        live_count = tl.maximum(0, tl.minimum(live_count, input_capacity))
        if program_start >= live_count:
            return
        lane = program_start + tl.arange(0, BLOCK_SIZE)
        valid = lane < live_count
        photon_id = tl.load(queue + lane, mask=valid, other=0).to(tl.int64)
        source = photon_id * 3
        target = lane * 3
        for axis in tl.static_range(3):
            tl.store(
                out_origins + target + axis,
                tl.load(positions + source + axis, mask=valid, other=0.0),
                mask=valid,
            )
            tl.store(
                out_directions + target + axis,
                tl.load(directions + source + axis, mask=valid, other=0.0),
                mask=valid,
            )
        tl.store(
            out_last_instances + lane,
            tl.load(last_instances + photon_id, mask=valid, other=-1),
            mask=valid,
        )
        tl.store(
            out_last_triangles + lane,
            tl.load(last_triangles + photon_id, mask=valid, other=-1),
            mask=valid,
        )

    @triton.jit(do_not_specialize=[3])
    def _nextafter_positive_inf_device_count_kernel(
        values,
        outputs,
        active_count,
        launch_capacity,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_start = tl.program_id(0) * BLOCK_SIZE
        live_count = tl.load(active_count).to(tl.int32)
        live_count = tl.maximum(0, tl.minimum(live_count, launch_capacity))
        if program_start >= live_count:
            return
        lane = program_start + tl.arange(0, BLOCK_SIZE)
        valid = lane < live_count
        value = tl.load(values + lane, mask=valid, other=0.0).to(tl.float32)
        bits = value.to(tl.uint32, bitcast=True)
        absolute = bits & 0x7FFFFFFF
        is_nan = absolute > 0x7F800000
        is_positive_inf = bits == 0x7F800000
        is_zero = absolute == 0
        incremented = bits + 1
        decremented = bits - 1
        next_bits = tl.where(
            is_zero,
            1,
            tl.where(value > 0.0, incremented, decremented),
        )
        next_bits = tl.where(is_nan | is_positive_inf, bits, next_bits)
        tl.store(
            outputs + lane,
            next_bits.to(tl.float32, bitcast=True),
            mask=valid,
        )

    @triton.jit(do_not_specialize=[26])
    def _merge_boundaries_device_count_kernel(
        analytic_distance,
        analytic_kind,
        analytic_index,
        analytic_primitive,
        analytic_normal,
        analytic_material_from,
        analytic_material_to,
        analytic_surface,
        pmt_distance,
        pmt_normal,
        pmt_instance,
        pmt_triangle,
        pmt_channel,
        directions,
        triangle_material1,
        triangle_material2,
        triangle_surface,
        out_distance,
        out_normal,
        out_material_from,
        out_material_to,
        out_surface,
        out_instance,
        out_triangle,
        out_channel,
        active_count,
        launch_capacity,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_start = tl.program_id(0) * BLOCK_SIZE
        live_count = tl.load(active_count).to(tl.int32)
        live_count = tl.maximum(0, tl.minimum(live_count, launch_capacity))
        if program_start >= live_count:
            return
        lane = program_start + tl.arange(0, BLOCK_SIZE)
        valid = lane < live_count

        analytic_d = tl.load(
            analytic_distance + lane, mask=valid, other=float("inf")
        )
        analytic_kind_id = tl.load(
            analytic_kind + lane, mask=valid, other=0
        ).to(tl.int32)
        analytic_hit = analytic_kind_id != 0
        analytic_index_id = tl.load(
            analytic_index + lane, mask=valid, other=-1
        ).to(tl.int32)
        analytic_primitive_id = tl.load(
            analytic_primitive + lane, mask=valid, other=-1
        ).to(tl.int32)
        pmt_d = tl.load(
            pmt_distance + lane, mask=valid, other=float("inf")
        )
        pmt_instance_id = tl.load(
            pmt_instance + lane, mask=valid, other=-1
        ).to(tl.int32)
        pmt_triangle_id = tl.load(
            pmt_triangle + lane, mask=valid, other=-1
        ).to(tl.int32)
        pmt_hit = pmt_instance_id >= 0

        analytic_strictly_closer = (
            analytic_d.to(tl.float64) + 1.0e-12 < pmt_d.to(tl.float64)
        )
        choose_analytic = analytic_hit & (
            ~pmt_hit | analytic_strictly_closer
        )
        choose_pmt = pmt_hit & ~choose_analytic
        no_hit = ~(choose_analytic | choose_pmt)

        safe_triangle = tl.maximum(pmt_triangle_id, 0)
        pmt_m1 = tl.load(
            triangle_material1 + safe_triangle, mask=valid, other=-1
        ).to(tl.int32)
        pmt_m2 = tl.load(
            triangle_material2 + safe_triangle, mask=valid, other=-1
        ).to(tl.int32)
        pmt_surface_id = tl.load(
            triangle_surface + safe_triangle, mask=valid, other=-1
        ).to(tl.int32)

        pnx = tl.load(pmt_normal + lane * 3, mask=valid, other=0.0)
        pny = tl.load(pmt_normal + lane * 3 + 1, mask=valid, other=0.0)
        pnz = tl.load(pmt_normal + lane * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + lane * 3, mask=valid, other=0.0)
        dy = tl.load(directions + lane * 3 + 1, mask=valid, other=0.0)
        dz = tl.load(directions + lane * 3 + 2, mask=valid, other=0.0)
        raw_dot = pnx * (-dx) + pny * (-dy) + pnz * (-dz)
        normal_faces_ray = raw_dot > 0.0
        oriented_x = tl.where(normal_faces_ray, pnx, -pnx)
        oriented_y = tl.where(normal_faces_ray, pny, -pny)
        oriented_z = tl.where(normal_faces_ray, pnz, -pnz)

        anx = tl.load(analytic_normal + lane * 3, mask=valid, other=0.0)
        any_ = tl.load(
            analytic_normal + lane * 3 + 1, mask=valid, other=0.0
        )
        anz = tl.load(
            analytic_normal + lane * 3 + 2, mask=valid, other=0.0
        )
        nx = tl.where(choose_analytic, anx, oriented_x)
        ny = tl.where(choose_analytic, any_, oriented_y)
        nz = tl.where(choose_analytic, anz, oriented_z)

        pmt_from = tl.where(normal_faces_ray, pmt_m2, pmt_m1)
        pmt_to = tl.where(normal_faces_ray, pmt_m1, pmt_m2)
        analytic_from = tl.load(
            analytic_material_from + lane, mask=valid, other=-1
        ).to(tl.int32)
        analytic_to = tl.load(
            analytic_material_to + lane, mask=valid, other=-1
        ).to(tl.int32)
        analytic_surface_id = tl.load(
            analytic_surface + lane, mask=valid, other=-1
        ).to(tl.int32)
        material_from = tl.where(choose_analytic, analytic_from, pmt_from)
        material_to = tl.where(choose_analytic, analytic_to, pmt_to)
        surface_id = tl.where(
            choose_analytic, analytic_surface_id, pmt_surface_id
        )
        material_from = tl.where(no_hit, -1, material_from)
        material_to = tl.where(no_hit, -1, material_to)
        surface_id = tl.where(no_hit, -1, surface_id)

        tl.store(
            out_distance + lane,
            tl.where(choose_analytic, analytic_d, pmt_d),
            mask=valid,
        )
        tl.store(out_normal + lane * 3, nx, mask=valid)
        tl.store(out_normal + lane * 3 + 1, ny, mask=valid)
        tl.store(out_normal + lane * 3 + 2, nz, mask=valid)
        tl.store(out_material_from + lane, material_from, mask=valid)
        tl.store(out_material_to + lane, material_to, mask=valid)
        tl.store(out_surface + lane, surface_id, mask=valid)
        analytic_box = choose_analytic & (analytic_kind_id == 1)
        encoded_box_instance = -analytic_index_id - 2
        tl.store(
            out_instance + lane,
            tl.where(
                choose_pmt,
                pmt_instance_id,
                tl.where(analytic_box, encoded_box_instance, -1),
            ),
            mask=valid,
        )
        tl.store(
            out_triangle + lane,
            tl.where(
                choose_pmt,
                pmt_triangle_id,
                tl.where(analytic_box, analytic_primitive_id, -1),
            ),
            mask=valid,
        )
        pmt_channel_id = tl.load(
            pmt_channel + lane, mask=valid, other=-1
        ).to(tl.int32)
        tl.store(
            out_channel + lane,
            tl.where(choose_pmt, pmt_channel_id, -1),
            mask=valid,
        )

else:  # pragma: no cover - clearer errors in CPU-only environments
    _gather_boundary_rays_device_count_kernel = None
    _nextafter_positive_inf_device_count_kernel = None
    _merge_boundaries_device_count_kernel = None


def _validate_state_rays(
    positions: Any,
    directions: Any,
    last_instances: Any,
    last_triangles: Any,
) -> Any:
    if (
        not isinstance(positions, torch.Tensor)
        or positions.ndim != 2
        or positions.shape[1] != 3
        or positions.dtype != torch.float32
        or not positions.is_cuda
        or not positions.is_contiguous()
    ):
        raise ValueError(
            "positions must be contiguous CUDA float32 [state_capacity,3]"
        )
    if (
        not isinstance(directions, torch.Tensor)
        or directions.shape != positions.shape
        or directions.dtype != torch.float32
        or directions.device != positions.device
        or not directions.is_contiguous()
    ):
        raise ValueError("directions must match positions")
    for values, name in (
        (last_instances, "last_instances"),
        (last_triangles, "last_triangles"),
    ):
        if (
            not isinstance(values, torch.Tensor)
            or values.shape != (positions.shape[0],)
            or values.dtype != torch.int32
            or values.device != positions.device
            or not values.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA int32 [state_capacity]"
            )
    return positions.device


def gather_boundary_rays_device_count(
    positions: Any,
    directions: Any,
    last_instances: Any,
    last_triangles: Any,
    queue: Any,
    queue_count: Optional[Any] = None,
    *,
    input_capacity: Optional[int] = None,
    launch_capacity: Optional[int] = None,
    out: Optional[DeviceBoundaryRayWorkspace] = None,
    block_size: int = 256,
) -> DeviceBoundaryRayWorkspace:
    """Gather a sparse photon queue into dense capacity-sized ray arrays.

    ``queue`` may be a :class:`chroma.triton.transport.DeviceQueue` or its raw
    CUDA index buffer accompanied by ``queue_count``.  A larger backing queue
    can be capped to the scheduler's current batch with ``input_capacity``;
    ``launch_capacity`` controls the grid and output prefix.
    """

    _require_backend()
    device = _validate_state_rays(
        positions, directions, last_instances, last_triangles
    )
    if queue_count is None and hasattr(queue, "buffer") and hasattr(queue, "count"):
        queue_buffer = queue.buffer
        queue_count = queue.count
    else:
        queue_buffer = queue
    if (
        not isinstance(queue_buffer, torch.Tensor)
        or not queue_buffer.is_cuda
        or queue_buffer.device != device
        or queue_buffer.ndim != 1
        or queue_buffer.dtype not in (torch.int32, torch.int64)
        or not queue_buffer.is_contiguous()
    ):
        raise ValueError("queue must be a contiguous CUDA int32/int64 vector")
    _validate_count(queue_count, device)
    physical_capacity = int(queue_buffer.numel())
    input_capacity = (
        physical_capacity if input_capacity is None else int(input_capacity)
    )
    launch_capacity = (
        input_capacity if launch_capacity is None else int(launch_capacity)
    )
    if input_capacity < 0 or input_capacity > physical_capacity:
        raise ValueError(
            "input_capacity must fit within the queue backing allocation"
        )
    if launch_capacity < input_capacity:
        raise ValueError(
            "launch_capacity must be at least input_capacity to avoid truncation"
        )
    if block_size not in (64, 128, 256, 512):
        raise ValueError("block_size must be 64, 128, 256, or 512")
    if out is None:
        out = DeviceBoundaryRayWorkspace.allocate(launch_capacity, device)
    if not isinstance(out, DeviceBoundaryRayWorkspace):
        raise TypeError("out must be a DeviceBoundaryRayWorkspace")
    if out.capacity < launch_capacity:
        raise ValueError("ray workspace is smaller than launch_capacity")
    result = out.prefix(launch_capacity)
    for values, shape, dtype, name in (
        (result.origins, (launch_capacity, 3), torch.float32, "origins"),
        (result.directions, (launch_capacity, 3), torch.float32, "directions"),
        (
            result.last_instances,
            (launch_capacity,),
            torch.int32,
            "last_instances",
        ),
        (
            result.last_triangles,
            (launch_capacity,),
            torch.int32,
            "last_triangles",
        ),
        (
            result.pmt_tmax,
            (launch_capacity,),
            torch.float32,
            "pmt_tmax",
        ),
    ):
        if (
            values.shape != shape
            or values.dtype != dtype
            or values.device != device
            or not values.is_contiguous()
        ):
            raise ValueError(f"out.{name} has incompatible storage")
    if launch_capacity:
        _gather_boundary_rays_device_count_kernel[
            (triton.cdiv(launch_capacity, block_size),)
        ](
            positions,
            directions,
            last_instances,
            last_triangles,
            queue_buffer,
            queue_count,
            result.origins,
            result.directions,
            result.last_instances,
            result.last_triangles,
            input_capacity,
            BLOCK_SIZE=block_size,
            num_warps=min(8, max(1, block_size // 32)),
        )
    return result


def nextafter_positive_inf_device_count(
    values: Any,
    active_count: Any,
    *,
    launch_capacity: Optional[int] = None,
    out: Optional[Any] = None,
    block_size: int = 256,
) -> Any:
    """Apply exact float32 ``nextafter(value, +inf)`` to the live prefix."""

    _require_backend()
    if (
        not isinstance(values, torch.Tensor)
        or values.ndim != 1
        or values.dtype != torch.float32
        or not values.is_cuda
        or not values.is_contiguous()
    ):
        raise ValueError("values must be a contiguous CUDA float32 vector")
    capacity = values.numel() if launch_capacity is None else int(launch_capacity)
    if capacity < 0 or capacity > values.numel():
        raise ValueError("launch_capacity must fit within values")
    _validate_count(active_count, values.device)
    if out is None:
        out = torch.empty(capacity, dtype=torch.float32, device=values.device)
    if (
        not isinstance(out, torch.Tensor)
        or out.shape != (capacity,)
        or out.dtype != torch.float32
        or out.device != values.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be contiguous CUDA float32 [launch_capacity]"
        )
    if capacity:
        _nextafter_positive_inf_device_count_kernel[
            (triton.cdiv(capacity, block_size),)
        ](
            values,
            out,
            active_count,
            capacity,
            BLOCK_SIZE=block_size,
            num_warps=min(8, max(1, block_size // 32)),
        )
    return out


def _validate_prefix(
    values: Any,
    capacity: int,
    device: Any,
    name: str,
    dtype: Any,
    tail_shape: tuple[int, ...] = (),
) -> Any:
    if not isinstance(values, torch.Tensor) or values.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on the merge device")
    if (
        values.ndim != 1 + len(tail_shape)
        or tuple(values.shape[1:]) != tail_shape
        or values.shape[0] < capacity
        or values.dtype != dtype
        or not values.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous {dtype} [N{''.join(',' + str(v) for v in tail_shape)}] "
            "with at least launch_capacity entries"
        )
    return values[:capacity]


def merge_boundaries_device_count(
    scene_device: Mapping[str, Any],
    analytic: Any,
    pmt: Any,
    directions: Any,
    active_count: Any,
    *,
    launch_capacity: Optional[int] = None,
    out: Optional[DeviceBoundaryMergeWorkspace] = None,
    block_size: int = 256,
):
    """Merge analytic and PMT winners without synchronizing the live count.

    The returned eight-tensor tuple is capacity-sized and follows the exact
    positional contract consumed by the production boundary physics kernel.
    Mesh-on-tie and PMT material-side conventions match the synchronized merge.
    """

    _require_backend()
    if (
        not isinstance(directions, torch.Tensor)
        or directions.ndim != 2
        or directions.shape[1] != 3
        or directions.dtype != torch.float32
        or not directions.is_cuda
        or not directions.is_contiguous()
    ):
        raise ValueError("directions must be contiguous CUDA float32 [N,3]")
    storage_capacity = int(directions.shape[0])
    capacity = (
        storage_capacity if launch_capacity is None else int(launch_capacity)
    )
    if capacity < 0 or capacity > storage_capacity:
        raise ValueError("launch_capacity must fit within direction storage")
    device = directions.device
    _validate_count(active_count, device)
    if block_size not in (64, 128, 256, 512):
        raise ValueError("block_size must be 64, 128, 256, or 512")

    analytic_distance = _validate_prefix(
        analytic.distance, capacity, device, "analytic.distance", torch.float32
    )
    analytic_kind = _validate_prefix(
        analytic.kind, capacity, device, "analytic.kind", torch.int8
    )
    analytic_index = _validate_prefix(
        analytic.index, capacity, device, "analytic.index", torch.int32
    )
    analytic_primitive = _validate_prefix(
        analytic.primitive_index,
        capacity,
        device,
        "analytic.primitive_index",
        torch.int32,
    )
    analytic_normal = _validate_prefix(
        analytic.surface_normal,
        capacity,
        device,
        "analytic.surface_normal",
        torch.float32,
        (3,),
    )
    analytic_material_from = _validate_prefix(
        analytic.material_from_index,
        capacity,
        device,
        "analytic.material_from_index",
        torch.int32,
    )
    analytic_material_to = _validate_prefix(
        analytic.material_to_index,
        capacity,
        device,
        "analytic.material_to_index",
        torch.int32,
    )
    analytic_surface = _validate_prefix(
        analytic.surface_index,
        capacity,
        device,
        "analytic.surface_index",
        torch.int32,
    )
    pmt_distance = _validate_prefix(
        pmt.distances, capacity, device, "pmt.distances", torch.float32
    )
    pmt_normal = _validate_prefix(
        pmt.world_normals,
        capacity,
        device,
        "pmt.world_normals",
        torch.float32,
        (3,),
    )
    pmt_instance = _validate_prefix(
        pmt.instance_ids, capacity, device, "pmt.instance_ids", torch.int32
    )
    pmt_triangle = _validate_prefix(
        pmt.triangle_ids, capacity, device, "pmt.triangle_ids", torch.int32
    )
    pmt_channel = _validate_prefix(
        pmt.channel_ids, capacity, device, "pmt.channel_ids", torch.int32
    )

    for name in (
        "pmt_scene_material1_index",
        "pmt_scene_material2_index",
        "pmt_scene_surface_index",
    ):
        if name not in scene_device:
            raise KeyError(f"scene_device is missing {name}")
        table = scene_device[name]
        if (
            not isinstance(table, torch.Tensor)
            or table.device != device
            or table.ndim != 1
            or table.dtype != torch.int32
            or not table.is_contiguous()
        ):
            raise ValueError(f"scene_device[{name!r}] has incompatible storage")

    if out is None:
        out = DeviceBoundaryMergeWorkspace.allocate(capacity, device)
    if not isinstance(out, DeviceBoundaryMergeWorkspace):
        raise TypeError("out must be a DeviceBoundaryMergeWorkspace")
    if out.capacity < capacity:
        raise ValueError("merge workspace is smaller than launch_capacity")
    outputs = out.outputs(capacity)
    if capacity:
        _merge_boundaries_device_count_kernel[
            (triton.cdiv(capacity, block_size),)
        ](
            analytic_distance,
            analytic_kind,
            analytic_index,
            analytic_primitive,
            analytic_normal,
            analytic_material_from,
            analytic_material_to,
            analytic_surface,
            pmt_distance,
            pmt_normal,
            pmt_instance,
            pmt_triangle,
            pmt_channel,
            directions[:capacity],
            scene_device["pmt_scene_material1_index"],
            scene_device["pmt_scene_material2_index"],
            scene_device["pmt_scene_surface_index"],
            *outputs,
            active_count,
            capacity,
            BLOCK_SIZE=block_size,
            num_warps=min(8, max(1, block_size // 32)),
        )
    return outputs


__all__ = [
    "DeviceBoundaryMergeWorkspace",
    "DeviceBoundaryRayWorkspace",
    "gather_boundary_rays_device_count",
    "merge_boundaries_device_count",
    "nextafter_positive_inf_device_count",
]
