"""Detector-specialized Triton simulation for ``reflect_reflect3wires``.

The backend is intentionally narrow: it compiles the exact detector named by
``detector_config_reflect_reflect3wires.py`` at its production wavelength of
450 nm.  Narrow specialization is what makes the following organization
possible without changing the simulated probability law:

* homogeneous LAr histories execute several collisions per Triton program;
* only photons at a certified empty-region exit enter the boundary queue;
* boxes and periodic wires are analytic, while all 81 reachable PMTs share one
  local triangle BVH;
* counter-based Philox streams are keyed by global photon ID, independent of
  tiling and queue order;
* terminal detections are compacted directly to flat ``(time, channel)`` hits.

The general PyCUDA Chroma path remains untouched.  Unsupported wavelengths,
weighted transport, re-emission, or other detector configurations fail closed
instead of silently changing physics.
"""

from __future__ import annotations

from dataclasses import dataclass
import time as _time
from typing import Any, Optional, Sequence

import numpy as np

from .triton_scene import compile_reflect3wires_scene


# Boundary-step status written per input queue element.
STEP_ACTIVE = 0
STEP_TERMINAL = 1

# Stable process identifiers written by the opt-in boundary random-tape trace.
# These values are deliberately independent of the internal queue/status enums
# so CUDA/Triton lockstep tools can compare one compact, photon-keyed record.
TAPE_PROCESS_UNSET = 0
TAPE_PROCESS_BULK_ABSORB = 1
TAPE_PROCESS_BULK_SCATTER = 2
TAPE_PROCESS_SURFACE_ABSORB = 3
TAPE_PROCESS_SURFACE_DETECT = 4
TAPE_PROCESS_SURFACE_DIFFUSE = 5
TAPE_PROCESS_SURFACE_SPECULAR = 6
TAPE_PROCESS_DIELECTRIC_REFLECT = 7
TAPE_PROCESS_DIELECTRIC_TRANSMIT = 8

# Dense debug state ledger layout shared with Chroma's tape kernel:
# position xyz, direction xyz, polarization xyz, wavelength, time, history,
# last_hit_triangle, weight, event index.  Every entry is one uninterpreted
# uint32 word.
STATE_CERTIFICATE_WIDTH = 15

# Conservative memory model for one detector-specialized resident wavefront.
# The main terms, in bytes per source photon, are:
#
# * 68 for the ten state arrays;
# * 36 for source/pending/survivor and collision queue storage;
# * 187 for a worst-case full boundary queue's analytic, gathered-ray, merge,
#   PMT prefix, and result arrays;
# * up to roughly 150 for sparse PMT candidate pairs and their BVH stacks.
#
# Rounding this subtotal up to 512 also covers temporary source generation,
# concatenation, allocator granularity, and modest implementation growth.  The
# planner spends only 70% of allocatable memory and withholds another 512 MiB,
# so this estimate is intentionally a capacity bound rather than a claim about
# typical residency.  Unlike the old 15M constant, it scales across GPUs and
# coexists with other processes using the same device.
AUTO_TILE_BYTES_PER_PHOTON = 512
AUTO_TILE_USABLE_FRACTION = 0.70
AUTO_TILE_RESERVE_BYTES = 512 * 1024 * 1024
AUTO_TILE_ALIGNMENT = 256


@dataclass(frozen=True)
class SimulationTilePlan:
    """Inspectable memory decision for one simulation call.

    ``available_bytes`` is ``None`` only for an explicit ``tile_size``.  The
    estimated peak covers the newly planned wavefront, while allocations that
    already exist on the device have already reduced ``available_bytes``.
    """

    total_photons: int
    tile_capacity: int
    tile_count: int
    available_bytes: Optional[int]
    usable_bytes: Optional[int]
    bytes_per_photon: int
    estimated_peak_bytes: int
    explicit: bool


def _simulation_tile_plan(
    nphotons,
    configured_tile_size=None,
    *,
    available_memory_bytes=None,
    bytes_per_photon=AUTO_TILE_BYTES_PER_PHOTON,
    usable_fraction=AUTO_TILE_USABLE_FRACTION,
    reserve_bytes=AUTO_TILE_RESERVE_BYTES,
    alignment=AUTO_TILE_ALIGNMENT,
):
    """Plan a source slab from an explicit cap or allocatable CUDA memory.

    All memory inputs are ordinary integers, keeping the policy independently
    testable on a CPU-only host.  CUDA discovery is deliberately handled by
    :func:`_cuda_available_memory_bytes` at the call site.
    """

    nphotons = int(nphotons)
    if nphotons < 0:
        raise ValueError("nphotons must be non-negative")
    bytes_per_photon = int(bytes_per_photon)
    reserve_bytes = int(reserve_bytes)
    alignment = int(alignment)
    if bytes_per_photon <= 0:
        raise ValueError("bytes_per_photon must be positive")
    if not (0.0 < float(usable_fraction) <= 1.0):
        raise ValueError("usable_fraction must lie in (0, 1]")
    if reserve_bytes < 0:
        raise ValueError("reserve_bytes must be non-negative")
    if alignment <= 0:
        raise ValueError("alignment must be positive")

    if configured_tile_size is not None:
        capacity = int(configured_tile_size)
        if capacity <= 0:
            raise ValueError("tile_size must be positive or None")
        capacity = max(1, min(nphotons, capacity))
        tile_count = (
            0 if nphotons == 0 else (nphotons + capacity - 1) // capacity
        )
        return SimulationTilePlan(
            total_photons=nphotons,
            tile_capacity=capacity,
            tile_count=tile_count,
            available_bytes=None,
            usable_bytes=None,
            bytes_per_photon=bytes_per_photon,
            estimated_peak_bytes=(0 if nphotons == 0 else capacity * bytes_per_photon),
            explicit=True,
        )

    if available_memory_bytes is None:
        raise ValueError(
            "available_memory_bytes is required when tile_size is None"
        )
    available_memory_bytes = int(available_memory_bytes)
    if available_memory_bytes < 0:
        raise ValueError("available_memory_bytes must be non-negative")
    budget = int(available_memory_bytes * float(usable_fraction))
    usable_bytes = max(0, budget - reserve_bytes)
    if nphotons == 0:
        return SimulationTilePlan(
            total_photons=0,
            tile_capacity=1,
            tile_count=0,
            available_bytes=available_memory_bytes,
            usable_bytes=usable_bytes,
            bytes_per_photon=bytes_per_photon,
            estimated_peak_bytes=0,
            explicit=False,
        )

    raw_capacity = usable_bytes // bytes_per_photon
    if raw_capacity <= 0:
        required = int(
            np.ceil((reserve_bytes + bytes_per_photon) / float(usable_fraction))
        )
        raise MemoryError(
            "automatic Triton tile planning cannot fit one photon: "
            f"{available_memory_bytes} bytes are available but at least "
            f"{required} bytes are required by the configured safety policy"
        )
    if raw_capacity >= nphotons:
        capacity = nphotons
    elif raw_capacity >= alignment:
        capacity = (raw_capacity // alignment) * alignment
    else:
        capacity = raw_capacity
    tile_count = (nphotons + capacity - 1) // capacity
    return SimulationTilePlan(
        total_photons=nphotons,
        tile_capacity=capacity,
        tile_count=tile_count,
        available_bytes=available_memory_bytes,
        usable_bytes=usable_bytes,
        bytes_per_photon=bytes_per_photon,
        estimated_peak_bytes=capacity * bytes_per_photon,
        explicit=False,
    )


def _simulation_tile_capacity(
    nphotons, configured_tile_size=None, *, available_memory_bytes=None, **kwargs
):
    """Compatibility wrapper returning only the planned positive capacity."""

    return _simulation_tile_plan(
        nphotons,
        configured_tile_size,
        available_memory_bytes=available_memory_bytes,
        **kwargs,
    ).tile_capacity


def _cuda_available_memory_bytes(torch, device):
    """Return memory available to new Torch allocations on ``device``.

    CUDA's free-memory counter excludes blocks cached by Torch even though the
    allocator can immediately reuse them.  Adding reserved-but-unallocated
    bytes avoids progressively shrinking auto tiles after repeated calls while
    still accounting for live tensors and memory owned by other processes.
    """

    driver_free, _ = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    return int(driver_free) + max(0, int(reserved) - int(allocated))


def _signed_seed32(seed, domain=0):
    """Return one stable signed-32-bit Philox key for every Python integer.

    Triton specializes kernels on scalar argument type.  Without this
    canonicalization, seeds on opposite sides of ``2**31`` trigger a second
    multi-second JIT compilation inside an otherwise warmed benchmark.
    """

    value = (int(seed) + int(domain)) & 0xFFFFFFFF
    return value if value < 0x80000000 else value - 0x100000000


@dataclass(frozen=True)
class FlatHits:
    """Detected photon arrival times and stable global PMT channel IDs."""

    time: Any
    channel: Any

    def __len__(self) -> int:
        return int(self.time.numel())

    def to_numpy(self):
        return (
            self.time.detach().cpu().numpy(),
            self.channel.detach().cpu().numpy(),
        )


@dataclass(frozen=True)
class SimulationStats:
    photons: int
    detections: int
    tiles: int
    boundary_rounds: int
    boundary_events: int
    elapsed_seconds: float
    dense_rounds: int = 0
    reservoir_rounds: int = 0
    reservoir_photons: int = 0

    @property
    def photons_per_second(self) -> float:
        if self.elapsed_seconds <= 0.0:
            return float("inf")
        return self.photons / self.elapsed_seconds


@dataclass(frozen=True)
class TritonSimulationResult:
    flat_hits: FlatHits
    stats: SimulationStats
    final_states: Optional[tuple[Any, ...]] = None
    tape_audit: Optional[Any] = None
    boundary_tape_trace: Optional[Any] = None
    tape_certificate: Optional[Any] = None
    state_certificate: Optional[Any] = None


@dataclass
class BoundaryTapeTrace:
    """Last boundary-tape interaction observed for every stable tape row.

    ``interaction`` is the pre-commit interaction index and ``draw_count`` is
    the number of semantic draws requested by that interaction.  Keeping the
    pre-commit values makes a first divergence reconstructible even though the
    shared audit cursor resets its draw component to zero after a successful
    interaction.  A lockstep harness can recover the raw word as
    ``tape.values[row, interaction, draw_slot].view(uint32)``.
    Dielectric decisions with draw counts four and five respectively identify
    a no-surface transition and a default-surface PASS transition.
    ``work_overflow`` mirrors sticky audit bits by expected global-ID row, so
    even an invalid tape-row index remains observable when no audit row exists.

    The trace is intentionally only the most recent boundary interaction.  A
    stepwise validator snapshots it after each scheduler round; production
    simulation does not allocate or write it.
    """

    interaction: Any
    draw_count: Any
    decision: Any
    work_overflow: Any

    @classmethod
    def allocate(cls, photon_count, device="cuda"):
        import torch

        photon_count = int(photon_count)
        if photon_count < 0:
            raise ValueError("photon_count cannot be negative")
        return cls(
            interaction=torch.full(
                (photon_count,), -1, dtype=torch.int32, device=device
            ),
            draw_count=torch.zeros(
                photon_count, dtype=torch.int32, device=device
            ),
            decision=torch.zeros(
                photon_count, dtype=torch.int32, device=device
            ),
            work_overflow=torch.zeros(
                photon_count, dtype=torch.int32, device=device
            ),
        )


@dataclass
class _BoundaryMergeWorkspace:
    """Reusable SoA outputs for the analytic/instanced boundary merge."""

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
    def allocate(cls, torch, device, capacity):
        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("merge workspace capacity must be non-negative")
        return cls(
            distance=torch.empty(capacity, dtype=torch.float32, device=device),
            normal=torch.empty((capacity, 3), dtype=torch.float32, device=device),
            material_from=torch.empty(capacity, dtype=torch.int32, device=device),
            material_to=torch.empty(capacity, dtype=torch.int32, device=device),
            surface=torch.empty(capacity, dtype=torch.int32, device=device),
            instance=torch.empty(capacity, dtype=torch.int32, device=device),
            triangle=torch.empty(capacity, dtype=torch.int32, device=device),
            channel=torch.empty(capacity, dtype=torch.int32, device=device),
            capacity=capacity,
        )

    def outputs(self, count):
        count = int(count)
        return (
            self.distance[:count],
            self.normal[:count],
            self.material_from[:count],
            self.material_to[:count],
            self.surface[:count],
            self.instance[:count],
            self.triangle[:count],
            self.channel[:count],
        )


@dataclass
class _BoundaryRayWorkspace:
    """Reusable gathered boundary rays and PMT distance caps."""

    origins: Any
    directions: Any
    last_instances: Any
    last_triangles: Any
    pmt_tmax: Any
    positive_infinity: Any
    capacity: int

    @classmethod
    def allocate(cls, torch, device, capacity):
        capacity = int(capacity)
        return cls(
            origins=torch.empty((capacity, 3), dtype=torch.float32, device=device),
            directions=torch.empty(
                (capacity, 3), dtype=torch.float32, device=device
            ),
            last_instances=torch.empty(
                capacity, dtype=torch.int32, device=device
            ),
            last_triangles=torch.empty(
                capacity, dtype=torch.int32, device=device
            ),
            pmt_tmax=torch.empty(capacity, dtype=torch.float32, device=device),
            positive_infinity=torch.full(
                (), float("inf"), dtype=torch.float32, device=device
            ),
            capacity=capacity,
        )

    def outputs(self, count):
        count = int(count)
        return (
            self.origins[:count],
            self.directions[:count],
            self.last_instances[:count],
            self.last_triangles[:count],
            self.pmt_tmax[:count],
        )


@dataclass
class _BoundaryBranchWorkspace:
    """Production-only queues for expensive boundary-physics branches.

    Queue entries are boundary-row indices rather than photon slots.  This
    lets a specialized worker recover both the stable photon slot and its
    already-resolved hit without gathering another copy of either structure.
    The three counts remain on the device, so launching the workers never
    introduces a host rendezvous.
    """

    rows: Any
    counts: Any
    capacity: int

    @classmethod
    def allocate(cls, torch, device, capacity):
        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("boundary branch capacity must be non-negative")
        return cls(
            rows=torch.empty(
                (3, capacity), dtype=torch.int32, device=device
            ),
            counts=torch.zeros(3, dtype=torch.int32, device=device),
            capacity=capacity,
        )


def _production_branch_specialization_enabled(
        requested, *, use_random_tape, legacy_specular_reflection,
        chroma_mesh_box_compatibility):
    """Return whether the split production physics path is admissible.

    Keeping this policy independent of CUDA makes the strict-routing
    invariant cheap to test.  Random-tape execution and either legacy switch
    always retain the single certified kernel.
    """

    return bool(
        requested
        and not use_random_tape
        and not legacy_specular_reflection
        and not chroma_mesh_box_compatibility
    )


def _production_portal_boundary_enabled(
        requested, *, branch_specialized_boundary, use_random_tape,
        legacy_specular_reflection, chroma_mesh_box_compatibility):
    """Return whether certified direct-face routing is admissible.

    Portal hit records feed the compact production classifier.  Keeping the
    feature behind every compatibility switch ensures strict replay continues
    to traverse and resolve the unchanged global Chroma geometry.
    """

    return bool(
        requested
        and branch_specialized_boundary
        and _production_branch_specialization_enabled(
            True,
            use_random_tape=use_random_tape,
            legacy_specular_reflection=legacy_specular_reflection,
            chroma_mesh_box_compatibility=chroma_mesh_box_compatibility,
        )
    )


def _production_fused_pmt_enabled(
        requested, *, use_random_tape, legacy_specular_reflection,
        chroma_mesh_box_compatibility, pmt_grid):
    """Admit the lattice-fused PMT query only for corrected production.

    Strict RNG/geometry replay and either historical arithmetic switch retain
    the independently certified traversal.  ``pmt_grid`` names the older
    synchronized count/materialize experiment and is intentionally mutually
    exclusive with the new per-ray fused implementation.
    """

    return bool(
        requested
        and not use_random_tape
        and not legacy_specular_reflection
        and not chroma_mesh_box_compatibility
        and not pmt_grid
    )


def _production_device_scheduler_enabled(
        requested, *, use_random_tape, legacy_specular_reflection,
        chroma_mesh_box_compatibility, branch_specialized_boundary,
        portal_boundary, pmt_grid, fused_pmt=False):
    """Return whether the host-free corrected-production scheduler may run.

    The device scheduler owns monolithic corrected boundary physics, split
    analytic geometry, and fused PMT TLAS/BLAS traversal.  Its optional portal
    partition also keeps both compact queue counts resident on the device and
    is therefore admissible here.  Branch-specialized physics and PMT-grid
    traversal retain their synchronized A/B paths, while every strict-
    compatibility switch keeps the already-certified host scheduler.  The
    experimental fused PMT lattice walk is admissible in this corrected
    scheduler too; the flag does not relax any of those production
    requirements.
    """

    return bool(
        requested
        and not use_random_tape
        and not legacy_specular_reflection
        and not chroma_mesh_box_compatibility
        and not branch_specialized_boundary
        and not pmt_grid
        and (
            not fused_pmt
            or _production_fused_pmt_enabled(
                True,
                use_random_tape=use_random_tape,
                legacy_specular_reflection=legacy_specular_reflection,
                chroma_mesh_box_compatibility=(
                    chroma_mesh_box_compatibility
                ),
                pmt_grid=pmt_grid,
            )
        )
    )


def _production_fused_portal_boundary_enabled(
        requested, *, device_scheduler, portal_boundary, use_random_tape,
        legacy_specular_reflection, chroma_mesh_box_compatibility,
        branch_specialized_boundary, pmt_grid, fused_pmt=False):
    """Admit fused portal physics only in the host-free production path.

    This policy is intentionally stricter than ordinary portal routing.  The
    fused kernel consumes a device-count queue and implements only corrected
    production arithmetic; tape replay and either historical compatibility
    mode must continue through the established geometry and boundary kernels.
    """

    return bool(
        requested
        and portal_boundary
        and _production_device_scheduler_enabled(
            device_scheduler,
            use_random_tape=use_random_tape,
            legacy_specular_reflection=legacy_specular_reflection,
            chroma_mesh_box_compatibility=chroma_mesh_box_compatibility,
            branch_specialized_boundary=branch_specialized_boundary,
            portal_boundary=portal_boundary,
            pmt_grid=pmt_grid,
            fused_pmt=fused_pmt,
        )
    )


def _safe_empty_bounds(scene):
    """Derive an obstacle-free LAr box from compiled detector coordinates."""

    if scene.reachability.source_x_sign != -1:
        raise ValueError("the production backend currently requires source_x_sign=-1")
    active = scene.boxes.kinds.index("active")
    cathode = scene.boxes.kinds.index("cathode")
    # The largest x extent of the rightmost retained wire cylinder is the last
    # obstacle before homogeneous bulk LAr.  Move that one bound by one FP64
    # ulp so the certified region cannot include the cylinder itself.
    # Exact compatibility retains both detector halves' wire planes so an
    # arbitrary source center remains valid.  The collision-first shortcut is
    # still certified only for the selected negative-X component; photons in
    # the opposite half begin outside this AABB and therefore fall through to
    # the exact global boundary stage without a geometry-free advance.
    selected_wire = scene.wires.origin[:, 0] < 0.0
    if not np.any(selected_wire):
        raise RuntimeError("compiled scene has no negative-X wire planes")
    wire_x_hi = np.max(
        scene.wires.origin[selected_wire, 0]
        + scene.wires.radius[selected_wire]
    )
    lower = np.array(
        [
            float(np.nextafter(
                np.float32(wire_x_hi), np.float32(np.inf), dtype=np.float32
            )),
            float(scene.boxes.bounds_min[active, 1]),
            float(scene.boxes.bounds_min[active, 2]),
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            float(scene.boxes.bounds_min[cathode, 0]),
            float(scene.boxes.bounds_max[active, 1]),
            float(scene.boxes.bounds_max[active, 2]),
        ],
        dtype=np.float64,
    )
    if np.any(lower >= upper):
        raise RuntimeError("compiled scene does not contain a homogeneous LAr region")
    return lower, upper


def _merge_boundaries_torch_reference(
        scene_device, analytic, pmt, directions):
    """Readable eager-Torch oracle for the fused merge kernel.

    This is intentionally kept out of the production scheduler.  It locks the
    exact previous implementation into tests, including mesh-on-tie behavior,
    PMT normal orientation, and the material-side convention.
    """

    import torch

    pmt_hit = pmt.instance_ids >= 0
    analytic_hit = analytic.kind != 0
    choose_analytic = analytic_hit & (
        ~pmt_hit
        | (
            analytic.distance.to(torch.float64) + 1.0e-12
            < pmt.distances.to(torch.float64)
        )
    )
    choose_pmt = pmt_hit & ~choose_analytic
    distance = torch.where(choose_analytic, analytic.distance, pmt.distances)
    tri = torch.clamp(pmt.triangle_ids.to(torch.int64), min=0)
    pmt_m1 = scene_device["pmt_scene_material1_index"][tri]
    pmt_m2 = scene_device["pmt_scene_material2_index"][tri]
    pmt_surface = scene_device["pmt_scene_surface_index"][tri]
    raw_dot = torch.sum(pmt.world_normals * (-directions), dim=1)
    pmt_oriented_normal = torch.where(
        (raw_dot > 0.0)[:, None], pmt.world_normals, -pmt.world_normals
    )
    normal = torch.where(
        choose_analytic[:, None],
        analytic.surface_normal,
        pmt_oriented_normal,
    ).contiguous()
    pmt_from = torch.where(raw_dot > 0.0, pmt_m2, pmt_m1)
    pmt_to = torch.where(raw_dot > 0.0, pmt_m1, pmt_m2)

    material_from = torch.where(
        choose_analytic, analytic.material_from_index, pmt_from
    ).to(torch.int32)
    material_to = torch.where(
        choose_analytic, analytic.material_to_index, pmt_to
    ).to(torch.int32)
    surface = torch.where(
        choose_analytic, analytic.surface_index, pmt_surface
    ).to(torch.int32)
    analytic_box = choose_analytic & (analytic.kind == 1)
    encoded_box = -(analytic.index.to(torch.int32) + 2)
    instance = torch.where(
        choose_pmt,
        pmt.instance_ids,
        torch.where(analytic_box, encoded_box, -1),
    ).to(torch.int32)
    triangle = torch.where(
        choose_pmt,
        pmt.triangle_ids,
        torch.where(analytic_box, analytic.primitive_index, -1),
    ).to(torch.int32)
    channel = torch.where(
        choose_pmt, pmt.channel_ids, torch.full_like(pmt.channel_ids, -1)
    ).to(torch.int32)

    no_hit = ~(choose_analytic | choose_pmt)
    material_from = torch.where(no_hit, -1, material_from)
    material_to = torch.where(no_hit, -1, material_to)
    surface = torch.where(no_hit, -1, surface)
    return (
        distance,
        normal,
        material_from,
        material_to,
        surface,
        instance,
        triangle,
        channel,
    )


def _load_boundary_merge_kernel():
    cached = getattr(_load_boundary_merge_kernel, "_cached", None)
    if cached is not None:
        return cached
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - optional installation
        raise RuntimeError("the specialized backend requires Triton") from exc

    # Triton 3.1 resolves JIT globals from the defining module.
    globals().update(triton=triton, tl=tl)

    @triton.jit(do_not_specialize=[25])
    def boundary_merge_kernel(
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
            nitems,
            BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < nitems

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

        # Keep the FP64 epsilon comparison verbatim.  Equal distances and
        # distances within 1e-12 select the mesh, matching fill_state.
        analytic_strictly_closer = (
            analytic_d.to(tl.float64) + 1.0e-12
            < pmt_d.to(tl.float64)
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

        pnx = tl.load(pmt_normal + lane * 3 + 0, mask=valid, other=0.0)
        pny = tl.load(pmt_normal + lane * 3 + 1, mask=valid, other=0.0)
        pnz = tl.load(pmt_normal + lane * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + lane * 3 + 0, mask=valid, other=0.0)
        dy = tl.load(directions + lane * 3 + 1, mask=valid, other=0.0)
        dz = tl.load(directions + lane * 3 + 2, mask=valid, other=0.0)
        raw_dot = pnx * (-dx) + pny * (-dy) + pnz * (-dz)
        normal_faces_ray = raw_dot > 0.0
        oriented_x = tl.where(normal_faces_ray, pnx, -pnx)
        oriented_y = tl.where(normal_faces_ray, pny, -pny)
        oriented_z = tl.where(normal_faces_ray, pnz, -pnz)

        anx = tl.load(
            analytic_normal + lane * 3 + 0, mask=valid, other=0.0
        )
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
        tl.store(out_normal + lane * 3 + 0, nx, mask=valid)
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
                choose_pmt, pmt_instance_id,
                tl.where(analytic_box, encoded_box_instance, -1),
            ),
            mask=valid,
        )
        tl.store(
            out_triangle + lane,
            tl.where(
                choose_pmt, pmt_triangle_id,
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

    _load_boundary_merge_kernel._cached = (triton, boundary_merge_kernel)
    return _load_boundary_merge_kernel._cached


def _load_chroma_global_merge_kernel():
    """Load the certificate-only Chroma global-mesh/wire merge."""

    cached = getattr(_load_chroma_global_merge_kernel, "_cached", None)
    if cached is not None:
        return cached
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - optional installation
        raise RuntimeError("the specialized backend requires Triton") from exc
    globals().update(triton=triton, tl=tl)

    @triton.jit(do_not_specialize=[21])
    def chroma_global_merge_kernel(
            analytic_distance,
            analytic_kind,
            analytic_normal,
            analytic_material_from,
            analytic_material_to,
            analytic_surface,
            mesh_distance,
            mesh_normal,
            mesh_material_from,
            mesh_material_to,
            mesh_surface,
            mesh_triangle,
            mesh_channel,
            out_distance,
            out_normal,
            out_material_from,
            out_material_to,
            out_surface,
            out_instance,
            out_triangle,
            out_channel,
            nitems,
            BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < nitems
        analytic_d = tl.load(
            analytic_distance + lane, mask=valid, other=float("inf")
        )
        # The historical CUDA supplement contains wires only.  Treating any
        # accidental analytic-box result as a miss makes this path fail closed
        # instead of replacing a triangle from the fingerprinted global mesh.
        wire_hit = tl.load(
            analytic_kind + lane, mask=valid, other=0
        ).to(tl.int32) == 2
        mesh_d = tl.load(
            mesh_distance + lane, mask=valid, other=float("inf")
        )
        mesh_triangle_id = tl.load(
            mesh_triangle + lane, mask=valid, other=-1
        ).to(tl.int32)
        mesh_hit = mesh_triangle_id >= 0

        # This is the literal comparison in photon.h: the analytic wire wins
        # only when it is more than 1e-12 closer after promotion to double.
        wire_strictly_closer = (
            analytic_d.to(tl.float64) + 1.0e-12 < mesh_d.to(tl.float64)
        )
        choose_wire = wire_hit & (~mesh_hit | wire_strictly_closer)
        choose_mesh = mesh_hit & ~choose_wire
        no_hit = ~(choose_wire | choose_mesh)

        # Select raw words rather than floating values so signed zero from the
        # CUDA-compatible normal and distance calculations survives the merge.
        analytic_distance_bits = analytic_d.to(tl.uint32, bitcast=True)
        mesh_distance_bits = mesh_d.to(tl.uint32, bitcast=True)
        selected_distance = tl.where(
            choose_wire, analytic_distance_bits, mesh_distance_bits
        ).to(tl.float32, bitcast=True)
        tl.store(out_distance + lane, selected_distance, mask=valid)
        for axis in tl.static_range(3):
            analytic_component = tl.load(
                analytic_normal + lane * 3 + axis, mask=valid, other=0.0
            )
            mesh_component = tl.load(
                mesh_normal + lane * 3 + axis, mask=valid, other=0.0
            )
            component_bits = tl.where(
                choose_wire,
                analytic_component.to(tl.uint32, bitcast=True),
                mesh_component.to(tl.uint32, bitcast=True),
            )
            tl.store(
                out_normal + lane * 3 + axis,
                component_bits.to(tl.float32, bitcast=True),
                mask=valid,
            )

        analytic_from = tl.load(
            analytic_material_from + lane, mask=valid, other=-1
        ).to(tl.int32)
        analytic_to = tl.load(
            analytic_material_to + lane, mask=valid, other=-1
        ).to(tl.int32)
        analytic_surface_id = tl.load(
            analytic_surface + lane, mask=valid, other=-1
        ).to(tl.int32)
        mesh_from = tl.load(
            mesh_material_from + lane, mask=valid, other=-1
        ).to(tl.int32)
        mesh_to = tl.load(
            mesh_material_to + lane, mask=valid, other=-1
        ).to(tl.int32)
        mesh_surface_id = tl.load(
            mesh_surface + lane, mask=valid, other=-1
        ).to(tl.int32)
        tl.store(
            out_material_from + lane,
            tl.where(
                no_hit, -1,
                tl.where(choose_wire, analytic_from, mesh_from),
            ),
            mask=valid,
        )
        tl.store(
            out_material_to + lane,
            tl.where(
                no_hit, -1,
                tl.where(choose_wire, analytic_to, mesh_to),
            ),
            mask=valid,
        )
        tl.store(
            out_surface + lane,
            tl.where(
                no_hit, -1,
                tl.where(choose_wire, analytic_surface_id, mesh_surface_id),
            ),
            mask=valid,
        )

        # Exact mode adopts Chroma's global last_hit_triangle namespace.
        # There is no second instance namespace; -2 remains the analytic-wire
        # sentinel used by photon.h.
        tl.store(out_instance + lane, -1, mask=valid)
        tl.store(
            out_triangle + lane,
            tl.where(
                choose_wire,
                -2,
                tl.where(choose_mesh, mesh_triangle_id, -1),
            ),
            mask=valid,
        )
        mesh_channel_id = tl.load(
            mesh_channel + lane, mask=valid, other=-1
        ).to(tl.int32)
        tl.store(
            out_channel + lane,
            tl.where(choose_mesh, mesh_channel_id, -1),
            mask=valid,
        )

    _load_chroma_global_merge_kernel._cached = (
        triton, chroma_global_merge_kernel
    )
    return _load_chroma_global_merge_kernel._cached


def _load_boundary_kernel():
    cached = getattr(_load_boundary_kernel, "_cached", None)
    if cached is not None:
        return cached
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
        from chroma.triton.physics_kernels import (
            acos_chroma_fast,
            advance_position_chroma_fast,
            advance_time_chroma_fast,
            exponential_distance_chroma_fast,
            fresnel_step_chroma,
            fresnel_step as physics_fresnel_step,
            rayleigh_scatter as physics_rayleigh_scatter,
            reflect_specular as physics_reflect_specular,
            reflect_specular_chroma as physics_reflect_specular_chroma,
            sample_bulk_collision as physics_sample_bulk_collision,
        )
        from chroma.triton.rng_alignment import random_tape_uniform_at
    except ImportError as exc:  # pragma: no cover - optional installation
        raise RuntimeError("the specialized backend requires torch and Triton") from exc

    # Triton 3.1 resolves names from module globals, not Python closures.
    globals().update(
        triton=triton,
        tl=tl,
        libdevice=libdevice,
        acos_chroma_fast=acos_chroma_fast,
        advance_position_chroma_fast=advance_position_chroma_fast,
        advance_time_chroma_fast=advance_time_chroma_fast,
        exponential_distance_chroma_fast=exponential_distance_chroma_fast,
        fresnel_step_chroma=fresnel_step_chroma,
        physics_fresnel_step=physics_fresnel_step,
        physics_rayleigh_scatter=physics_rayleigh_scatter,
        physics_reflect_specular=physics_reflect_specular,
        physics_reflect_specular_chroma=physics_reflect_specular_chroma,
        physics_sample_bulk_collision=physics_sample_bulk_collision,
        random_tape_uniform_at=random_tape_uniform_at,
    )

    @triton.jit
    def boundary_legacy_pick_new_direction(
            axis_x, axis_y, axis_z, theta, phi):
        """Literal ``photon.h::pick_new_direction`` for tape lockstep."""

        cos_theta = libdevice.fast_cosf(theta)
        sin_theta = libdevice.fast_sinf(theta)
        cos_phi = libdevice.fast_cosf(phi)
        sin_phi = libdevice.fast_sinf(phi)
        sin_axis_theta = tl.sqrt(1.0 - axis_z * axis_z)
        ordinary_axis = (sin_axis_theta == sin_axis_theta) & (
            sin_axis_theta >= 0.00001
        )
        cos_axis_phi = tl.where(ordinary_axis, axis_x / sin_axis_theta, 1.0)
        sin_axis_phi = tl.where(ordinary_axis, axis_y / sin_axis_theta, 0.0)
        return (
            cos_theta * axis_x + sin_theta * (
                axis_z * cos_phi * cos_axis_phi - sin_phi * sin_axis_phi
            ),
            cos_theta * axis_y + sin_theta * (
                cos_phi * axis_z * sin_axis_phi + sin_phi * cos_axis_phi
            ),
            cos_theta * axis_z - sin_theta * cos_phi * sin_axis_theta,
        )

    globals()["boundary_legacy_pick_new_direction"] = (
        boundary_legacy_pick_new_direction
    )

    @triton.jit
    def boundary_legacy_rayleigh(qx, qy, qz, uniform_theta, uniform_phi):
        """Two-draw Rayleigh sampler in legacy Chroma's draw order."""

        pi: tl.constexpr = 3.14159265358979323846
        cos_theta = 2.0 * libdevice.fast_cosf(
            (acos_chroma_fast(1.0 - 2.0 * uniform_theta) - 2.0 * pi) / 3.0
        )
        cos_theta = tl.maximum(-1.0, tl.minimum(1.0, cos_theta))
        theta = acos_chroma_fast(cos_theta)
        phi = uniform_phi * (2.0 * pi)
        dx, dy, dz = boundary_legacy_pick_new_direction(
            qx, qy, qz, theta, phi
        )
        special = 1.0 - tl.abs(cos_theta) < 1.0e-6
        special_qx, special_qy, special_qz = (
            boundary_legacy_pick_new_direction(
                qx, qy, qz, pi / 2.0, phi
            )
        )
        ordinary_qx = qx - cos_theta * dx
        ordinary_qy = qy - cos_theta * dy
        ordinary_qz = qz - cos_theta * dz
        new_qx = tl.where(special, special_qx, ordinary_qx)
        new_qy = tl.where(special, special_qy, ordinary_qy)
        new_qz = tl.where(special, special_qz, ordinary_qz)
        direction_norm = tl.sqrt(dx * dx + dy * dy + dz * dz)
        polarization_norm = tl.sqrt(
            new_qx * new_qx + new_qy * new_qy + new_qz * new_qz
        )
        return (
            dx / direction_norm,
            dy / direction_norm,
            dz / direction_norm,
            new_qx / polarization_norm,
            new_qy / polarization_norm,
            new_qz / polarization_norm,
        )

    globals()["boundary_legacy_rayleigh"] = boundary_legacy_rayleigh

    @triton.jit
    def boundary_legacy_uniform_sphere(uniform_theta, uniform_z):
        """Literal CUDA-12.4 lowering of ``random.h::uniform_sphere``."""

        # Keep this opaque: CUDA fast-math uses two FMAs for the affine
        # transforms, a separately rounded square/subtract, approximate sqrt
        # and trig, then separate radial products.  Ordinary Triton algebra
        # selects different trig intrinsics and changes the diffuse result by
        # several ulps.
        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .f32 theta;
                .reg .f32 z;
                .reg .f32 square;
                .reg .f32 complement;
                .reg .f32 radial;
                .reg .f32 cosine;
                .reg .f32 sine;
                fma.rn.ftz.f32 theta, $3, 0f40C90FDB, 0f00000000;
                fma.rn.ftz.f32 z, $4, 0f40000000, 0fBF800000;
                mul.ftz.f32 square, z, z;
                sub.ftz.f32 complement, 0f3F800000, square;
                sqrt.approx.ftz.f32 radial, complement;
                cos.approx.ftz.f32 cosine, theta;
                mul.ftz.f32 $0, radial, cosine;
                sin.approx.ftz.f32 sine, theta;
                mul.ftz.f32 $1, radial, sine;
                mov.f32 $2, z;
            }
            """,
            constraints="=f,=f,=f,f,f",
            args=[uniform_theta, uniform_z],
            dtype=(tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )

    globals()["boundary_legacy_uniform_sphere"] = (
        boundary_legacy_uniform_sphere
    )

    @triton.jit
    def boundary_legacy_orient_diffuse(
            candidate_x, candidate_y, candidate_z,
            normal_x, normal_y, normal_z):
        """Match CUDA's diffuse normal dot tree and conditional negation."""

        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .f32 partial;
                .reg .f32 dot_value;
                .reg .f32 negative_x;
                .reg .f32 negative_y;
                .reg .f32 negative_z;
                .reg .f32 negative_dot;
                .reg .pred keep;
                mul.ftz.f32 partial, $5, $8;
                fma.rn.ftz.f32 dot_value, $4, $7, partial;
                fma.rn.ftz.f32 dot_value, $6, $9, dot_value;
                setp.geu.ftz.f32 keep, dot_value, 0f00000000;
                neg.ftz.f32 negative_x, $4;
                neg.ftz.f32 negative_y, $5;
                neg.ftz.f32 negative_z, $6;
                neg.ftz.f32 negative_dot, dot_value;
                selp.f32 $0, $4, negative_x, keep;
                selp.f32 $1, $5, negative_y, keep;
                selp.f32 $2, $6, negative_z, keep;
                selp.f32 $3, dot_value, negative_dot, keep;
            }
            """,
            constraints="=f,=f,=f,=f,f,f,f,f,f,f",
            args=[
                candidate_x, candidate_y, candidate_z,
                normal_x, normal_y, normal_z,
            ],
            dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )

    globals()["boundary_legacy_orient_diffuse"] = (
        boundary_legacy_orient_diffuse
    )

    @triton.jit
    def boundary_legacy_diffuse_polarization(
            sphere_x, sphere_y, sphere_z,
            direction_x, direction_y, direction_z):
        """Match CUDA cross/norm/division words for diffuse polarization."""

        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .f32 left;
                .reg .f32 right;
                .reg .f32 cross_x;
                .reg .f32 cross_y;
                .reg .f32 cross_z;
                .reg .f32 norm2;
                .reg .f32 length;
                mul.ftz.f32 left, $4, $8;
                mul.ftz.f32 right, $5, $7;
                sub.ftz.f32 cross_x, left, right;
                mul.ftz.f32 left, $5, $6;
                mul.ftz.f32 right, $3, $8;
                sub.ftz.f32 cross_y, left, right;
                mul.ftz.f32 left, $3, $7;
                mul.ftz.f32 right, $4, $6;
                sub.ftz.f32 cross_z, left, right;
                mul.ftz.f32 norm2, cross_y, cross_y;
                fma.rn.ftz.f32 norm2, cross_x, cross_x, norm2;
                fma.rn.ftz.f32 norm2, cross_z, cross_z, norm2;
                sqrt.approx.ftz.f32 length, norm2;
                div.approx.ftz.f32 $0, cross_x, length;
                div.approx.ftz.f32 $1, cross_y, length;
                div.approx.ftz.f32 $2, cross_z, length;
            }
            """,
            constraints="=f,=f,=f,f,f,f,f,f,f",
            args=[
                sphere_x, sphere_y, sphere_z,
                direction_x, direction_y, direction_z,
            ],
            dtype=(tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )

    globals()["boundary_legacy_diffuse_polarization"] = (
        boundary_legacy_diffuse_polarization
    )

    @triton.jit(do_not_specialize=[30, 31, 32, 33])
    def boundary_step_kernel(
            positions,
            directions,
            polarizations,
            times,
            histories,
            rng_counters,
            last_instances,
            last_triangles,
            detected_channels,
            step_counts,
            input_queue,
            hit_distances,
            hit_normals,
            material_from,
            material_to,
            surface_indices,
            hit_instances,
            hit_triangles,
            hit_channels,
            material_refractive_index,
            material_absorption_length,
            material_scattering_length,
            surface_detect,
            surface_absorb,
            surface_reflect_diffuse,
            surface_reflect_specular,
            global_photon_ids,
            output_status,
            output_queue,
            output_count,
            nitems,
            seed,
            photon_id_base,
            max_steps,
            tape_values,
            tape_global_ids,
            tape_row_indices,
            tape_interaction_cursor,
            tape_draw_cursor,
            tape_overflow,
            tape_interaction_certificate,
            tape_state_certificate,
            tape_trace_interaction,
            tape_trace_draw_count,
            tape_trace_decision,
            tape_trace_work_overflow,
            tape_photon_count,
            USE_GLOBAL_IDS: tl.constexpr,
            USE_RANDOM_TAPE: tl.constexpr,
            USE_TAPE_ROWS: tl.constexpr,
            CERTIFY_RANDOM_TAPE: tl.constexpr,
            CERTIFY_STATE: tl.constexpr,
            TRACE_RANDOM_TAPE: tl.constexpr,
            MAX_TAPE_INTERACTIONS: tl.constexpr,
            TAPE_DRAWS_PER_INTERACTION: tl.constexpr,
            STATE_WORDS_PER_INTERACTION: tl.constexpr,
            MAX_DIFFUSE_ATTEMPTS: tl.constexpr,
            LEGACY_SPECULAR_REFLECTION: tl.constexpr,
            DIRECT_QUEUE: tl.constexpr,
            NITEMS_IS_POINTER: tl.constexpr,
            BLOCK: tl.constexpr):
        program_start = tl.program_id(0) * BLOCK
        if NITEMS_IS_POINTER:
            live_items = tl.load(nitems).to(tl.int32)
            # Device-resident scheduling launches the known capacity so the
            # boundary consumer can follow geometry without a host count
            # read.  A lane mask alone would still execute the monolithic
            # Rayleigh/diffuse/Fresnel body in every empty suffix CTA.  This
            # condition is uniform across the program and retires it after a
            # single scalar counter load.
            if program_start >= live_items:
                return
        else:
            live_items = nitems
        lane = program_start + tl.arange(0, BLOCK)
        valid = lane < live_items
        photon_id = tl.load(input_queue + lane, mask=valid, other=0).to(tl.int64)
        base = photon_id * 3
        px = tl.load(positions + base, mask=valid, other=0.0)
        py = tl.load(positions + base + 1, mask=valid, other=0.0)
        pz = tl.load(positions + base + 2, mask=valid, other=0.0)
        dx = tl.load(directions + base, mask=valid, other=1.0)
        dy = tl.load(directions + base + 1, mask=valid, other=0.0)
        dz = tl.load(directions + base + 2, mask=valid, other=0.0)
        qx = tl.load(polarizations + base, mask=valid, other=0.0)
        qy = tl.load(polarizations + base + 1, mask=valid, other=1.0)
        qz = tl.load(polarizations + base + 2, mask=valid, other=0.0)
        photon_time = tl.load(times + photon_id, mask=valid, other=0.0)
        history = tl.load(histories + photon_id, mask=valid, other=0).to(tl.int32)
        rng_counter = tl.load(rng_counters + photon_id, mask=valid, other=0).to(tl.int64)
        step_count = tl.load(step_counts + photon_id, mask=valid, other=0).to(tl.int32)

        distance_to_boundary = tl.load(
            hit_distances + lane, mask=valid, other=float("inf")
        )
        nx = tl.load(hit_normals + lane*3, mask=valid, other=1.0)
        ny = tl.load(hit_normals + lane*3 + 1, mask=valid, other=0.0)
        nz = tl.load(hit_normals + lane*3 + 2, mask=valid, other=0.0)
        from_index = tl.load(material_from + lane, mask=valid, other=-1).to(tl.int32)
        to_index = tl.load(material_to + lane, mask=valid, other=-1).to(tl.int32)
        surface_index = tl.load(surface_indices + lane, mask=valid, other=-1).to(tl.int32)
        hit_instance = tl.load(hit_instances + lane, mask=valid, other=-1).to(tl.int32)
        hit_triangle = tl.load(hit_triangles + lane, mask=valid, other=-1).to(tl.int32)
        hit_channel = tl.load(hit_channels + lane, mask=valid, other=-1).to(tl.int32)
        has_hit = valid & (from_index >= 0) & (distance_to_boundary < float("inf"))

        safe_from = tl.maximum(from_index, 0)
        safe_to = tl.maximum(to_index, 0)
        refractive1 = tl.load(
            material_refractive_index + safe_from, mask=has_hit, other=1.0
        )
        refractive2 = tl.load(
            material_refractive_index + safe_to, mask=has_hit, other=1.0
        )
        absorption_length = tl.load(
            material_absorption_length + safe_from, mask=has_hit, other=float("inf")
        )
        scattering_length = tl.load(
            material_scattering_length + safe_from, mask=has_hit, other=float("inf")
        )

        if USE_GLOBAL_IDS:
            global_id = tl.load(
                global_photon_ids + photon_id, mask=valid, other=0
            ).to(tl.int64)
        else:
            global_id = photon_id + photon_id_base.to(tl.int64)

        # The tape row is stable under queue reorder, source tiling, and the
        # cross-tile reservoir.  Any mapping/cursor failure is sticky and fails
        # closed; alignment mode never falls back to production Philox.
        if USE_RANDOM_TAPE:
            if USE_TAPE_ROWS:
                tape_row = tl.load(
                    tape_row_indices + photon_id, mask=valid, other=-1
                ).to(tl.int64)
            else:
                tape_row = photon_id
            tape_row_valid = valid & (tape_row >= 0) & (
                tape_row < tape_photon_count
            )
            stored_global_id = tl.load(
                tape_global_ids + tape_row,
                mask=tape_row_valid,
                other=-2,
            ).to(tl.int64)
            mapping_ok = tape_row_valid & (stored_global_id == global_id)
            tape_interaction = tl.load(
                tape_interaction_cursor + tape_row,
                mask=tape_row_valid,
                other=0,
            ).to(tl.int32)
            tape_draw = tl.load(
                tape_draw_cursor + tape_row,
                mask=tape_row_valid,
                other=0,
            ).to(tl.int32)
            initial_tape_draw = tape_draw
            tape_flags = tl.load(
                tape_overflow + tape_row,
                mask=tape_row_valid,
                other=0,
            ).to(tl.int32)
            tape_flags |= tl.where(valid & ~tape_row_valid, 8, 0)
            tape_flags |= tl.where(tape_row_valid & ~mapping_ok, 4, 0)
            interaction_ok = (
                (tape_interaction >= 0)
                & (tape_interaction < MAX_TAPE_INTERACTIONS)
            )
            tape_flags |= tl.where(mapping_ok & ~interaction_ok, 2, 0)
            prior_tape_failure = has_hit & (tape_flags != 0)
            distance_draws_ok = (
                has_hit
                & (tape_flags == 0)
                & mapping_ok
                & interaction_ok
                & (tape_draw >= 0)
                & (tape_draw + 1 < TAPE_DRAWS_PER_INTERACTION)
            )
            u0 = random_tape_uniform_at(
                tape_values, tape_row, tape_interaction, tape_draw,
                distance_draws_ok,
                MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
            )
            u1 = random_tape_uniform_at(
                tape_values, tape_row, tape_interaction, tape_draw + 1,
                distance_draws_ok,
                MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
            )
            distance_draw_failure = has_hit & ~prior_tape_failure & (
                ~distance_draws_ok
            )
        else:
            tape_row = photon_id
            tape_row_valid = valid
            mapping_ok = valid
            tape_interaction = tl.zeros((BLOCK,), tl.int32)
            tape_draw = tl.zeros((BLOCK,), tl.int32)
            initial_tape_draw = tape_draw
            tape_flags = tl.zeros((BLOCK,), tl.int32)
            prior_tape_failure = tl.zeros((BLOCK,), tl.int1)
            distance_draw_failure = tl.zeros((BLOCK,), tl.int1)
            random_offset = global_id * 4294967296 + rng_counter
            u0, u1, u2, u3 = tl.rand4x(seed, random_offset)
            u4, u5, u6, u7 = tl.rand4x(seed, random_offset + 1)
            rng_counter += has_hit.to(tl.int64) * 2

        if USE_RANDOM_TAPE:
            huge: tl.constexpr = 1.0e30
            absorption_distance = tl.where(
                absorption_length <= 0.0,
                0.0,
                tl.where(
                    absorption_length == float("inf"),
                    huge,
                    exponential_distance_chroma_fast(absorption_length, u0),
                ),
            )
            scattering_distance = tl.where(
                scattering_length <= 0.0,
                0.0,
                tl.where(
                    scattering_length == float("inf"),
                    huge,
                    exponential_distance_chroma_fast(scattering_length, u1),
                ),
            )
            absorption_wins = absorption_distance <= scattering_distance
            collision_distance = tl.where(
                absorption_wins, absorption_distance, scattering_distance
            )
            bulk_collision_candidate = distance_draws_ok & (
                collision_distance <= distance_to_boundary
            )
            bulk_absorb_candidate = bulk_collision_candidate & absorption_wins
            bulk_scatter_candidate = bulk_collision_candidate & ~absorption_wins
            rayleigh_draws_ok = bulk_scatter_candidate & (
                tape_draw + 3 < TAPE_DRAWS_PER_INTERACTION
            )
            rayleigh_u0 = random_tape_uniform_at(
                tape_values, tape_row, tape_interaction, tape_draw + 2,
                rayleigh_draws_ok,
                MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
            )
            rayleigh_u1 = random_tape_uniform_at(
                tape_values, tape_row, tape_interaction, tape_draw + 3,
                rayleigh_draws_ok,
                MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
            )
            rayleigh_draw_failure = bulk_scatter_candidate & ~rayleigh_draws_ok
            bulk_absorb = bulk_absorb_candidate
            bulk_scatter = bulk_scatter_candidate & rayleigh_draws_ok
            bulk_draw_failure = distance_draw_failure | rayleigh_draw_failure
            at_boundary = distance_draws_ok & ~bulk_collision_candidate
            new_dx, new_dy, new_dz, new_qx, new_qy, new_qz = (
                boundary_legacy_rayleigh(
                    qx, qy, qz, rayleigh_u0, rayleigh_u1
                )
            )
        else:
            collision_distance, process = physics_sample_bulk_collision(
                absorption_length, scattering_length, u0, u1
            )
            # Chroma treats a zero absorption length as a zero-distance absorber.
            zero_absorption = has_hit & (absorption_length <= 0.0)
            collision_distance = tl.where(
                zero_absorption, 0.0, collision_distance
            )
            process = tl.where(zero_absorption, 1, process)  # BULK_ABSORB
            bulk_collision = has_hit & (process != 0) & (
                collision_distance <= distance_to_boundary
            )
            bulk_absorb = bulk_collision & (process == 1)
            bulk_scatter = bulk_collision & (process == 2)
            bulk_absorb_candidate = bulk_absorb
            bulk_scatter_candidate = bulk_scatter
            bulk_draw_failure = tl.zeros((BLOCK,), tl.int1)
            at_boundary = has_hit & ~bulk_collision
            new_dx, new_dy, new_dz, new_qx, new_qy, new_qz = (
                physics_rayleigh_scatter(
                    dx, dy, dz, qx, qy, qz, u2, u3, u4, u5
                )
            )

        advance = tl.where(
            bulk_absorb | bulk_scatter, collision_distance,
            tl.where(at_boundary, distance_to_boundary, 0.0),
        )
        if USE_RANDOM_TAPE:
            px = advance_position_chroma_fast(px, advance, dx)
            py = advance_position_chroma_fast(py, advance, dy)
            pz = advance_position_chroma_fast(pz, advance, dz)
            photon_time = advance_time_chroma_fast(
                photon_time, advance, refractive1
            )
        else:
            px += advance * dx
            py += advance * dy
            pz += advance * dz
            photon_time += advance * refractive1 / 299.792458

        history = tl.where(bulk_absorb, history | (1 << 1), history)
        history = tl.where(bulk_scatter, history | (1 << 4), history)
        dx = tl.where(bulk_scatter, new_dx, dx)
        dy = tl.where(bulk_scatter, new_dy, dy)
        dz = tl.where(bulk_scatter, new_dz, dz)
        qx = tl.where(bulk_scatter, new_qx, qx)
        qy = tl.where(bulk_scatter, new_qy, qy)
        qz = tl.where(bulk_scatter, new_qz, qz)

        has_surface = at_boundary & (surface_index >= 0)
        safe_surface = tl.maximum(surface_index, 0)
        probability_absorb = tl.load(
            surface_absorb + safe_surface, mask=has_surface, other=0.0
        )
        probability_detect = tl.load(
            surface_detect + safe_surface, mask=has_surface, other=0.0
        )
        probability_diffuse = tl.load(
            surface_reflect_diffuse + safe_surface, mask=has_surface, other=0.0
        )
        probability_specular = tl.load(
            surface_reflect_specular + safe_surface, mask=has_surface, other=0.0
        )
        cut_absorb = probability_absorb
        cut_detect = cut_absorb + probability_detect
        cut_diffuse = cut_detect + probability_diffuse
        cut_specular = cut_diffuse + probability_specular

        if USE_RANDOM_TAPE:
            surface_selector_ok = has_surface & (
                tape_draw + 2 < TAPE_DRAWS_PER_INTERACTION
            )
            surface_uniform = random_tape_uniform_at(
                tape_values, tape_row, tape_interaction, tape_draw + 2,
                surface_selector_ok,
                MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
            )
            surface_selector_failure = has_surface & ~surface_selector_ok
            surface_absorbed = surface_selector_ok & (
                surface_uniform < cut_absorb
            )
            surface_detected = surface_selector_ok & (
                surface_uniform >= cut_absorb
            ) & (surface_uniform < cut_detect)
            diffuse_candidate = surface_selector_ok & (
                surface_uniform >= cut_detect
            ) & (surface_uniform < cut_diffuse)
            specular = surface_selector_ok & (
                surface_uniform >= cut_diffuse
            ) & (surface_uniform < cut_specular)
            surface_pass = surface_selector_ok & (
                surface_uniform >= cut_specular
            )
            dielectric_candidate = at_boundary & (
                ~has_surface | surface_pass
            )

            # Default diffuse surfaces use Chroma's variable-length rejection
            # loop exactly: two sphere draws and one accept draw per attempt,
            # followed by two more sphere draws for polarization.  The tape is
            # bounded, so exhaustion is explicit rather than silently wrapping.
            diffuse_accepted = tl.zeros((BLOCK,), tl.int1)
            diffuse_cursor = tape_draw + 3
            diffuse_dx = dx
            diffuse_dy = dy
            diffuse_dz = dz
            for _ in tl.static_range(MAX_DIFFUSE_ATTEMPTS):
                attempt = diffuse_candidate & ~diffuse_accepted & (
                    diffuse_cursor + 2 < TAPE_DRAWS_PER_INTERACTION
                )
                sphere_theta = random_tape_uniform_at(
                    tape_values, tape_row, tape_interaction, diffuse_cursor,
                    attempt,
                    MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                    DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
                )
                sphere_z = random_tape_uniform_at(
                    tape_values, tape_row, tape_interaction, diffuse_cursor + 1,
                    attempt,
                    MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                    DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
                )
                accept_uniform = random_tape_uniform_at(
                    tape_values, tape_row, tape_interaction, diffuse_cursor + 2,
                    attempt,
                    MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                    DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
                )
                candidate_x, candidate_y, candidate_z = (
                    boundary_legacy_uniform_sphere(sphere_theta, sphere_z)
                )
                candidate_x, candidate_y, candidate_z, ndotv = (
                    boundary_legacy_orient_diffuse(
                        candidate_x, candidate_y, candidate_z, nx, ny, nz
                    )
                )
                accepted_now = attempt & (accept_uniform < ndotv)
                diffuse_dx = tl.where(accepted_now, candidate_x, diffuse_dx)
                diffuse_dy = tl.where(accepted_now, candidate_y, diffuse_dy)
                diffuse_dz = tl.where(accepted_now, candidate_z, diffuse_dz)
                diffuse_accepted |= accepted_now
                diffuse_cursor += attempt.to(tl.int32) * 3

            diffuse_polarization_ok = diffuse_accepted & (
                diffuse_cursor + 1 < TAPE_DRAWS_PER_INTERACTION
            )
            polarization_theta = random_tape_uniform_at(
                tape_values, tape_row, tape_interaction, diffuse_cursor,
                diffuse_polarization_ok,
                MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
            )
            polarization_z = random_tape_uniform_at(
                tape_values, tape_row, tape_interaction, diffuse_cursor + 1,
                diffuse_polarization_ok,
                MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
            )
            pol_x, pol_y, pol_z = boundary_legacy_uniform_sphere(
                polarization_theta, polarization_z
            )
            diffuse_qx, diffuse_qy, diffuse_qz = (
                boundary_legacy_diffuse_polarization(
                    pol_x, pol_y, pol_z,
                    diffuse_dx, diffuse_dy, diffuse_dz,
                )
            )
            diffuse = diffuse_candidate & diffuse_polarization_ok
            diffuse_draw_failure = diffuse_candidate & ~diffuse

            dielectric_draw_base = tl.where(
                has_surface, tape_draw + 3, tape_draw + 2
            )
            dielectric_draws_ok = dielectric_candidate & (
                dielectric_draw_base + 1 < TAPE_DRAWS_PER_INTERACTION
            )
            dielectric_u0 = random_tape_uniform_at(
                tape_values, tape_row, tape_interaction,
                dielectric_draw_base, dielectric_draws_ok,
                MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
            )
            dielectric_u1 = random_tape_uniform_at(
                tape_values, tape_row, tape_interaction,
                dielectric_draw_base + 1, dielectric_draws_ok,
                MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
            )
            dielectric = dielectric_candidate & dielectric_draws_ok
            dielectric_draw_failure = dielectric_candidate & ~dielectric
        else:
            surface_absorbed = has_surface & (u2 < cut_absorb)
            surface_detected = has_surface & (u2 >= cut_absorb) & (
                u2 < cut_detect
            )
            diffuse = has_surface & (u2 >= cut_detect) & (u2 < cut_diffuse)
            diffuse_candidate = diffuse
            specular = has_surface & (u2 >= cut_diffuse) & (u2 < cut_specular)
            # A probability remainder means PASS in Chroma, followed
            # immediately by dielectric propagation in this same interaction.
            dielectric = at_boundary & (~has_surface | (u2 >= cut_specular))
            surface_selector_failure = tl.zeros((BLOCK,), tl.int1)
            diffuse_draw_failure = tl.zeros((BLOCK,), tl.int1)
            dielectric_draw_failure = tl.zeros((BLOCK,), tl.int1)
            dielectric_u0 = u6
            dielectric_u1 = u7

            # The production sampler is fixed-work and distribution-exact.
            use_xy = tl.abs(nz) < 0.9
            b1x = tl.where(use_xy, -ny, 0.0)
            b1y = tl.where(use_xy, nx, -nz)
            b1z = tl.where(use_xy, 0.0, ny)
            inv_b1 = tl.rsqrt(tl.maximum(
                b1x*b1x + b1y*b1y + b1z*b1z, 1.0e-20
            ))
            b1x, b1y, b1z = b1x*inv_b1, b1y*inv_b1, b1z*inv_b1
            b2x = ny*b1z - nz*b1y
            b2y = nz*b1x - nx*b1z
            b2z = nx*b1y - ny*b1x
            radial = tl.sqrt(u3)
            axial = tl.sqrt(tl.maximum(0.0, 1.0-u3))
            phi = 6.283185307179586 * u4
            tangent_cos = radial * tl.cos(phi)
            tangent_sin = radial * tl.sin(phi)
            diffuse_dx = axial*nx + tangent_cos*b1x + tangent_sin*b2x
            diffuse_dy = axial*ny + tangent_cos*b1y + tangent_sin*b2y
            diffuse_dz = axial*nz + tangent_cos*b1z + tangent_sin*b2z
            pol_use_xy = tl.abs(diffuse_dz) < 0.9
            c1x = tl.where(pol_use_xy, -diffuse_dy, 0.0)
            c1y = tl.where(pol_use_xy, diffuse_dx, -diffuse_dz)
            c1z = tl.where(pol_use_xy, 0.0, diffuse_dy)
            inv_c1 = tl.rsqrt(tl.maximum(
                c1x*c1x + c1y*c1y + c1z*c1z, 1.0e-20
            ))
            c1x, c1y, c1z = c1x*inv_c1, c1y*inv_c1, c1z*inv_c1
            c2x = diffuse_dy*c1z - diffuse_dz*c1y
            c2y = diffuse_dz*c1x - diffuse_dx*c1z
            c2z = diffuse_dx*c1y - diffuse_dy*c1x
            pol_phi = 6.283185307179586 * u5
            diffuse_qx = tl.cos(pol_phi)*c1x + tl.sin(pol_phi)*c2x
            diffuse_qy = tl.cos(pol_phi)*c1y + tl.sin(pol_phi)*c2y
            diffuse_qz = tl.cos(pol_phi)*c1z + tl.sin(pol_phi)*c2z

        history = tl.where(surface_absorbed, history | (1 << 3), history)
        history = tl.where(surface_detected, history | (1 << 2), history)

        if LEGACY_SPECULAR_REFLECTION:
            reflected_dx, reflected_dy, reflected_dz = (
                physics_reflect_specular_chroma(
                    dx, dy, dz, nx, ny, nz
                )
            )
        else:
            reflected_dx, reflected_dy, reflected_dz = physics_reflect_specular(
                dx, dy, dz, nx, ny, nz
            )
        dx = tl.where(specular, reflected_dx, dx)
        dy = tl.where(specular, reflected_dy, dy)
        dz = tl.where(specular, reflected_dz, dz)
        history = tl.where(specular, history | (1 << 6), history)

        dx = tl.where(diffuse, diffuse_dx, dx)
        dy = tl.where(diffuse, diffuse_dy, dy)
        dz = tl.where(diffuse, diffuse_dz, dz)
        qx = tl.where(diffuse, diffuse_qx, qx)
        qy = tl.where(diffuse, diffuse_qy, qy)
        qz = tl.where(diffuse, diffuse_qz, qz)
        history = tl.where(diffuse, history | (1 << 5), history)

        if USE_RANDOM_TAPE:
            (
                fresnel_dx, fresnel_dy, fresnel_dz,
                fresnel_qx, fresnel_qy, fresnel_qz,
                fresnel_reflected, _, _, _,
            ) = fresnel_step_chroma(
                dx, dy, dz, qx, qy, qz, nx, ny, nz,
                refractive1, refractive2, dielectric_u0, dielectric_u1,
            )
        else:
            (
                fresnel_dx, fresnel_dy, fresnel_dz,
                fresnel_qx, fresnel_qy, fresnel_qz,
                fresnel_reflected, _, _, _,
            ) = physics_fresnel_step(
                dx, dy, dz, qx, qy, qz, nx, ny, nz,
                refractive1, refractive2, dielectric_u0, dielectric_u1,
            )
        dx = tl.where(dielectric, fresnel_dx, dx)
        dy = tl.where(dielectric, fresnel_dy, dy)
        dz = tl.where(dielectric, fresnel_dz, dz)
        qx = tl.where(dielectric, fresnel_qx, qx)
        qy = tl.where(dielectric, fresnel_qy, qy)
        qz = tl.where(dielectric, fresnel_qz, qz)
        history = tl.where(
            dielectric & fresnel_reflected, history | (1 << 6), history
        )

        if USE_RANDOM_TAPE:
            tape_draw_failure = (
                bulk_draw_failure
                | surface_selector_failure
                | diffuse_draw_failure
                | dielectric_draw_failure
            )
            tape_flags |= tl.where(tape_draw_failure, 1, 0)
            completed_interaction = (
                bulk_absorb
                | bulk_scatter
                | surface_absorbed
                | surface_detected
                | diffuse
                | specular
                | dielectric
            )
            decision = tl.full((BLOCK,), 0, tl.int32)
            decision = tl.where(bulk_absorb, 1, decision)
            decision = tl.where(bulk_scatter, 2, decision)
            decision = tl.where(surface_absorbed, 3, decision)
            decision = tl.where(surface_detected, 4, decision)
            decision = tl.where(diffuse, 5, decision)
            decision = tl.where(specular, 6, decision)
            decision = tl.where(
                dielectric & fresnel_reflected, 7, decision
            )
            decision = tl.where(
                dielectric & ~fresnel_reflected, 8, decision
            )

            semantic_draw_count = tl.zeros((BLOCK,), tl.int32)
            semantic_draw_count = tl.where(
                bulk_absorb_candidate, 2, semantic_draw_count
            )
            semantic_draw_count = tl.where(
                bulk_scatter_candidate, 4, semantic_draw_count
            )
            semantic_draw_count = tl.where(
                surface_absorbed | surface_detected | specular,
                3,
                semantic_draw_count,
            )
            diffuse_required_end = tl.where(
                diffuse_accepted, diffuse_cursor + 2, diffuse_cursor + 1
            )
            semantic_draw_count = tl.where(
                diffuse_candidate,
                diffuse_required_end - initial_tape_draw,
                semantic_draw_count,
            )
            semantic_draw_count = tl.where(
                dielectric_candidate,
                dielectric_draw_base + 2 - initial_tape_draw,
                semantic_draw_count,
            )
            semantic_draw_count = tl.where(
                surface_selector_failure, 3, semantic_draw_count
            )
            semantic_draw_count = tl.where(
                distance_draw_failure, 2, semantic_draw_count
            )
            certificate_commit = (
                completed_interaction
                & tape_row_valid
                & mapping_ok
                & interaction_ok
                & ~prior_tape_failure
                & ~tape_draw_failure
            )
            state_certificate_interaction = tape_interaction
            if CERTIFY_RANDOM_TAPE:
                certificate_process = decision.to(tl.uint32)
                certificate_word = (
                    (certificate_process << 28)
                    | semantic_draw_count.to(tl.uint32)
                )
                certificate_offset = (
                    tape_row * MAX_TAPE_INTERACTIONS + tape_interaction
                )
                tl.store(
                    tape_interaction_certificate + certificate_offset,
                    certificate_word,
                    mask=certificate_commit,
                )
            failed_tape_interaction = prior_tape_failure | tape_draw_failure
            history = tl.where(
                failed_tape_interaction,
                history | (1 << 0) | (1 << 15),
                history,
            )
            tape_draw = tl.where(
                tape_draw_failure,
                TAPE_DRAWS_PER_INTERACTION,
                tape_draw,
            )
            tape_interaction += completed_interaction.to(tl.int32)
            tape_draw = tl.where(completed_interaction, 0, tape_draw)
            tape_flags |= tl.where(
                completed_interaction
                & (tape_interaction >= MAX_TAPE_INTERACTIONS),
                2,
                0,
            )
            if TRACE_RANDOM_TAPE:
                trace_mask = tape_row_valid & has_hit
                tl.store(
                    tape_trace_interaction + tape_row,
                    tape_interaction - completed_interaction.to(tl.int32),
                    mask=trace_mask,
                )
                tl.store(
                    tape_trace_draw_count + tape_row,
                    semantic_draw_count,
                    mask=trace_mask,
                )
                tl.store(
                    tape_trace_decision + tape_row,
                    decision,
                    mask=trace_mask,
                )
                # An out-of-range row has nowhere in the row-indexed audit to
                # store its flag.  Full-simulation tape mode requires global
                # IDs 0..N-1, so use that expected row for the trace fallback.
                trace_overflow_row = tl.where(
                    tape_row_valid, tape_row, global_id
                )
                trace_overflow_valid = has_hit & (
                    trace_overflow_row >= 0
                ) & (trace_overflow_row < tape_photon_count)
                tl.store(
                    tape_trace_work_overflow + trace_overflow_row,
                    tape_flags,
                    mask=trace_overflow_valid,
                )
            tl.store(
                tape_interaction_cursor + tape_row,
                tape_interaction,
                mask=tape_row_valid,
            )
            tl.store(
                tape_draw_cursor + tape_row,
                tape_draw,
                mask=tape_row_valid,
            )
            tl.store(
                tape_overflow + tape_row,
                tape_flags,
                mask=tape_row_valid,
            )
        else:
            failed_tape_interaction = tl.zeros((BLOCK,), tl.int1)
            state_certificate_interaction = tl.zeros((BLOCK,), tl.int32)
            certificate_commit = tl.zeros((BLOCK,), tl.int1)

        no_hit = valid & ~has_hit
        history = tl.where(no_hit, history | (1 << 0), history)
        step_count += valid.to(tl.int32)
        active = (
            (bulk_scatter | diffuse | specular | dielectric)
            & ~failed_tape_interaction
            & (step_count < max_steps)
        )
        status = tl.where(active, 0, 1)  # STEP_ACTIVE / STEP_TERMINAL

        # Chroma clears the last triangle after a bulk interaction.  At a
        # boundary it retains the winner so the next query suppresses self-hit.
        new_last_instance = tl.where(at_boundary, hit_instance, -1)
        new_last_triangle = tl.where(at_boundary, hit_triangle, -1)
        if CERTIFY_STATE:
            # This store is deliberately adjacent to the final state writes,
            # after every process-specific position/direction/polarization,
            # time, history, and last-triangle mutation.  Bitcasts retain
            # signed zero, NaN payloads, and negative integer sentinels.
            state_offset = (
                (
                    tape_row * MAX_TAPE_INTERACTIONS
                    + state_certificate_interaction
                )
                * STATE_WORDS_PER_INTERACTION
            )
            tl.store(
                tape_state_certificate + state_offset + 0,
                px.to(tl.uint32, bitcast=True), mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 1,
                py.to(tl.uint32, bitcast=True), mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 2,
                pz.to(tl.uint32, bitcast=True), mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 3,
                dx.to(tl.uint32, bitcast=True), mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 4,
                dy.to(tl.uint32, bitcast=True), mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 5,
                dz.to(tl.uint32, bitcast=True), mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 6,
                qx.to(tl.uint32, bitcast=True), mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 7,
                qy.to(tl.uint32, bitcast=True), mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 8,
                qz.to(tl.uint32, bitcast=True), mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 9,
                tl.full((BLOCK,), 450.0, tl.float32).to(
                    tl.uint32, bitcast=True
                ),
                mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 10,
                photon_time.to(tl.uint32, bitcast=True),
                mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 11,
                history.to(tl.uint32, bitcast=True),
                mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 12,
                new_last_triangle.to(tl.uint32, bitcast=True),
                mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 13,
                tl.full((BLOCK,), 1.0, tl.float32).to(
                    tl.uint32, bitcast=True
                ),
                mask=certificate_commit,
            )
            tl.store(
                tape_state_certificate + state_offset + 14,
                tl.zeros((BLOCK,), tl.uint32),
                mask=certificate_commit,
            )
        tl.store(last_instances + photon_id, new_last_instance, mask=valid)
        tl.store(last_triangles + photon_id, new_last_triangle, mask=valid)
        tl.store(
            detected_channels + photon_id, hit_channel,
            mask=surface_detected & (hit_channel >= 0),
        )
        tl.store(positions + base, px, mask=valid)
        tl.store(positions + base + 1, py, mask=valid)
        tl.store(positions + base + 2, pz, mask=valid)
        tl.store(directions + base, dx, mask=valid)
        tl.store(directions + base + 1, dy, mask=valid)
        tl.store(directions + base + 2, dz, mask=valid)
        tl.store(polarizations + base, qx, mask=valid)
        tl.store(polarizations + base + 1, qy, mask=valid)
        tl.store(polarizations + base + 2, qz, mask=valid)
        tl.store(times + photon_id, photon_time, mask=valid)
        tl.store(histories + photon_id, history, mask=valid)
        tl.store(rng_counters + photon_id, rng_counter, mask=valid)
        tl.store(step_counts + photon_id, step_count, mask=valid)
        if DIRECT_QUEUE:
            flag = active.to(tl.int32)
            local_offset = tl.cumsum(flag, axis=0) - flag
            count = tl.sum(flag, axis=0)
            output_base = tl.atomic_add(output_count, count)
            tl.store(
                output_queue + output_base + local_offset,
                photon_id.to(tl.int32),
                mask=active,
            )
        else:
            tl.store(output_status + lane, status, mask=valid)

    _load_boundary_kernel._cached = (triton, boundary_step_kernel)
    return _load_boundary_kernel._cached


def _load_production_boundary_kernels():
    """Load the production classifier and branch-specific physics workers.

    Triton's masked ``where`` expressions do not make expensive vector math
    conditional at lane granularity: both sides are normally emitted.  The
    legacy monolithic boundary kernel consequently evaluated Rayleigh,
    diffuse, specular, and Fresnel alternatives before choosing one result.
    This production-only path first classifies with cheap scalar work, then
    launches separate compact-queue variants whose constexpr branch removes
    every unrelated physics implementation at compile time.
    """

    cached = getattr(_load_production_boundary_kernels, "_cached", None)
    if cached is not None:
        return cached
    try:
        import triton
        import triton.language as tl
        from chroma.triton.physics_kernels import (
            fresnel_step as physics_fresnel_step,
            rayleigh_scatter as physics_rayleigh_scatter,
            reflect_specular as physics_reflect_specular,
            sample_bulk_collision as physics_sample_bulk_collision,
        )
    except ImportError as exc:  # pragma: no cover - optional installation
        raise RuntimeError("the specialized backend requires torch and Triton") from exc

    globals().update(
        triton=triton,
        tl=tl,
        physics_fresnel_step=physics_fresnel_step,
        physics_rayleigh_scatter=physics_rayleigh_scatter,
        physics_reflect_specular=physics_reflect_specular,
        physics_sample_bulk_collision=physics_sample_bulk_collision,
    )

    @triton.jit(do_not_specialize=[29, 32, 33, 34, 35])
    def production_boundary_classify_kernel(
            positions,
            directions,
            polarizations,
            times,
            histories,
            rng_counters,
            last_instances,
            last_triangles,
            detected_channels,
            step_counts,
            input_queue,
            hit_distances,
            hit_normals,
            material_from,
            material_to,
            surface_indices,
            hit_instances,
            hit_triangles,
            hit_channels,
            material_refractive_index,
            material_absorption_length,
            material_scattering_length,
            surface_detect,
            surface_absorb,
            surface_reflect_diffuse,
            surface_reflect_specular,
            global_photon_ids,
            branch_rows,
            branch_counts,
            branch_capacity,
            survivor_queue,
            survivor_count,
            nitems,
            seed,
            photon_id_base,
            max_steps,
            USE_GLOBAL_IDS: tl.constexpr,
            BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < nitems
        photon_id = tl.load(input_queue + lane, mask=valid, other=0).to(tl.int64)
        base = photon_id * 3
        px = tl.load(positions + base, mask=valid, other=0.0)
        py = tl.load(positions + base + 1, mask=valid, other=0.0)
        pz = tl.load(positions + base + 2, mask=valid, other=0.0)
        dx = tl.load(directions + base, mask=valid, other=1.0)
        dy = tl.load(directions + base + 1, mask=valid, other=0.0)
        dz = tl.load(directions + base + 2, mask=valid, other=0.0)
        photon_time = tl.load(times + photon_id, mask=valid, other=0.0)
        history = tl.load(histories + photon_id, mask=valid, other=0).to(tl.int32)
        rng_counter = tl.load(
            rng_counters + photon_id, mask=valid, other=0
        ).to(tl.int64)
        step_count = tl.load(
            step_counts + photon_id, mask=valid, other=0
        ).to(tl.int32)

        distance_to_boundary = tl.load(
            hit_distances + lane, mask=valid, other=float("inf")
        )
        nx = tl.load(hit_normals + lane * 3, mask=valid, other=1.0)
        ny = tl.load(hit_normals + lane * 3 + 1, mask=valid, other=0.0)
        nz = tl.load(hit_normals + lane * 3 + 2, mask=valid, other=0.0)
        from_index = tl.load(
            material_from + lane, mask=valid, other=-1
        ).to(tl.int32)
        surface_index = tl.load(
            surface_indices + lane, mask=valid, other=-1
        ).to(tl.int32)
        hit_instance = tl.load(
            hit_instances + lane, mask=valid, other=-1
        ).to(tl.int32)
        hit_triangle = tl.load(
            hit_triangles + lane, mask=valid, other=-1
        ).to(tl.int32)
        hit_channel = tl.load(
            hit_channels + lane, mask=valid, other=-1
        ).to(tl.int32)
        has_hit = valid & (from_index >= 0) & (
            distance_to_boundary < float("inf")
        )

        safe_from = tl.maximum(from_index, 0)
        refractive1 = tl.load(
            material_refractive_index + safe_from, mask=has_hit, other=1.0
        )
        absorption_length = tl.load(
            material_absorption_length + safe_from,
            mask=has_hit,
            other=float("inf"),
        )
        scattering_length = tl.load(
            material_scattering_length + safe_from,
            mask=has_hit,
            other=float("inf"),
        )

        if USE_GLOBAL_IDS:
            global_id = tl.load(
                global_photon_ids + photon_id, mask=valid, other=0
            ).to(tl.int64)
        else:
            global_id = photon_id + photon_id_base.to(tl.int64)
        random_offset = global_id * 4294967296 + rng_counter
        u0, u1, u2, _ = tl.rand4x(seed, random_offset)
        rng_counter += has_hit.to(tl.int64) * 2

        collision_distance, process = physics_sample_bulk_collision(
            absorption_length, scattering_length, u0, u1
        )
        zero_absorption = has_hit & (absorption_length <= 0.0)
        collision_distance = tl.where(
            zero_absorption, 0.0, collision_distance
        )
        process = tl.where(zero_absorption, 1, process)
        bulk_collision = has_hit & (process != 0) & (
            collision_distance <= distance_to_boundary
        )
        bulk_absorb = bulk_collision & (process == 1)
        bulk_scatter = bulk_collision & (process == 2)
        at_boundary = has_hit & ~bulk_collision

        advance = tl.where(
            bulk_collision,
            collision_distance,
            tl.where(at_boundary, distance_to_boundary, 0.0),
        )
        px += advance * dx
        py += advance * dy
        pz += advance * dz
        photon_time += advance * refractive1 / 299.792458

        has_surface = at_boundary & (surface_index >= 0)
        safe_surface = tl.maximum(surface_index, 0)
        probability_absorb = tl.load(
            surface_absorb + safe_surface, mask=has_surface, other=0.0
        )
        probability_detect = tl.load(
            surface_detect + safe_surface, mask=has_surface, other=0.0
        )
        probability_diffuse = tl.load(
            surface_reflect_diffuse + safe_surface,
            mask=has_surface,
            other=0.0,
        )
        probability_specular = tl.load(
            surface_reflect_specular + safe_surface,
            mask=has_surface,
            other=0.0,
        )
        cut_absorb = probability_absorb
        cut_detect = cut_absorb + probability_detect
        cut_diffuse = cut_detect + probability_diffuse
        cut_specular = cut_diffuse + probability_specular
        surface_absorbed = has_surface & (u2 < cut_absorb)
        surface_detected = has_surface & (u2 >= cut_absorb) & (
            u2 < cut_detect
        )
        diffuse = has_surface & (u2 >= cut_detect) & (u2 < cut_diffuse)
        specular = has_surface & (u2 >= cut_diffuse) & (
            u2 < cut_specular
        )
        dielectric = at_boundary & (
            ~has_surface | (u2 >= cut_specular)
        )

        history = tl.where(bulk_absorb, history | (1 << 1), history)
        # The process bit is independent of its sampled direction and can be
        # committed by the classifier before the Rayleigh worker runs.
        history = tl.where(bulk_scatter, history | (1 << 4), history)
        history = tl.where(surface_absorbed, history | (1 << 3), history)
        history = tl.where(surface_detected, history | (1 << 2), history)

        # Specular reflection is six FMAs/multiplies and dominates steel
        # surface traffic, so finishing it here is cheaper than another queue.
        reflected_dx, reflected_dy, reflected_dz = physics_reflect_specular(
            dx, dy, dz, nx, ny, nz
        )
        dx = tl.where(specular, reflected_dx, dx)
        dy = tl.where(specular, reflected_dy, dy)
        dz = tl.where(specular, reflected_dz, dz)
        history = tl.where(specular, history | (1 << 6), history)

        no_hit = valid & ~has_hit
        history = tl.where(no_hit, history | (1 << 0), history)
        step_count += valid.to(tl.int32)
        new_last_instance = tl.where(at_boundary, hit_instance, -1)
        new_last_triangle = tl.where(at_boundary, hit_triangle, -1)

        tl.store(positions + base, px, mask=valid)
        tl.store(positions + base + 1, py, mask=valid)
        tl.store(positions + base + 2, pz, mask=valid)
        tl.store(directions + base, dx, mask=valid)
        tl.store(directions + base + 1, dy, mask=valid)
        tl.store(directions + base + 2, dz, mask=valid)
        tl.store(times + photon_id, photon_time, mask=valid)
        tl.store(histories + photon_id, history, mask=valid)
        tl.store(rng_counters + photon_id, rng_counter, mask=valid)
        tl.store(step_counts + photon_id, step_count, mask=valid)
        tl.store(last_instances + photon_id, new_last_instance, mask=valid)
        tl.store(last_triangles + photon_id, new_last_triangle, mask=valid)
        tl.store(
            detected_channels + photon_id,
            hit_channel,
            mask=surface_detected & (hit_channel >= 0),
        )

        # Compact boundary-row indices.  Workers can then recover the photon
        # slot from input_queue and branch-specific hit data from the row.
        scatter_flag = bulk_scatter.to(tl.int32)
        scatter_local = tl.cumsum(scatter_flag, axis=0) - scatter_flag
        scatter_count = tl.sum(scatter_flag, axis=0)
        scatter_base = tl.atomic_add(branch_counts, scatter_count)
        tl.store(
            branch_rows + scatter_base + scatter_local,
            lane.to(tl.int32),
            mask=bulk_scatter,
        )

        diffuse_flag = diffuse.to(tl.int32)
        diffuse_local = tl.cumsum(diffuse_flag, axis=0) - diffuse_flag
        diffuse_count = tl.sum(diffuse_flag, axis=0)
        diffuse_base = tl.atomic_add(branch_counts + 1, diffuse_count)
        tl.store(
            branch_rows + branch_capacity + diffuse_base + diffuse_local,
            lane.to(tl.int32),
            mask=diffuse,
        )

        dielectric_flag = dielectric.to(tl.int32)
        dielectric_local = tl.cumsum(dielectric_flag, axis=0) - dielectric_flag
        dielectric_count = tl.sum(dielectric_flag, axis=0)
        dielectric_base = tl.atomic_add(branch_counts + 2, dielectric_count)
        tl.store(
            branch_rows + 2 * branch_capacity + dielectric_base + dielectric_local,
            lane.to(tl.int32),
            mask=dielectric,
        )

        specular_active = specular & (step_count < max_steps)
        survivor_flag = specular_active.to(tl.int32)
        survivor_local = tl.cumsum(survivor_flag, axis=0) - survivor_flag
        survivor_n = tl.sum(survivor_flag, axis=0)
        survivor_base = tl.atomic_add(survivor_count, survivor_n)
        tl.store(
            survivor_queue + survivor_base + survivor_local,
            photon_id.to(tl.int32),
            mask=specular_active,
        )

    # BRANCH is 0=bulk Rayleigh, 1=diffuse reflection, 2=dielectric Fresnel.
    @triton.jit(do_not_specialize=[15, 16, 17, 18, 19])
    def production_boundary_branch_kernel(
            directions,
            polarizations,
            histories,
            rng_counters,
            step_counts,
            input_queue,
            hit_normals,
            material_from,
            material_to,
            material_refractive_index,
            global_photon_ids,
            branch_rows,
            branch_counts,
            survivor_queue,
            survivor_count,
            branch_capacity,
            launch_capacity,
            seed,
            photon_id_base,
            max_steps,
            BRANCH: tl.constexpr,
            USE_GLOBAL_IDS: tl.constexpr,
            BLOCK: tl.constexpr):
        block_start = tl.program_id(0) * BLOCK
        branch_count = tl.load(branch_counts + BRANCH).to(tl.int32)
        # This condition is uniform across the program and lowers to one CTA
        # branch.  Merely masking lanes is insufficient: Triton would still
        # execute the selected branch's transcendental math in every launched
        # program.  Empty suffix CTAs return after one counter load.
        if block_start >= branch_count:
            return
        lane = block_start + tl.arange(0, BLOCK)
        valid = (lane < launch_capacity) & (lane < branch_count)
        boundary_row = tl.load(
            branch_rows + BRANCH * branch_capacity + lane,
            mask=valid,
            other=0,
        ).to(tl.int64)
        photon_id = tl.load(
            input_queue + boundary_row, mask=valid, other=0
        ).to(tl.int64)
        base = photon_id * 3
        dx = tl.load(directions + base, mask=valid, other=1.0)
        dy = tl.load(directions + base + 1, mask=valid, other=0.0)
        dz = tl.load(directions + base + 2, mask=valid, other=0.0)
        qx = tl.load(polarizations + base, mask=valid, other=0.0)
        qy = tl.load(polarizations + base + 1, mask=valid, other=1.0)
        qz = tl.load(polarizations + base + 2, mask=valid, other=0.0)
        history = tl.load(
            histories + photon_id, mask=valid, other=0
        ).to(tl.int32)
        rng_counter = tl.load(
            rng_counters + photon_id, mask=valid, other=2
        ).to(tl.int64)
        step_count = tl.load(
            step_counts + photon_id, mask=valid, other=max_steps
        ).to(tl.int32)
        if USE_GLOBAL_IDS:
            global_id = tl.load(
                global_photon_ids + photon_id, mask=valid, other=0
            ).to(tl.int64)
        else:
            global_id = photon_id + photon_id_base.to(tl.int64)
        random_offset = global_id * 4294967296 + (rng_counter - 2)

        if BRANCH == 0:
            _, _, u2, u3 = tl.rand4x(seed, random_offset)
            u4, u5, _, _ = tl.rand4x(seed, random_offset + 1)
            dx, dy, dz, qx, qy, qz = physics_rayleigh_scatter(
                dx, dy, dz, qx, qy, qz, u2, u3, u4, u5
            )
        elif BRANCH == 1:
            _, _, _, u3 = tl.rand4x(seed, random_offset)
            u4, u5, _, _ = tl.rand4x(seed, random_offset + 1)
            nx = tl.load(
                hit_normals + boundary_row * 3, mask=valid, other=1.0
            )
            ny = tl.load(
                hit_normals + boundary_row * 3 + 1, mask=valid, other=0.0
            )
            nz = tl.load(
                hit_normals + boundary_row * 3 + 2, mask=valid, other=0.0
            )
            use_xy = tl.abs(nz) < 0.9
            b1x = tl.where(use_xy, -ny, 0.0)
            b1y = tl.where(use_xy, nx, -nz)
            b1z = tl.where(use_xy, 0.0, ny)
            inv_b1 = tl.rsqrt(tl.maximum(
                b1x * b1x + b1y * b1y + b1z * b1z, 1.0e-20
            ))
            b1x, b1y, b1z = b1x * inv_b1, b1y * inv_b1, b1z * inv_b1
            b2x = ny * b1z - nz * b1y
            b2y = nz * b1x - nx * b1z
            b2z = nx * b1y - ny * b1x
            radial = tl.sqrt(u3)
            axial = tl.sqrt(tl.maximum(0.0, 1.0 - u3))
            phi = 6.283185307179586 * u4
            tangent_cos = radial * tl.cos(phi)
            tangent_sin = radial * tl.sin(phi)
            diffuse_dx = axial * nx + tangent_cos * b1x + tangent_sin * b2x
            diffuse_dy = axial * ny + tangent_cos * b1y + tangent_sin * b2y
            diffuse_dz = axial * nz + tangent_cos * b1z + tangent_sin * b2z
            pol_use_xy = tl.abs(diffuse_dz) < 0.9
            c1x = tl.where(pol_use_xy, -diffuse_dy, 0.0)
            c1y = tl.where(pol_use_xy, diffuse_dx, -diffuse_dz)
            c1z = tl.where(pol_use_xy, 0.0, diffuse_dy)
            inv_c1 = tl.rsqrt(tl.maximum(
                c1x * c1x + c1y * c1y + c1z * c1z, 1.0e-20
            ))
            c1x, c1y, c1z = c1x * inv_c1, c1y * inv_c1, c1z * inv_c1
            c2x = diffuse_dy * c1z - diffuse_dz * c1y
            c2y = diffuse_dz * c1x - diffuse_dx * c1z
            c2z = diffuse_dx * c1y - diffuse_dy * c1x
            pol_phi = 6.283185307179586 * u5
            diffuse_qx = tl.cos(pol_phi) * c1x + tl.sin(pol_phi) * c2x
            diffuse_qy = tl.cos(pol_phi) * c1y + tl.sin(pol_phi) * c2y
            diffuse_qz = tl.cos(pol_phi) * c1z + tl.sin(pol_phi) * c2z
            dx, dy, dz = diffuse_dx, diffuse_dy, diffuse_dz
            qx, qy, qz = diffuse_qx, diffuse_qy, diffuse_qz
            history |= 1 << 5
        else:
            _, _, u6, u7 = tl.rand4x(seed, random_offset + 1)
            nx = tl.load(
                hit_normals + boundary_row * 3, mask=valid, other=1.0
            )
            ny = tl.load(
                hit_normals + boundary_row * 3 + 1, mask=valid, other=0.0
            )
            nz = tl.load(
                hit_normals + boundary_row * 3 + 2, mask=valid, other=0.0
            )
            from_index = tl.load(
                material_from + boundary_row, mask=valid, other=0
            ).to(tl.int32)
            to_index = tl.load(
                material_to + boundary_row, mask=valid, other=0
            ).to(tl.int32)
            refractive1 = tl.load(
                material_refractive_index + tl.maximum(from_index, 0),
                mask=valid,
                other=1.0,
            )
            refractive2 = tl.load(
                material_refractive_index + tl.maximum(to_index, 0),
                mask=valid,
                other=1.0,
            )
            (
                dx, dy, dz, qx, qy, qz, reflected, _, _, _,
            ) = physics_fresnel_step(
                dx, dy, dz, qx, qy, qz, nx, ny, nz,
                refractive1, refractive2, u6, u7,
            )
            history = tl.where(reflected, history | (1 << 6), history)

        tl.store(directions + base, dx, mask=valid)
        tl.store(directions + base + 1, dy, mask=valid)
        tl.store(directions + base + 2, dz, mask=valid)
        tl.store(polarizations + base, qx, mask=valid)
        tl.store(polarizations + base + 1, qy, mask=valid)
        tl.store(polarizations + base + 2, qz, mask=valid)
        tl.store(histories + photon_id, history, mask=valid)

        active = valid & (step_count < max_steps)
        flag = active.to(tl.int32)
        local_offset = tl.cumsum(flag, axis=0) - flag
        count = tl.sum(flag, axis=0)
        output_base = tl.atomic_add(survivor_count, count)
        tl.store(
            survivor_queue + output_base + local_offset,
            photon_id.to(tl.int32),
            mask=active,
        )

    _load_production_boundary_kernels._cached = (
        triton,
        production_boundary_classify_kernel,
        production_boundary_branch_kernel,
    )
    return _load_production_boundary_kernels._cached


def _load_boundary_gather_kernel():
    cached = getattr(_load_boundary_gather_kernel, "_cached", None)
    if cached is not None:
        return cached
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - optional installation
        raise RuntimeError("the specialized backend requires torch and Triton") from exc
    globals().update(triton=triton, tl=tl)

    @triton.jit(do_not_specialize=[9])
    def gather_boundary_rays(
            positions,
            directions,
            last_instances,
            last_triangles,
            queue,
            out_origins,
            out_directions,
            out_last_instances,
            out_last_triangles,
            nitems,
            BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < nitems
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

    _load_boundary_gather_kernel._cached = (triton, gather_boundary_rays)
    return _load_boundary_gather_kernel._cached


class Reflect3WiresTritonSimulation:
    """End-to-end wavefront simulator for the production 450-nm photon bomb.

    Memory-sized source slabs execute a dense prefix independently.  Their
    surviving photon state is then compacted into one cross-slab reservoir so
    rare long histories are drained once rather than once per slab.  Set
    ``reservoir_rounds=None`` to retain the legacy independently-drained path.

    With ``tile_size=None``, the simulator queries allocatable CUDA memory at
    the beginning of every call and applies the conservative module-level
    footprint policy.  ``auto_memory_bytes`` overrides only that query, which
    is useful for reproducible planning tests.  A positive explicit
    ``tile_size`` remains a hard photon-count cap and bypasses memory planning.
    """

    def __init__(
        self,
        *,
        scene=None,
        device="cuda",
        tile_size=None,
        auto_memory_bytes=None,
        history_length=8,
        block_size=128,
        reservoir_rounds=128,
        history_epochs_per_poll=2,
        legacy_specular_reflection=False,
        chroma_mesh_box_compatibility=False,
        chroma_global_bvh_artifact=None,
        branch_specialized_boundary=False,
        portal_boundary=False,
        fused_portal_boundary=False,
        pmt_grid=False,
        fused_pmt=False,
        fused_pmt_max_candidates=81,
        fused_pmt_compact_union=True,
        fused_pmt_routing_diagnostics=False,
        device_scheduler=False,
        device_round_batch=8,
        geometry_only=False,
    ):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("an NVIDIA CUDA device is required")
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("the Triton backend requires a CUDA device")
        exact_compatibility = bool(chroma_mesh_box_compatibility)
        if exact_compatibility and chroma_global_bvh_artifact is None:
            raise ValueError(
                "exact Chroma compatibility requires an explicit "
                "chroma_global_bvh_artifact path or artifact object"
            )
        if not exact_compatibility and chroma_global_bvh_artifact is not None:
            raise ValueError(
                "chroma_global_bvh_artifact is valid only when "
                "chroma_mesh_box_compatibility=True"
            )
        self.scene = scene or compile_reflect3wires_scene(
            source_x_sign=-1,
            wavelength_nm=450.0,
            retain_all_wires=exact_compatibility,
        )
        self.scene.validate()
        self.geometry_only = bool(geometry_only)
        if not self.geometry_only and abs(float(self.scene.wavelength_nm) - 450.0) > 1.0e-6:
            raise ValueError("production specialization is compiled only at 450 nm")
        if np.any(self.scene.tables.material_num_reemission_components != 0):
            raise ValueError("bulk re-emission requires the general Chroma backend")
        if not self.geometry_only and np.any(self.scene.tables.surface_models != 0):
            raise ValueError("target specialization supports Chroma default surfaces only")
        self.tile_size = None if tile_size is None else int(tile_size)
        self.auto_memory_bytes = (
            None if auto_memory_bytes is None else int(auto_memory_bytes)
        )
        self.history_length = int(history_length)
        self.block_size = int(block_size)
        self.reservoir_rounds = (
            None if reservoir_rounds is None else int(reservoir_rounds)
        )
        self.history_epochs_per_poll = int(history_epochs_per_poll)
        self.legacy_specular_reflection = bool(legacy_specular_reflection)
        self.chroma_mesh_box_compatibility = exact_compatibility
        self.branch_specialized_boundary = bool(branch_specialized_boundary)
        self.portal_boundary = bool(portal_boundary)
        self.fused_portal_boundary = bool(fused_portal_boundary)
        self.pmt_grid = bool(pmt_grid)
        self.fused_pmt = bool(fused_pmt)
        self.fused_pmt_max_candidates = int(fused_pmt_max_candidates)
        self.fused_pmt_compact_union = bool(fused_pmt_compact_union)
        self.fused_pmt_routing_diagnostics = bool(
            fused_pmt_routing_diagnostics
        )
        self.fused_pmt_routing_counters = None
        self.device_scheduler = bool(device_scheduler)
        self.device_round_batch = int(device_round_batch)
        fused_incompatibilities = []
        if self.chroma_mesh_box_compatibility:
            fused_incompatibilities.append("chroma_mesh_box_compatibility")
        if self.legacy_specular_reflection:
            fused_incompatibilities.append("legacy_specular_reflection")
        if self.pmt_grid:
            fused_incompatibilities.append("pmt_grid")
        if self.fused_pmt and fused_incompatibilities:
            raise ValueError(
                "fused_pmt is corrected-production-only and cannot be "
                "combined with " + ", ".join(fused_incompatibilities)
            )
        if self.fused_portal_boundary and not self.device_scheduler:
            raise ValueError(
                "fused_portal_boundary=True requires device_scheduler=True"
            )
        if self.fused_portal_boundary and not self.portal_boundary:
            raise ValueError(
                "fused_portal_boundary=True requires portal_boundary=True"
            )
        if self.fused_portal_boundary and not (
            _production_fused_portal_boundary_enabled(
                True,
                device_scheduler=self.device_scheduler,
                portal_boundary=self.portal_boundary,
                use_random_tape=False,
                legacy_specular_reflection=self.legacy_specular_reflection,
                chroma_mesh_box_compatibility=(
                    self.chroma_mesh_box_compatibility
                ),
                branch_specialized_boundary=(
                    self.branch_specialized_boundary
                ),
                pmt_grid=self.pmt_grid,
                fused_pmt=self.fused_pmt,
            )
        ):
            raise ValueError(
                "fused_portal_boundary is corrected-production-only and "
                "cannot be combined with branch-specialized, PMT-grid, "
                "legacy-specular, or Chroma-global compatibility modes"
            )
        if self.fused_pmt_max_candidates <= 0:
            raise ValueError("fused_pmt_max_candidates must be positive")
        if not self.fused_pmt and self.fused_pmt_max_candidates != 81:
            raise ValueError(
                "fused_pmt_max_candidates requires fused_pmt=True"
            )
        if not self.fused_pmt and not self.fused_pmt_compact_union:
            raise ValueError("fused_pmt_compact_union=False requires fused_pmt=True")
        if self.fused_pmt_routing_diagnostics and not self.fused_pmt:
            raise ValueError(
                "fused_pmt_routing_diagnostics requires fused_pmt=True"
            )
        if self.chroma_mesh_box_compatibility and not np.array_equal(
            self.scene.wires.source_wireplane_index,
            np.arange(6, dtype=np.int32),
        ):
            raise ValueError(
                "exact Chroma compatibility requires a scene compiled with "
                "retain_all_wires=True (source wire-plane indices 0..5)"
            )
        if self.tile_size is not None and self.tile_size <= 0:
            raise ValueError("tile_size must be positive or None")
        if self.auto_memory_bytes is not None and self.auto_memory_bytes < 0:
            raise ValueError("auto_memory_bytes must be non-negative or None")
        if self.history_length <= 0:
            raise ValueError("history_length must be positive")
        if self.reservoir_rounds is not None and self.reservoir_rounds <= 0:
            raise ValueError("reservoir_rounds must be positive or None")
        if self.history_epochs_per_poll <= 0:
            raise ValueError("history_epochs_per_poll must be positive")
        if self.device_round_batch <= 0:
            raise ValueError("device_round_batch must be positive")
        self.scene_device = self.scene.to_torch(self.device)
        # Boundary workspaces are grown to the first observed queue.  Eagerly
        # sizing these to the auto limit would reserve hundreds of MB even for
        # a smoke test, and PMT pair storage is much sparser than ray storage.
        self.merge_workspace = _BoundaryMergeWorkspace.allocate(
            torch, self.device, 0
        )
        self.boundary_ray_workspace = _BoundaryRayWorkspace.allocate(
            torch, self.device, 0
        )
        from .triton_scene.intersect import prepare_scene_triton
        self.analytic_scene = prepare_scene_triton(self.scene, self.device)
        from .triton_scene.intersect import allocate_split_intersection_workspace
        self.analytic_workspace = allocate_split_intersection_workspace(
            0, self.device
        )
        self.safe_lower, self.safe_upper = _safe_empty_bounds(self.scene)

        if self.chroma_mesh_box_compatibility:
            from .triton_scene import (
                TARGET_OPTICAL_SEMANTICS_SHA256,
                TARGET_TRAVERSAL_SHA256,
                ChromaGlobalBVHArtifact,
                ChromaGlobalBVHDevice,
                load_chroma_global_bvh_artifact,
            )
            from .triton_scene.chroma_global_bvh import TARGET_MESH_MD5

            if isinstance(chroma_global_bvh_artifact, ChromaGlobalBVHArtifact):
                host_artifact = chroma_global_bvh_artifact
            elif isinstance(chroma_global_bvh_artifact, (str, bytes)) or hasattr(
                chroma_global_bvh_artifact, "__fspath__"
            ):
                host_artifact = load_chroma_global_bvh_artifact(
                    chroma_global_bvh_artifact,
                    expected_mesh_md5=TARGET_MESH_MD5,
                    expected_traversal_sha256=TARGET_TRAVERSAL_SHA256,
                    expected_optical_semantics_sha256=(
                        TARGET_OPTICAL_SEMANTICS_SHA256
                    ),
                )
            else:
                raise TypeError(
                    "chroma_global_bvh_artifact must be a path or "
                    "ChromaGlobalBVHArtifact"
                )
            host_artifact.validate()
            if host_artifact.mesh_md5 != TARGET_MESH_MD5:
                raise ValueError(
                    "exact Chroma compatibility artifact has the wrong "
                    f"mesh MD5: {host_artifact.mesh_md5}"
                )
            if (
                host_artifact.optical_semantics_sha256
                != TARGET_OPTICAL_SEMANTICS_SHA256
            ):
                raise ValueError(
                    "exact Chroma compatibility artifact has the wrong "
                    "optical semantics SHA-256: "
                    f"{host_artifact.optical_semantics_sha256}"
                )
            self.chroma_global_host_artifact = host_artifact
            self.chroma_global_accelerator = ChromaGlobalBVHDevice.from_host(
                host_artifact,
                device=self.device,
                expected_traversal_sha256=TARGET_TRAVERSAL_SHA256,
                target_material_names=self.scene.tables.material_names,
                target_surface_names=self.scene.tables.surface_names,
            )
            self.chroma_global_workspace = (
                self.chroma_global_accelerator.allocate_workspace(0)
            )
            # The exact global mesh supersedes both analytic macro boxes and
            # the production shared-PMT acceleration structure.
            self.pmt_accelerator = None
            self.pmt_workspace = None
            self.portal_descriptor = None
            self.portal_workspace = None
            self.fused_portal_boundary_workspace = None
        else:
            self.chroma_global_host_artifact = None
            self.chroma_global_accelerator = None
            self.chroma_global_workspace = None
            from .triton_scene.instances import build_pmt_instance_accelerator

            self.pmt_accelerator = build_pmt_instance_accelerator(
                self.scene,
                device=self.device,
                chroma_world_compatibility=False,
            )
            if self.fused_pmt and self.pmt_accelerator.grid_locator is None:
                raise ValueError(
                    "fused_pmt requires a compiler-certified regular PMT lattice"
                )
            self.pmt_workspace = self.pmt_accelerator.allocate_workspace(
                0, candidate_capacity=0
            )
            if self.fused_pmt_routing_diagnostics:
                from .triton_scene.fused_pmt import FusedPMTRoutingCounters

                self.fused_pmt_routing_counters = (
                    FusedPMTRoutingCounters.allocate(self.device)
                )
            from .triton_scene.portals import (
                PortalWorkspace,
                certify_reflect3wires_portals,
            )
            # Re-prove the empty-box portals against the accelerator's actual
            # outward-padded PMT bounds at construction time.  A geometry or
            # broadphase-policy change therefore fails closed before launch.
            self.portal_descriptor = None if self.geometry_only else certify_reflect3wires_portals(
                self.scene,
                self.safe_lower,
                self.safe_upper,
                padded_pmt_bounds_max=(
                    self.pmt_accelerator.host_bounds_max
                ),
            )
            self.portal_workspace = PortalWorkspace.allocate(0, self.device)
            if self.fused_portal_boundary:
                from .triton_scene.fused_portal_boundary import (
                    FusedPortalBoundaryWorkspace,
                )

                self.fused_portal_boundary_workspace = (
                    FusedPortalBoundaryWorkspace.allocate(0, self.device)
                )
            else:
                self.fused_portal_boundary_workspace = None
        # Boundary physics writes survivors directly into this reusable queue;
        # no status array or follow-up compaction kernel is required.
        self.survivor_queue_buffer = torch.empty(
            0, dtype=torch.int32, device=self.device
        )
        self.survivor_queue_count = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )
        self.boundary_branch_workspace = _BoundaryBranchWorkspace.allocate(
            torch, self.device, 0
        )
        self.collision_queue_workspaces = None
        self.collision_boundary_accumulator = None
        self.device_pending_count = None
        self.device_boundary_ray_workspace = None
        self.device_boundary_merge_workspace = None
        self.device_round_trace = None
        self.device_continuation_counts = None
        self.last_device_scheduler_syncs = 0
        self.last_portal_counts = (0, 0)
        self.last_tile_plan = None

        material_names = self.scene.tables.material_names
        self.lar_index = material_names.index("liquid_argon")
        self.lar_refractive_index = float(
            self.scene.tables.material_refractive_index[self.lar_index]
        )
        self.lar_absorption_length = float(
            self.scene.tables.material_absorption_length[self.lar_index]
        )
        self.lar_scattering_length = float(
            self.scene.tables.material_scattering_length[self.lar_index]
        )

    def _merge_boundaries(self, analytic, pmt, directions):
        """Merge winners in one launch and return reusable output views."""

        count = int(directions.shape[0])
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("directions must have shape [N,3]")
        if (
            analytic.distance.shape != (count,)
            or pmt.distances.shape != (count,)
        ):
            raise ValueError("analytic, PMT, and direction counts must match")
        if count > self.merge_workspace.capacity:
            grown = max(count, max(1, 2 * self.merge_workspace.capacity))
            self.merge_workspace = _BoundaryMergeWorkspace.allocate(
                self.torch, self.device, grown
            )
        outputs = self.merge_workspace.outputs(count)
        if count == 0:
            return outputs

        triton, kernel = _load_boundary_merge_kernel()
        block = 256
        kernel[(triton.cdiv(count, block),)](
            analytic.distance,
            analytic.kind,
            analytic.index,
            analytic.primitive_index,
            analytic.surface_normal,
            analytic.material_from_index,
            analytic.material_to_index,
            analytic.surface_index,
            pmt.distances,
            pmt.world_normals,
            pmt.instance_ids,
            pmt.triangle_ids,
            pmt.channel_ids,
            directions,
            self.scene_device["pmt_scene_material1_index"],
            self.scene_device["pmt_scene_material2_index"],
            self.scene_device["pmt_scene_surface_index"],
            *outputs,
            count,
            BLOCK=block,
            num_warps=8,
        )
        return outputs

    def _merge_chroma_global_boundaries(self, analytic, mesh):
        """Merge exact global-mesh and analytic-wire certificate results."""

        count = int(mesh.distances.shape[0])
        if analytic.distance.shape != (count,):
            raise ValueError("analytic and global-mesh result counts must match")
        if count > self.merge_workspace.capacity:
            grown = max(count, max(1, 2 * self.merge_workspace.capacity))
            self.merge_workspace = _BoundaryMergeWorkspace.allocate(
                self.torch, self.device, grown
            )
        outputs = self.merge_workspace.outputs(count)
        if count == 0:
            return outputs

        triton, kernel = _load_chroma_global_merge_kernel()
        block = 256
        kernel[(triton.cdiv(count, block),)](
            analytic.distance,
            analytic.kind,
            analytic.surface_normal,
            analytic.material_from_index,
            analytic.material_to_index,
            analytic.surface_index,
            mesh.distances,
            mesh.surface_normals,
            mesh.material_from_indices,
            mesh.material_to_indices,
            mesh.surface_indices,
            mesh.triangle_ids,
            mesh.channel_ids,
            *outputs,
            count,
            BLOCK=block,
            num_warps=8,
        )
        return outputs

    def _ensure_boundary_workspace(self, count):
        """Lazily size fixed-per-ray scratch to an observed boundary queue."""

        count = int(count)
        if count < 0:
            raise ValueError("boundary queue size cannot be negative")
        if count > self.analytic_workspace.capacity:
            from .triton_scene.intersect import (
                allocate_split_intersection_workspace,
            )
            self.analytic_workspace = allocate_split_intersection_workspace(
                count, self.device
            )
        if count > self.boundary_ray_workspace.capacity:
            self.boundary_ray_workspace = _BoundaryRayWorkspace.allocate(
                self.torch, self.device, count
            )
        if self.chroma_mesh_box_compatibility:
            if count > self.chroma_global_workspace.ray_capacity:
                previous_sticky = self.chroma_global_workspace.sticky_overflow
                grown_workspace = (
                    self.chroma_global_accelerator.allocate_workspace(count)
                )
                grown_workspace.sticky_overflow.bitwise_or_(previous_sticky)
                self.chroma_global_workspace = grown_workspace
        else:
            self.pmt_workspace.ensure_ray_capacity(count)

    def _ensure_survivor_queue(self, count):
        count = int(count)
        if count > self.survivor_queue_buffer.numel():
            self.survivor_queue_buffer = self.torch.empty(
                count, dtype=self.torch.int32, device=self.device
            )

    def _ensure_boundary_branch_workspace(self, count):
        """Grow the production branch queues without a per-round allocation."""

        count = int(count)
        if count > self.boundary_branch_workspace.capacity:
            grown = max(count, max(1, 2 * self.boundary_branch_workspace.capacity))
            self.boundary_branch_workspace = _BoundaryBranchWorkspace.allocate(
                self.torch, self.device, grown
            )

    def _ensure_portal_workspace(self, count):
        """Grow the production direct-face queues and hit records."""

        count = int(count)
        if self.portal_workspace is None:
            raise RuntimeError("portal workspace is unavailable in compatibility mode")
        if count > self.portal_workspace.capacity:
            from .triton_scene.portals import PortalWorkspace

            grown = max(count, max(1, 2 * self.portal_workspace.capacity))
            self.portal_workspace = PortalWorkspace.allocate(grown, self.device)

    def _ensure_fused_portal_boundary_workspace(self, count):
        """Grow the fused path's fallback-only queue without hit storage."""

        count = int(count)
        workspace = self.fused_portal_boundary_workspace
        if workspace is None:
            raise RuntimeError("fused portal boundary workspace is unavailable")
        if count > workspace.capacity:
            from .triton_scene.fused_portal_boundary import (
                FusedPortalBoundaryWorkspace,
            )

            grown = max(count, max(1, 2 * workspace.capacity))
            self.fused_portal_boundary_workspace = (
                FusedPortalBoundaryWorkspace.allocate(grown, self.device)
            )

    def _ensure_collision_queues(self, count):
        """Grow ping-pong collision queues and their boundary accumulator."""

        from chroma.triton.transport import (
            CollisionQueueWorkspace,
            DeviceQueue,
        )

        count = int(count)
        current = (
            0
            if self.collision_queue_workspaces is None
            else self.collision_queue_workspaces[0].capacity
        )
        if count <= current:
            return
        self.collision_queue_workspaces = tuple(
            CollisionQueueWorkspace.allocate(
                count, device=self.device, dtype=self.torch.int32
            )
            for _ in range(2)
        )
        self.collision_boundary_accumulator = DeviceQueue.allocate(
            count, device=self.device, dtype=self.torch.int32
        )
        if self.device_pending_count is None:
            self.device_pending_count = self.torch.zeros(
                1, dtype=self.torch.int32, device=self.device
            )

    def _ensure_device_boundary_workspace(self, count):
        """Grow the host-free geometry and round-ledger workspaces."""

        from .triton_scene.device_geometry import (
            DeviceBoundaryMergeWorkspace,
            DeviceBoundaryRayWorkspace,
        )
        from .triton_scene.device_scheduler import DeviceRoundTrace
        from .triton_scene.intersect import (
            allocate_split_intersection_workspace,
        )

        count = int(count)
        if count < 0:
            raise ValueError("device boundary capacity cannot be negative")
        if (
            self.device_boundary_ray_workspace is None
            or count > self.device_boundary_ray_workspace.capacity
        ):
            self.device_boundary_ray_workspace = (
                DeviceBoundaryRayWorkspace.allocate(count, self.device)
            )
        if (
            self.device_boundary_merge_workspace is None
            or count > self.device_boundary_merge_workspace.capacity
        ):
            self.device_boundary_merge_workspace = (
                DeviceBoundaryMergeWorkspace.allocate(count, self.device)
            )
        if count > self.analytic_workspace.capacity:
            self.analytic_workspace = allocate_split_intersection_workspace(
                count, self.device
            )
        self.pmt_workspace.ensure_ray_capacity(count)
        self.pmt_workspace.ensure_result_capacity(count)
        if (
            self.device_round_trace is None
            or self.device_round_trace.capacity < self.device_round_batch
        ):
            self.device_round_trace = DeviceRoundTrace.allocate(
                self.device_round_batch, device=self.device
            )
            self.device_continuation_counts = self.torch.empty(
                self.device_round_batch,
                dtype=self.torch.int32,
                device=self.device,
            )

    def _resolve_boundaries_device(self, state, queue, launch_capacity):
        """Resolve a DeviceQueue without materializing its live count."""

        from .triton_scene.device_geometry import (
            gather_boundary_rays_device_count,
            merge_boundaries_device_count,
            nextafter_positive_inf_device_count,
        )
        from .triton_scene.intersect import intersect_scene_triton_device_count

        if self.chroma_mesh_box_compatibility:
            raise ValueError(
                "device-count geometry is unavailable in exact Chroma mode"
            )
        launch_capacity = int(launch_capacity)
        self._ensure_device_boundary_workspace(launch_capacity)
        rays = gather_boundary_rays_device_count(
            state[0],
            state[1],
            state[6],
            state[7],
            queue,
            input_capacity=launch_capacity,
            launch_capacity=launch_capacity,
            out=self.device_boundary_ray_workspace,
            block_size=256,
        )
        analytic = intersect_scene_triton_device_count(
            self.analytic_scene,
            rays.origins,
            rays.directions,
            queue.count,
            last_instance=rays.last_instances,
            last_triangle=rays.last_triangles,
            workspace=self.analytic_workspace,
            out=self.analytic_workspace.outputs(launch_capacity),
            split_wires=True,
        )
        nextafter_positive_inf_device_count(
            analytic.distance,
            queue.count,
            launch_capacity=launch_capacity,
            out=rays.pmt_tmax,
        )
        if self.fused_pmt:
            from .triton_scene.fused_pmt import (
                nearest_pmt_hit_fused_grid_device_count as pmt_query,
            )
            pmt_options = {
                "maximum_grid_candidates": (
                    self.fused_pmt_max_candidates
                ),
                "routing_counters": self.fused_pmt_routing_counters,
                "compact_union": self.fused_pmt_compact_union,
            }
        else:
            from .triton_scene.instances import (
                nearest_pmt_hit_tlas_device_count as pmt_query,
            )
            pmt_options = {}
        pmt = pmt_query(
            self.pmt_accelerator,
            rays.origins,
            rays.directions,
            queue.count,
            launch_capacity=launch_capacity,
            tmax=rays.pmt_tmax,
            last_instance=rays.last_instances,
            last_triangle=rays.last_triangles,
            workspace=self.pmt_workspace,
            out=self.pmt_workspace.outputs(launch_capacity),
            **pmt_options,
        )
        return merge_boundaries_device_count(
            self.scene_device,
            analytic,
            pmt,
            rays.directions,
            queue.count,
            launch_capacity=launch_capacity,
            out=self.device_boundary_merge_workspace,
        )

    def _resolve_boundaries(self, state, queue, *, fused_pmt=False):
        torch = self.torch
        from .triton_scene.intersect import (
            intersect_scene_triton,
            intersect_scene_triton_split,
        )

        self._ensure_boundary_workspace(queue.numel())
        positions, directions = state[0], state[1]
        origins, rays, last_instance, last_triangle, pmt_tmax = (
            self.boundary_ray_workspace.outputs(queue.numel())
        )
        triton, gather = _load_boundary_gather_kernel()
        gather[(triton.cdiv(queue.numel(), 256),)](
            positions,
            directions,
            state[6],
            state[7],
            queue,
            origins,
            rays,
            last_instance,
            last_triangle,
            queue.numel(),
            BLOCK=256,
            num_warps=8,
        )
        if self.chroma_mesh_box_compatibility:
            # Chroma's global mesh remains authoritative for every flattened
            # triangle.  Its only analytic supplement is the periodic wire
            # query, so compile out the production macro-box replacements.
            from .triton_scene import nearest_chroma_global_hit

            mesh = nearest_chroma_global_hit(
                self.chroma_global_accelerator,
                origins,
                rays,
                last_triangle=last_triangle,
                ray_tile=max(1, queue.numel()),
                workspace=self.chroma_global_workspace,
                # Topology validation proved the reference stack bound; the
                # simulation performs one sticky check after event sync.
                check_overflow=False,
            )
            # photon.h uses the mesh distance (or finite 1e30 miss sentinel)
            # to bound its exact ascending wire-lattice scan.  Retain that
            # incumbent and disable the production-only discriminant cull.
            pmt_tmax.copy_(mesh.distances)
            pmt_tmax.masked_fill_(mesh.triangle_ids < 0, 1.0e30)
            analytic = intersect_scene_triton(
                self.analytic_scene,
                origins,
                rays,
                tmax=pmt_tmax,
                out=self.analytic_workspace.outputs(queue.numel()),
                chroma_wire_frame=True,
                chroma_wire_full_scan=True,
                _box_count_override=0,
            )
            return self._merge_chroma_global_boundaries(analytic, mesh)

        analytic = intersect_scene_triton_split(
            self.analytic_scene,
            origins,
            rays,
            last_instance=last_instance,
            last_triangle=last_triangle,
            workspace=self.analytic_workspace,
            out=self.analytic_workspace.outputs(queue.numel()),
            chroma_mesh_boxes=self.chroma_mesh_box_compatibility,
            chroma_wire_frame=self.chroma_mesh_box_compatibility,
        )
        # Include an equal-distance mesh candidate so the merge retains Chroma's
        # mesh-on-tie rule while pruning farther PMT traversal.
        torch.nextafter(
            analytic.distance,
            self.boundary_ray_workspace.positive_infinity,
            out=pmt_tmax,
        )
        if fused_pmt:
            from .triton_scene.fused_pmt import nearest_pmt_hit_fused_grid

            pmt = nearest_pmt_hit_fused_grid(
                self.pmt_accelerator,
                origins,
                rays,
                tmax=pmt_tmax,
                last_instance=last_instance,
                last_triangle=last_triangle,
                workspace=self.pmt_workspace,
                out=self.pmt_workspace.outputs(queue.numel()),
                maximum_grid_candidates=(
                    self.fused_pmt_max_candidates
                ),
                routing_counters=self.fused_pmt_routing_counters,
                compact_union=self.fused_pmt_compact_union,
                # simulate() performs one event-level sticky overflow audit.
                check_overflow=False,
            )
        else:
            from .triton_scene.instances import nearest_pmt_hit

            pmt = nearest_pmt_hit(
                self.pmt_accelerator,
                origins,
                rays,
                tmax=pmt_tmax,
                last_instance=last_instance,
                last_triangle=last_triangle,
                # The scheduler has already bounded this boundary queue.  A
                # second PMT sub-tiling loop would only add launch overhead.
                ray_tile=None,
                workspace=self.pmt_workspace,
                out=self.pmt_workspace.outputs(queue.numel()),
                # Topology-derived TLAS/BLAS stacks append to a sticky device
                # audit bit.  Checking each boundary round would introduce a
                # host rendezvous; simulate() checks it once after event sync.
                check_overflow=False,
                use_grid=self.pmt_grid,
                chroma_world_compatibility=(
                    self.chroma_mesh_box_compatibility
                ),
            )
        return self._merge_boundaries(analytic, pmt, rays)

    def _step_boundaries_specialized(
            self, state, queue, hit, seed, photon_id_base, max_steps,
            global_photon_ids=None, reset_survivors=True):
        """Run production boundary physics through compact branch queues.

        All launches are ordered on the caller's current CUDA stream.  The
        classifier's three counts stay device-resident; workers launch over
        the known input capacity and mask against those counts, avoiding the
        synchronization that a branch-size host read would otherwise add.
        """

        from chroma.triton.transport import DeviceQueue

        count = int(queue.numel())
        self._ensure_survivor_queue(count)
        self._ensure_boundary_branch_workspace(count)
        if reset_survivors:
            self.survivor_queue_count.zero_()
        workspace = self.boundary_branch_workspace
        workspace.counts.zero_()
        triton, classify, branch = _load_production_boundary_kernels()
        grid = (triton.cdiv(count, self.block_size),)
        classify[grid](
            *state,
            queue,
            *hit,
            self.scene_device["tables_material_refractive_index"],
            self.scene_device["tables_material_absorption_length"],
            self.scene_device["tables_material_scattering_length"],
            self.scene_device["tables_surface_detect"],
            self.scene_device["tables_surface_absorb"],
            self.scene_device["tables_surface_reflect_diffuse"],
            self.scene_device["tables_surface_reflect_specular"],
            state[4] if global_photon_ids is None else global_photon_ids,
            workspace.rows,
            workspace.counts,
            workspace.capacity,
            self.survivor_queue_buffer,
            self.survivor_queue_count,
            count,
            int(seed),
            int(photon_id_base),
            int(max_steps),
            USE_GLOBAL_IDS=global_photon_ids is not None,
            BLOCK=self.block_size,
            num_warps=min(4, max(1, self.block_size // 32)),
        )
        branch_arguments = (
            state[1],
            state[2],
            state[4],
            state[5],
            state[9],
            queue,
            hit[1],
            hit[2],
            hit[3],
            self.scene_device["tables_material_refractive_index"],
            state[4] if global_photon_ids is None else global_photon_ids,
            workspace.rows,
            workspace.counts,
            self.survivor_queue_buffer,
            self.survivor_queue_count,
            workspace.capacity,
            count,
            int(seed),
            int(photon_id_base),
            int(max_steps),
        )
        # Separate constexpr variants ensure each generated kernel contains
        # exactly one expensive physics implementation.
        for branch_id in range(3):
            branch[grid](
                *branch_arguments,
                BRANCH=branch_id,
                USE_GLOBAL_IDS=global_photon_ids is not None,
                BLOCK=self.block_size,
                num_warps=min(4, max(1, self.block_size // 32)),
            )
        return DeviceQueue(
            self.survivor_queue_buffer, self.survivor_queue_count
        )

    def _step_portal_boundaries(
            self, state, queue, seed, photon_id_base, max_steps,
            global_photon_ids=None):
        """Resolve certified walls directly and spill ambiguous rays.

        The partition is performed on the already host-sized boundary tensor.
        Direct and fallback counts are copied together in one synchronization,
        after which both boundary-physics launches append to the same survivor
        queue.  Hit rows remain paired with their atomically compacted photon
        IDs, so inter-CTA queue order has no semantic effect.
        """

        from chroma.triton.transport import DeviceQueue
        from .triton_scene.portals import partition_certified_box_portals

        count = int(queue.numel())
        self._ensure_survivor_queue(count)
        self._ensure_portal_workspace(count)
        partition = partition_certified_box_portals(
            state[0],
            state[1],
            queue,
            state[6],
            state[7],
            self.portal_descriptor,
            workspace=self.portal_workspace,
            block_size=self.block_size,
        )
        direct_count, fallback_count = map(
            int, self.portal_workspace.counts.cpu().tolist()
        )
        self.last_portal_counts = (direct_count, fallback_count)
        if direct_count + fallback_count != count:
            raise RuntimeError(
                "portal partition did not preserve the boundary population"
            )

        self.survivor_queue_count.zero_()
        if direct_count:
            direct_queue = partition.direct.buffer[:direct_count]
            direct_hit = tuple(value[:direct_count] for value in partition.hit)
            self._step_boundaries_specialized(
                state,
                direct_queue,
                direct_hit,
                seed,
                photon_id_base,
                max_steps,
                global_photon_ids=global_photon_ids,
                reset_survivors=False,
            )
        if fallback_count:
            fallback_queue = partition.fallback.buffer[:fallback_count]
            fallback_hit = self._resolve_boundaries(
                state,
                fallback_queue,
                fused_pmt=_production_fused_pmt_enabled(
                    self.fused_pmt,
                    use_random_tape=False,
                    legacy_specular_reflection=(
                        self.legacy_specular_reflection
                    ),
                    chroma_mesh_box_compatibility=(
                        self.chroma_mesh_box_compatibility
                    ),
                    pmt_grid=self.pmt_grid,
                ),
            )
            self._step_boundaries_specialized(
                state,
                fallback_queue,
                fallback_hit,
                seed,
                photon_id_base,
                max_steps,
                global_photon_ids=global_photon_ids,
                reset_survivors=False,
            )
        return DeviceQueue(
            self.survivor_queue_buffer, self.survivor_queue_count
        )

    def _step_portal_boundaries_device(
            self, state, queue, seed, photon_id_base, max_steps,
            *, input_capacity, launch_capacity, survivor_queue,
            global_photon_ids=None):
        """Route and advance a device-count boundary queue without a poll.

        ``input_capacity`` bounds the queue's live prefix while
        ``launch_capacity`` bounds the capacity-sized portal grid.  Direct
        portal hits run through the monolithic corrected boundary kernel
        first.  Only the fallback queue traverses analytic/PMT geometry, and
        both disjoint populations append into the caller's collision-carry
        queue.  Their device counts are never materialized on the host.
        """

        input_capacity = int(input_capacity)
        launch_capacity = int(launch_capacity)
        if self.fused_portal_boundary:
            from .triton_scene.fused_portal_boundary import (
                step_fused_direct_portals,
            )

            self._ensure_fused_portal_boundary_workspace(launch_capacity)
            fallback = step_fused_direct_portals(
                state,
                queue,
                survivor_queue,
                self.portal_descriptor,
                self.scene_device,
                workspace=self.fused_portal_boundary_workspace,
                input_capacity=input_capacity,
                launch_capacity=launch_capacity,
                seed=seed,
                photon_id_base=photon_id_base,
                max_steps=max_steps,
                global_photon_ids=global_photon_ids,
                block_size=self.block_size,
            )
            fallback_hit = self._resolve_boundaries_device(
                state, fallback, input_capacity
            )
            self._step_boundaries(
                state,
                fallback,
                fallback_hit,
                seed,
                photon_id_base,
                max_steps,
                global_photon_ids=global_photon_ids,
                input_capacity=input_capacity,
                survivor_queue=survivor_queue,
                reset_survivors=False,
            )
            return survivor_queue

        from .triton_scene.portals import partition_certified_box_portals

        self._ensure_portal_workspace(launch_capacity)
        partition = partition_certified_box_portals(
            state[0],
            state[1],
            queue,
            state[6],
            state[7],
            self.portal_descriptor,
            workspace=self.portal_workspace,
            block_size=self.block_size,
            input_capacity=input_capacity,
            launch_capacity=launch_capacity,
        )
        self._step_boundaries(
            state,
            partition.direct,
            partition.hit,
            seed,
            photon_id_base,
            max_steps,
            global_photon_ids=global_photon_ids,
            input_capacity=input_capacity,
            survivor_queue=survivor_queue,
            reset_survivors=False,
        )
        fallback_hit = self._resolve_boundaries_device(
            state, partition.fallback, input_capacity
        )
        return self._step_boundaries(
            state,
            partition.fallback,
            fallback_hit,
            seed,
            photon_id_base,
            max_steps,
            global_photon_ids=global_photon_ids,
            input_capacity=input_capacity,
            survivor_queue=survivor_queue,
            reset_survivors=False,
        )

    def _step_boundaries(
            self, state, queue, hit, seed, photon_id_base, max_steps,
            global_photon_ids=None, random_tape=None, tape_audit=None,
            tape_row_indices=None, tape_trace=None, tape_certificate=None,
            state_certificate=None, input_capacity=None,
            survivor_queue=None, reset_survivors=True):
        """Apply one boundary interaction and return its live device queue.

        A tensor input retains the historical host-sized launch.  A
        :class:`~chroma.triton.transport.DeviceQueue` instead supplies a
        device count and uses ``input_capacity`` (the full buffer by default)
        for the launch grid.  This lets a capacity-shaped geometry pipeline
        feed the monolithic boundary kernel without reading the live count on
        the host; completely empty suffix CTAs return before any boundary
        physics.

        Passing ``survivor_queue`` reuses caller-owned storage.  Set
        ``reset_survivors=False`` to append after a prior producer without a
        host read or device reset.  As with every ``DeviceQueue`` producer,
        the caller must guarantee that the cumulative survivors fit that
        queue's capacity.  Device-count scheduling is production-only.  The
        random-tape/certificate path continues to use a tensor queue and the
        same monolithic generated implementation.
        """
        from chroma.triton.transport import DeviceQueue

        state_device = state[0].device
        input_count_is_pointer = isinstance(queue, DeviceQueue)
        explicit_input_capacity = input_capacity is not None
        if input_count_is_pointer:
            input_buffer = queue.buffer
            input_count = queue.count
            if (
                not isinstance(input_buffer, self.torch.Tensor)
                or not input_buffer.is_cuda
                or input_buffer.device != state_device
                or input_buffer.ndim != 1
                or not input_buffer.is_contiguous()
                or input_buffer.dtype not in (
                    self.torch.int32, self.torch.int64
                )
            ):
                raise ValueError(
                    "DeviceQueue buffer must be contiguous one-dimensional "
                    "CUDA indices on the simulation device"
                )
            if (
                not isinstance(input_count, self.torch.Tensor)
                or not input_count.is_cuda
                or input_count.device != state_device
                or input_count.dtype != self.torch.int32
                or input_count.shape != (1,)
                or not input_count.is_contiguous()
            ):
                raise ValueError(
                    "DeviceQueue count must be contiguous CUDA int32 with "
                    "shape (1,) on the simulation device"
                )
            launch_capacity = (
                input_buffer.numel()
                if input_capacity is None else int(input_capacity)
            )
            if launch_capacity < 0 or launch_capacity > input_buffer.numel():
                raise ValueError(
                    "input_capacity must fit the DeviceQueue buffer"
                )
            nitems_argument = input_count
        else:
            input_buffer = queue
            if not isinstance(input_buffer, self.torch.Tensor):
                raise TypeError("queue must be a CUDA tensor or DeviceQueue")
            if input_capacity is not None and int(input_capacity) != queue.numel():
                raise ValueError(
                    "input_capacity is only variable for DeviceQueue input"
                )
            launch_capacity = int(queue.numel())
            nitems_argument = launch_capacity
        use_random_tape = random_tape is not None or tape_audit is not None
        if input_count_is_pointer and (
            use_random_tape or self.chroma_mesh_box_compatibility
        ):
            raise ValueError(
                "DeviceQueue boundary input is production-only; exact "
                "compatibility and random-tape launches require a tensor queue"
            )
        if input_count_is_pointer:
            if not isinstance(hit, (tuple, list)) or len(hit) != 8:
                raise ValueError("boundary hit data must contain eight tensors")
            hit_layout = (
                ("distance", self.torch.float32, 1),
                ("normal", self.torch.float32, 2),
                ("material_from", self.torch.int32, 1),
                ("material_to", self.torch.int32, 1),
                ("surface", self.torch.int32, 1),
                ("instance", self.torch.int32, 1),
                ("triangle", self.torch.int32, 1),
                ("channel", self.torch.int32, 1),
            )
            for value, (name, dtype, ndim) in zip(hit, hit_layout):
                shape_ok = (
                    isinstance(value, self.torch.Tensor)
                    and value.ndim == ndim
                    and value.shape[0] >= launch_capacity
                )
                if ndim == 2:
                    shape_ok = shape_ok and value.shape[1] == 3
                if (
                    not shape_ok
                    or not value.is_cuda
                    or value.device != state_device
                    or value.dtype != dtype
                    or not value.is_contiguous()
                ):
                    expected = (
                        f"[at least {launch_capacity}, 3]"
                        if ndim == 2 else f"[at least {launch_capacity}]"
                    )
                    raise ValueError(
                        f"boundary hit {name} must be contiguous same-device "
                        f"{dtype} with shape {expected}"
                    )
        if (random_tape is None) != (tape_audit is None):
            raise ValueError(
                "random_tape and tape_audit must be supplied together"
            )
        certify_random_tape = tape_certificate is not None
        certify_state = state_certificate is not None
        if certify_random_tape and not use_random_tape:
            raise ValueError(
                "tape_certificate requires random_tape and tape_audit"
            )
        if certify_state and not use_random_tape:
            raise ValueError(
                "state_certificate requires random_tape and tape_audit"
            )
        if certify_state and not certify_random_tape:
            raise ValueError(
                "state_certificate requires tape_certificate"
            )
        use_tape_rows = tape_row_indices is not None
        if use_tape_rows and not use_random_tape:
            raise ValueError("tape_row_indices requires a random tape")
        if tape_trace is not None and not use_random_tape:
            raise ValueError("tape_trace requires a random tape")
        if _production_branch_specialization_enabled(
            self.branch_specialized_boundary,
            use_random_tape=use_random_tape,
            legacy_specular_reflection=self.legacy_specular_reflection,
            chroma_mesh_box_compatibility=self.chroma_mesh_box_compatibility,
        ) and (
            not input_count_is_pointer
            and not explicit_input_capacity
            and survivor_queue is None
            and reset_survivors
        ):
            return self._step_boundaries_specialized(
                state,
                input_buffer,
                hit,
                seed,
                photon_id_base,
                max_steps,
                global_photon_ids=global_photon_ids,
            )
        triton, kernel = _load_boundary_kernel()
        if use_random_tape:
            from chroma.triton.rng_alignment import (
                CERTIFICATE_DRAW_MASK,
                STATE_CERTIFICATE_FIELD_COUNT,
                TorchInteractionCertificate,
                TorchRandomTape,
                TorchStateCertificate,
                TorchTapeAudit,
            )

            if STATE_CERTIFICATE_FIELD_COUNT != STATE_CERTIFICATE_WIDTH:
                raise RuntimeError(
                    "Triton boundary state-certificate layout drifted from "
                    "chroma.triton.rng_alignment"
                )

            if not isinstance(random_tape, TorchRandomTape):
                raise TypeError("random_tape must be a TorchRandomTape")
            if not isinstance(tape_audit, TorchTapeAudit):
                raise TypeError("tape_audit must be a TorchTapeAudit")
            tape_values = random_tape.values
            tape_global_ids = random_tape.global_photon_ids
            if (
                not isinstance(tape_values, self.torch.Tensor)
                or not tape_values.is_cuda
                or tape_values.device != state_device
                or tape_values.dtype != self.torch.float32
                or tape_values.ndim != 3
                or not tape_values.is_contiguous()
            ):
                raise ValueError(
                    "random tape values must be contiguous CUDA float32 "
                    "[row, interaction, draw] on the simulation device"
                )
            tape_photon_count = int(tape_values.shape[0])
            if (
                not isinstance(tape_global_ids, self.torch.Tensor)
                or not tape_global_ids.is_cuda
                or tape_global_ids.device != state_device
                or tape_global_ids.dtype != self.torch.int64
                or tape_global_ids.shape != (tape_photon_count,)
                or not tape_global_ids.is_contiguous()
            ):
                raise ValueError(
                    "random tape global IDs must be contiguous CUDA int64 [row]"
                )
            if (
                tape_values.shape[1]
                != int(random_tape.spec.max_interactions)
                or tape_values.shape[2]
                != int(random_tape.spec.draws_per_interaction)
            ):
                raise ValueError("random tape tensor shape does not match its spec")
            for value, name in (
                (tape_audit.interaction_cursor, "interaction_cursor"),
                (tape_audit.draw_cursor, "draw_cursor"),
                (tape_audit.overflow, "overflow"),
            ):
                if (
                    not isinstance(value, self.torch.Tensor)
                    or not value.is_cuda
                    or value.device != state_device
                    or value.dtype != self.torch.int32
                    or value.shape != (tape_photon_count,)
                    or not value.is_contiguous()
                ):
                    raise ValueError(
                        f"tape audit {name} must be contiguous CUDA int32 [row]"
                    )
            if use_tape_rows:
                if (
                    not isinstance(tape_row_indices, self.torch.Tensor)
                    or not tape_row_indices.is_cuda
                    or tape_row_indices.device != state_device
                    or tape_row_indices.dtype != self.torch.int32
                    or tape_row_indices.shape != state[3].shape
                    or not tape_row_indices.is_contiguous()
                ):
                    raise ValueError(
                        "tape_row_indices must be contiguous CUDA int32 and "
                        "match the resident state"
                    )
            elif tape_photon_count != state[3].numel():
                raise ValueError(
                    "implicit tape rows require one row per resident state slot"
                )
            if tape_trace is not None:
                if not isinstance(tape_trace, BoundaryTapeTrace):
                    raise TypeError("tape_trace must be a BoundaryTapeTrace")
                for value, name in (
                    (tape_trace.interaction, "interaction"),
                    (tape_trace.draw_count, "draw_count"),
                    (tape_trace.decision, "decision"),
                    (tape_trace.work_overflow, "work_overflow"),
                ):
                    if (
                        not isinstance(value, self.torch.Tensor)
                        or not value.is_cuda
                        or value.device != state_device
                        or value.dtype != self.torch.int32
                        or value.shape != (tape_photon_count,)
                        or not value.is_contiguous()
                    ):
                        raise ValueError(
                            f"boundary tape trace {name} must be contiguous "
                            "CUDA int32 [row]"
                        )
            max_tape_interactions = int(random_tape.spec.max_interactions)
            tape_draws_per_interaction = int(
                random_tape.spec.draws_per_interaction
            )
            if certify_random_tape:
                if not isinstance(
                    tape_certificate, TorchInteractionCertificate
                ):
                    raise TypeError(
                        "tape_certificate must be a "
                        "TorchInteractionCertificate"
                    )
                certificate_words = tape_certificate.words
                if (
                    not isinstance(certificate_words, self.torch.Tensor)
                    or not certificate_words.is_cuda
                    or certificate_words.device != state_device
                    or certificate_words.dtype != self.torch.int32
                    or certificate_words.shape != (
                        tape_photon_count, max_tape_interactions
                    )
                    or not certificate_words.is_contiguous()
                ):
                    raise ValueError(
                        "tape certificate must be contiguous CUDA int32 "
                        "[row, interaction] on the simulation device"
                    )
                if tape_draws_per_interaction > int(CERTIFICATE_DRAW_MASK):
                    raise ValueError(
                        "certificate draw counts must fit 28 bits"
                    )
            else:
                # Real dead pointer for the certificate-disabled tape variant.
                certificate_words = tape_audit.overflow
            if certify_state:
                if not isinstance(state_certificate, TorchStateCertificate):
                    raise TypeError(
                        "state_certificate must be a TorchStateCertificate"
                    )
                state_certificate_words = state_certificate.words
                if (
                    not isinstance(
                        state_certificate_words, self.torch.Tensor
                    )
                    or not state_certificate_words.is_cuda
                    or state_certificate_words.device != state_device
                    or state_certificate_words.dtype != self.torch.int32
                    or state_certificate_words.shape != (
                        tape_photon_count,
                        max_tape_interactions,
                        STATE_CERTIFICATE_WIDTH,
                    )
                    or not state_certificate_words.is_contiguous()
                ):
                    raise ValueError(
                        "state certificate must be contiguous CUDA int32 "
                        "[row, interaction, 15] on the simulation device"
                    )
            else:
                # Compile-time-dead pointer when state capture is disabled.
                state_certificate_words = tape_audit.overflow
            # Enough static attempts to consume every complete three-draw
            # rejection tuple which can fit in the bounded interaction row.
            max_diffuse_attempts = max(
                1, (tape_draws_per_interaction + 2) // 3
            )
            tape_interaction_pointer = tape_audit.interaction_cursor
            tape_draw_pointer = tape_audit.draw_cursor
            tape_overflow_pointer = tape_audit.overflow
            if tape_trace is None:
                trace_interaction = trace_draw_count = trace_decision = state[4]
                trace_work_overflow = state[4]
            else:
                trace_interaction = tape_trace.interaction
                trace_draw_count = tape_trace.draw_count
                trace_decision = tape_trace.decision
                trace_work_overflow = tape_trace.work_overflow
        else:
            # Dead pointers in the production constexpr specialization.
            tape_values = state[0]
            tape_global_ids = state[5]
            tape_row_indices = input_buffer
            tape_interaction_pointer = state[4]
            tape_draw_pointer = state[4]
            tape_overflow_pointer = state[4]
            certificate_words = state[4]
            state_certificate_words = state[4]
            trace_interaction = trace_draw_count = trace_decision = state[4]
            trace_work_overflow = state[4]
            tape_photon_count = 0
            max_tape_interactions = 1
            tape_draws_per_interaction = 1
            max_diffuse_attempts = 1
        if survivor_queue is None:
            if not reset_survivors:
                raise ValueError(
                    "reset_survivors=False requires an explicit survivor_queue"
                )
            self._ensure_survivor_queue(launch_capacity)
            output = DeviceQueue(
                self.survivor_queue_buffer, self.survivor_queue_count
            )
        else:
            if not isinstance(survivor_queue, DeviceQueue):
                raise TypeError("survivor_queue must be a DeviceQueue")
            output = survivor_queue
            if (
                not isinstance(output.buffer, self.torch.Tensor)
                or not output.buffer.is_cuda
                or output.buffer.device != state_device
                or output.buffer.dtype != self.torch.int32
                or output.buffer.ndim != 1
                or not output.buffer.is_contiguous()
                or output.capacity < launch_capacity
            ):
                raise ValueError(
                    "survivor_queue must have a contiguous same-device int32 "
                    "buffer with capacity at least input_capacity"
                )
            if (
                not isinstance(output.count, self.torch.Tensor)
                or not output.count.is_cuda
                or output.count.device != state_device
                or output.count.dtype != self.torch.int32
                or output.count.shape != (1,)
                or not output.count.is_contiguous()
            ):
                raise ValueError(
                    "survivor_queue count must be contiguous CUDA int32 with "
                    "shape (1,) on the simulation device"
                )
            if launch_capacity:
                input_begin = int(input_buffer.data_ptr())
                input_end = (
                    input_begin
                    + input_buffer.numel() * input_buffer.element_size()
                )
                output_begin = int(output.buffer.data_ptr())
                output_end = (
                    output_begin
                    + output.buffer.numel() * output.buffer.element_size()
                )
                if max(input_begin, output_begin) < min(input_end, output_end):
                    raise ValueError(
                        "input and survivor queue buffers must not alias"
                    )
            if input_count_is_pointer and (
                int(input_count.data_ptr()) == int(output.count.data_ptr())
            ):
                raise ValueError(
                    "input and survivor queue counters must not alias"
                )
        if reset_survivors:
            output.reset()
        if launch_capacity == 0:
            return output
        grid = (triton.cdiv(launch_capacity, self.block_size),)
        kernel[grid](
            *state,
            input_buffer,
            *hit,
            self.scene_device["tables_material_refractive_index"],
            self.scene_device["tables_material_absorption_length"],
            self.scene_device["tables_material_scattering_length"],
            self.scene_device["tables_surface_detect"],
            self.scene_device["tables_surface_absorb"],
            self.scene_device["tables_surface_reflect_diffuse"],
            self.scene_device["tables_surface_reflect_specular"],
            state[4] if global_photon_ids is None else global_photon_ids,
            output.buffer,
            output.buffer,
            output.count,
            nitems_argument,
            int(seed),
            int(photon_id_base),
            int(max_steps),
            tape_values,
            tape_global_ids,
            tape_row_indices,
            tape_interaction_pointer,
            tape_draw_pointer,
            tape_overflow_pointer,
            certificate_words,
            state_certificate_words,
            trace_interaction,
            trace_draw_count,
            trace_decision,
            trace_work_overflow,
            tape_photon_count,
            USE_GLOBAL_IDS=global_photon_ids is not None,
            USE_RANDOM_TAPE=use_random_tape,
            USE_TAPE_ROWS=use_tape_rows,
            CERTIFY_RANDOM_TAPE=certify_random_tape,
            CERTIFY_STATE=certify_state,
            TRACE_RANDOM_TAPE=tape_trace is not None,
            MAX_TAPE_INTERACTIONS=max_tape_interactions,
            TAPE_DRAWS_PER_INTERACTION=tape_draws_per_interaction,
            STATE_WORDS_PER_INTERACTION=STATE_CERTIFICATE_WIDTH,
            MAX_DIFFUSE_ATTEMPTS=max_diffuse_attempts,
            LEGACY_SPECULAR_REFLECTION=self.legacy_specular_reflection,
            DIRECT_QUEUE=True,
            NITEMS_IS_POINTER=input_count_is_pointer,
            BLOCK=self.block_size,
            num_warps=min(4, max(1, self.block_size // 32)),
        )
        return output

    def _new_state(self, count, center, voxel_size, seed, photon_id_base):
        torch = self.torch
        from chroma.triton.physics_kernels import generate_photon_bomb
        source = generate_photon_bomb(
            count,
            seed=_signed_seed32(seed),
            center=center,
            voxel_size=float(voxel_size),
            wavelength=450.0,
            photon_id_base=int(photon_id_base),
            device=self.device,
            block_size=self.block_size,
        )
        positions = source.position().contiguous()
        directions = source.direction().contiguous()
        polarizations = source.polarization().contiguous()
        times = torch.zeros(count, dtype=torch.float32, device=self.device)
        histories = torch.zeros(count, dtype=torch.int32, device=self.device)
        rng_counters = torch.zeros(count, dtype=torch.int64, device=self.device)
        last_instances = torch.full(
            (count,), -1, dtype=torch.int32, device=self.device
        )
        last_triangles = torch.full_like(last_instances, -1)
        detected_channels = torch.full_like(last_instances, -1)
        step_counts = torch.zeros(count, dtype=torch.int32, device=self.device)
        return (
            positions, directions, polarizations, times, histories, rng_counters,
            last_instances, last_triangles, detected_channels,
            step_counts,
        )

    def _propagate_state_device(
            self,
            state,
            pending,
            seed,
            photon_id_base,
            max_steps,
            max_boundary_rounds,
            global_photon_ids=None):
        """Advance corrected production in fixed asynchronous round batches.

        Every batch is one uninterrupted CUDA-stream dependency chain.  Queue
        counts remain on the device through collision, exact geometry, and
        boundary physics; one ledger transfer at the end of the batch both
        validates queue capacities and supplies the next smaller launch cap.
        Per-photon RNG counters and global IDs make the resulting inter-photon
        execution order immaterial.
        """

        from chroma.triton.transport import DeviceQueue, collision_first_epoch

        max_boundary_rounds = int(max_boundary_rounds)
        if max_boundary_rounds <= 0:
            raise ValueError("max_boundary_rounds must be positive")
        if global_photon_ids is not None:
            if (
                global_photon_ids.dtype != self.torch.int64
                or not global_photon_ids.is_cuda
                or not global_photon_ids.is_contiguous()
                or global_photon_ids.shape != state[3].shape
            ):
                raise ValueError(
                    "global_photon_ids must be contiguous CUDA int64 and match state"
                )

        launch_capacity = int(pending.numel())
        if launch_capacity == 0:
            return pending, 0, 0, []
        self._ensure_collision_queues(launch_capacity)
        self._ensure_device_boundary_workspace(launch_capacity)
        pending_count = self.device_pending_count
        pending_count.fill_(launch_capacity)
        current = DeviceQueue(pending, pending_count)
        trace = []
        boundary_events = 0
        logical_rounds = 0

        def workspace_after(queue):
            """Choose a collision output whose storage cannot alias input."""

            queue_pointer = int(queue.buffer.data_ptr())
            first_pointer = int(
                self.collision_queue_workspaces[0].continuing.buffer.data_ptr()
            )
            return 1 if queue_pointer == first_pointer else 0

        while launch_capacity and logical_rounds < max_boundary_rounds:
            scheduled = min(
                self.device_round_batch,
                max_boundary_rounds - logical_rounds,
            )
            round_ledger = self.device_round_trace.reset()
            for batch_round in range(scheduled):
                boundary_queue = self.collision_boundary_accumulator.reset()
                history_queue = current
                workspace_index = workspace_after(history_queue)
                for _ in range(self.history_epochs_per_poll):
                    workspace = self.collision_queue_workspaces[workspace_index]
                    epoch = collision_first_epoch(
                        state[0], state[1], state[2], state[3], state[4], state[5],
                        history_queue,
                        self.safe_lower,
                        self.safe_upper,
                        self.lar_absorption_length,
                        self.lar_scattering_length,
                        self.lar_refractive_index,
                        seed=_signed_seed32(seed, 0x51A3),
                        photon_id_base=int(photon_id_base),
                        max_scatter=self.history_length,
                        block_size=self.block_size,
                        partition="active",
                        step_counts=state[9],
                        max_steps=max_steps,
                        last_instances=state[6],
                        last_triangles=state[7],
                        global_photon_ids=global_photon_ids,
                        queue_workspace=workspace,
                        input_capacity=launch_capacity,
                        boundary_accumulator=boundary_queue,
                        append_boundary=True,
                    )
                    history_queue = epoch.continuing
                    workspace_index = 1 - workspace_index

                # Rare photons which used every register-resident collision
                # slot remain in ``history_queue``.  Boundary survivors append
                # to that same disjoint carry queue, so no state is dropped and
                # no concatenation or count read is required.
                # Preserve the collision carry count before boundary survivors
                # append to it.  This asynchronous one-word D2D copy lets the
                # host reconstruct the historical ``(boundary, survivors)``
                # trace contract at the eventual batched rendezvous.
                self.device_continuation_counts[
                    batch_round : batch_round + 1
                ].copy_(history_queue.count)
                boundary_seed = _signed_seed32(seed, 0x7F4A)
                if self.portal_boundary:
                    current = self._step_portal_boundaries_device(
                        state,
                        boundary_queue,
                        boundary_seed,
                        photon_id_base,
                        max_steps,
                        input_capacity=launch_capacity,
                        launch_capacity=launch_capacity,
                        survivor_queue=history_queue,
                        global_photon_ids=global_photon_ids,
                    )
                else:
                    hit = self._resolve_boundaries_device(
                        state, boundary_queue, launch_capacity
                    )
                    current = self._step_boundaries(
                        state,
                        boundary_queue,
                        hit,
                        boundary_seed,
                        photon_id_base,
                        max_steps,
                        global_photon_ids=global_photon_ids,
                        input_capacity=launch_capacity,
                        survivor_queue=history_queue,
                        reset_survivors=False,
                    )
                round_ledger.snapshot(
                    batch_round,
                    boundary_queue,
                    current,
                    boundary_capacity=launch_capacity,
                    next_pending_capacity=launch_capacity,
                )

            # This is the sole host rendezvous for the whole fixed round batch.
            materialized = round_ledger.read(scheduled).records
            collision_continuations = (
                self.device_continuation_counts[:scheduled]
                .detach().cpu().numpy()
            )
            self.last_device_scheduler_syncs += 1
            final_count = int(materialized[-1, 1])
            stopped = False
            for row, (boundary_count, pending_survivors) in enumerate(
                    materialized.tolist()):
                boundary_count = int(boundary_count)
                pending_survivors = int(pending_survivors)
                continuing_count = int(collision_continuations[row])
                if not 0 <= continuing_count <= pending_survivors:
                    raise RuntimeError(
                        "device scheduler continuation count is inconsistent: "
                        f"continuing={continuing_count}, "
                        f"pending={pending_survivors}"
                    )
                boundary_survivors = pending_survivors - continuing_count
                if boundary_survivors > boundary_count:
                    raise RuntimeError(
                        "device scheduler boundary survivors exceed events: "
                        f"survivors={boundary_survivors}, "
                        f"events={boundary_count}"
                    )
                if boundary_count == 0 and pending_survivors == 0:
                    stopped = True
                    break
                # A collision-only continuation tick is not a boundary round.
                # It is deliberately absent from the public trace, matching
                # the synchronized scheduler's round/statistics definition.
                if boundary_count:
                    trace.append((boundary_count, boundary_survivors))
                    boundary_events += boundary_count
                    logical_rounds += 1
                if pending_survivors == 0:
                    stopped = True
                    break
            launch_capacity = final_count
            if stopped:
                launch_capacity = 0
                break

        return (
            current.buffer[:launch_capacity],
            logical_rounds,
            boundary_events,
            trace,
        )

    def _propagate_state(
            self,
            state,
            pending,
            seed,
            photon_id_base,
            max_steps,
            max_boundary_rounds,
            global_photon_ids=None,
            random_tape=None,
            tape_audit=None,
            tape_row_indices=None,
            tape_trace=None,
            tape_certificate=None,
            state_certificate=None):
        """Advance a resident state arena for a bounded number of rounds.

        ``pending`` contains arena-local slots.  Initial source slabs obtain
        their random stream from ``photon_id_base + slot``.  A compacted
        reservoir instead supplies an explicit global-ID array, allowing
        states from unrelated slabs and events to share one dense queue
        without changing any Philox stream.
        """

        torch = self.torch
        from chroma.triton.transport import collision_first_epoch

        if tape_certificate is not None and (
            random_tape is None or tape_audit is None
        ):
            raise ValueError(
                "tape_certificate requires random_tape and tape_audit"
            )
        if state_certificate is not None and (
            random_tape is None or tape_audit is None
        ):
            raise ValueError(
                "state_certificate requires random_tape and tape_audit"
            )
        if state_certificate is not None and tape_certificate is None:
            raise ValueError(
                "state_certificate requires tape_certificate"
            )

        max_boundary_rounds = int(max_boundary_rounds)
        if max_boundary_rounds <= 0:
            raise ValueError("max_boundary_rounds must be positive")
        if global_photon_ids is not None:
            if (
                global_photon_ids.dtype != torch.int64
                or not global_photon_ids.is_cuda
                or not global_photon_ids.is_contiguous()
                or global_photon_ids.shape != state[3].shape
            ):
                raise ValueError(
                    "global_photon_ids must be contiguous CUDA int64 and match state"
                )
        strict_debug = any(
            value is not None
            for value in (
                random_tape,
                tape_audit,
                tape_row_indices,
                tape_trace,
                tape_certificate,
                state_certificate,
            )
        )
        if _production_device_scheduler_enabled(
            self.device_scheduler,
            use_random_tape=strict_debug,
            legacy_specular_reflection=self.legacy_specular_reflection,
            chroma_mesh_box_compatibility=self.chroma_mesh_box_compatibility,
            branch_specialized_boundary=self.branch_specialized_boundary,
            portal_boundary=self.portal_boundary,
            pmt_grid=self.pmt_grid,
            fused_pmt=self.fused_pmt,
        ):
            return self._propagate_state_device(
                state,
                pending,
                seed,
                photon_id_base,
                max_steps,
                max_boundary_rounds,
                global_photon_ids=global_photon_ids,
            )
        boundary_rounds = 0
        boundary_events = 0
        trace = []
        while pending.numel() and boundary_rounds < max_boundary_rounds:
            history_capacity = int(pending.numel())
            self._ensure_collision_queues(history_capacity)
            boundary_accumulator = self.collision_boundary_accumulator.reset()
            history_queue = pending
            launch_capacity = history_capacity
            workspace_index = 0
            boundary_count = 0
            # A rare photon may scatter more than history_length times before
            # reaching the certified exit.  Several ping-pong epochs execute
            # from device-resident queue counts before the host polls once.
            # Boundary IDs append into one accumulator, so no intermediate
            # boundary tensors or concatenation are required.
            while True:
                for _ in range(self.history_epochs_per_poll):
                    workspace = self.collision_queue_workspaces[workspace_index]
                    epoch = collision_first_epoch(
                        state[0], state[1], state[2], state[3], state[4], state[5],
                        history_queue,
                        self.safe_lower,
                        self.safe_upper,
                        self.lar_absorption_length,
                        self.lar_scattering_length,
                        self.lar_refractive_index,
                        seed=_signed_seed32(seed, 0x51A3),
                        photon_id_base=int(photon_id_base),
                        max_scatter=self.history_length,
                        block_size=self.block_size,
                        partition="active",
                        step_counts=state[9],
                        max_steps=max_steps,
                        last_instances=state[6],
                        last_triangles=state[7],
                        global_photon_ids=global_photon_ids,
                        queue_workspace=workspace,
                        # Queue storage remains sized for the outer wavefront,
                        # but after a host count poll only the observed live
                        # prefix needs programs.  Reusing history_capacity here
                        # turns a handful of rare scatterers into another full
                        # wavefront launch on every epoch.
                        input_capacity=launch_capacity,
                        boundary_accumulator=boundary_accumulator,
                        append_boundary=True,
                        random_tape=random_tape,
                        tape_audit=tape_audit,
                        tape_row_indices=tape_row_indices,
                        tape_certificate=tape_certificate,
                        state_certificate=state_certificate,
                        state_wavelength=np.float32(450.0),
                        state_weight=np.float32(1.0),
                        state_evidx=np.uint32(0),
                    )
                    history_queue = epoch.continuing
                    workspace_index = 1 - workspace_index
                queue_counts = torch.cat(
                    (boundary_accumulator.count, history_queue.count)
                ).cpu().tolist()
                boundary_count, continuing_count = map(int, queue_counts)
                if (
                    boundary_count < 0
                    or continuing_count < 0
                    or boundary_count + continuing_count > history_capacity
                ):
                    raise RuntimeError(
                        "collision queue partition exceeded its source "
                        f"population: boundary={boundary_count}, "
                        f"continuing={continuing_count}, "
                        f"capacity={history_capacity}"
                    )
                if continuing_count == 0:
                    break
                launch_capacity = continuing_count

            if boundary_count == 0:
                pending = torch.empty(
                    0, dtype=torch.int32, device=self.device
                )
                break
            boundary_queue = boundary_accumulator.buffer[:boundary_count]
            boundary_events += int(boundary_queue.numel())
            if _production_portal_boundary_enabled(
                self.portal_boundary,
                branch_specialized_boundary=self.branch_specialized_boundary,
                use_random_tape=strict_debug,
                legacy_specular_reflection=self.legacy_specular_reflection,
                chroma_mesh_box_compatibility=(
                    self.chroma_mesh_box_compatibility
                ),
            ):
                survivors = self._step_portal_boundaries(
                    state,
                    boundary_queue,
                    _signed_seed32(seed, 0x7F4A),
                    photon_id_base,
                    max_steps,
                    global_photon_ids=global_photon_ids,
                )
            else:
                hit = self._resolve_boundaries(
                    state,
                    boundary_queue,
                    fused_pmt=_production_fused_pmt_enabled(
                        self.fused_pmt,
                        use_random_tape=strict_debug,
                        legacy_specular_reflection=(
                            self.legacy_specular_reflection
                        ),
                        chroma_mesh_box_compatibility=(
                            self.chroma_mesh_box_compatibility
                        ),
                        pmt_grid=self.pmt_grid,
                    ),
                )
                survivors = self._step_boundaries(
                    state, boundary_queue, hit,
                    _signed_seed32(seed, 0x7F4A), photon_id_base,
                    max_steps,
                    global_photon_ids=global_photon_ids,
                    random_tape=random_tape,
                    tape_audit=tape_audit,
                    tape_row_indices=tape_row_indices,
                    tape_trace=tape_trace,
                    tape_certificate=tape_certificate,
                    state_certificate=state_certificate,
                )
            pending = survivors.tensor()
            trace.append((int(boundary_queue.numel()), int(pending.numel())))
            boundary_rounds += 1

        return pending, boundary_rounds, boundary_events, trace

    def _simulate_tile(
            self,
            count,
            center,
            voxel_size,
            seed,
            photon_id_base,
            max_steps,
            random_tape=None,
            tape_audit=None,
            tape_row_indices=None,
            tape_trace=None,
            tape_certificate=None,
            state_certificate=None):
        """Compatibility path that completely drains one source slab."""

        torch = self.torch
        state = self._new_state(
            count, center, voxel_size, seed, photon_id_base
        )
        pending = torch.arange(count, dtype=torch.int32, device=self.device)
        pending, boundary_rounds, boundary_events, trace = self._propagate_state(
            state,
            pending,
            seed,
            photon_id_base,
            max_steps,
            max_steps,
            random_tape=random_tape,
            tape_audit=tape_audit,
            tape_row_indices=tape_row_indices,
            tape_trace=tape_trace,
            tape_certificate=tape_certificate,
            state_certificate=state_certificate,
        )

        if pending.numel():
            state[4][pending.to(torch.int64)] |= (1 << 0)
        detected = state[8] >= 0
        hit_times = state[3][detected]
        hit_channels = state[8][detected]
        return (
            FlatHits(hit_times, hit_channels), state, boundary_rounds,
            boundary_events, trace,
        )

    def _compact_reservoir_state(
            self, state, pending, photon_id_base, global_photon_ids=None):
        """Gather live slots into a small, globally identified state arena."""

        torch = self.torch
        if not pending.numel():
            return None
        # Atomic queue appends do not promise block order.  Sorting once at the
        # dense/tail boundary makes the reservoir deterministic and restores
        # global photon order inside each source slab.
        slots = torch.sort(pending.to(torch.int64)).values
        compacted = tuple(value.index_select(0, slots).contiguous() for value in state)
        if global_photon_ids is None:
            ids = slots + int(photon_id_base)
        else:
            ids = global_photon_ids.index_select(0, slots).contiguous()
        return compacted, ids

    def _state_hits(self, state):
        """Return compact hits while retaining no reference to the full state."""

        detected = state[8] >= 0
        return FlatHits(state[3][detected], state[8][detected])

    def fused_pmt_routing_snapshot(self):
        """Return opt-in event routing counters, or ``None`` when disabled."""

        if self.fused_pmt_routing_counters is None:
            return None
        return self.fused_pmt_routing_counters.snapshot()

    def simulate(
        self,
        nphotons: int,
        center: Sequence[float],
        *,
        voxel_size=30.0,
        wavelength=450.0,
        seed=1,
        max_steps=1000,
        keep_final_states=False,
        random_tape=None,
        tape_audit=None,
        tape_certificate=None,
        state_certificate=None,
    ) -> TritonSimulationResult:
        """Simulate one voxel and return compact flat hits on the CUDA device.

        Source slabs are bounded by the selected memory policy.  Unless final
        per-photon states are requested, each slab runs a dense prefix and its
        surviving state joins one shared reservoir.  This makes total event
        size independent of the rare-history drain cost.

        ``random_tape`` enables bounded CUDA/Triton lockstep debugging.  It
        must contain rows ordered by global photon IDs ``0..nphotons-1``.  A
        matching audit and dense per-interaction process/draw certificate are
        allocated when omitted; the mutated audit, certificate, and a
        row-indexed boundary trace are returned.  Source generation remains
        independent of this propagation-only tape.  Supplying the optional
        ``state_certificate`` additionally records all fifteen raw Photon
        words after every committed interaction; it is never allocated or
        written by the production path.
        """

        torch = self.torch
        nphotons = int(nphotons)
        if self.geometry_only:
            raise RuntimeError("geometry-only instances cannot run monochromatic transport; use the spectral transport driver")
        if nphotons < 0:
            raise ValueError("nphotons must be non-negative")
        if len(center) != 3:
            raise ValueError("center must contain three coordinates")
        if wavelength != 450.0:
            raise ValueError("the exact production specialization requires wavelength=450")
        if voxel_size < 0.0 or max_steps <= 0:
            raise ValueError("voxel_size and max_steps must be positive")
        if self.fused_pmt_routing_counters is not None:
            self.fused_pmt_routing_counters.reset()

        use_random_tape = (
            random_tape is not None
            or tape_audit is not None
            or tape_certificate is not None
            or state_certificate is not None
        )
        if random_tape is None and (
            tape_audit is not None
            or tape_certificate is not None
            or state_certificate is not None
        ):
            raise ValueError(
                "tape_audit, tape_certificate, and state_certificate require "
                "random_tape"
            )
        if self.fused_pmt and use_random_tape:
            raise ValueError(
                "fused_pmt is unavailable during strict random-tape or state "
                "certificate replay; construct the simulation with "
                "fused_pmt=False"
            )
        boundary_tape_trace = None
        if use_random_tape:
            from chroma.triton.rng_alignment import (
                CERTIFICATE_DRAW_MASK,
                TorchInteractionCertificate,
                TorchRandomTape,
                TorchStateCertificate,
                TorchTapeAudit,
                allocate_torch_audit,
                allocate_torch_certificate,
            )

            if not isinstance(random_tape, TorchRandomTape):
                raise TypeError("random_tape must be a TorchRandomTape")
            simulation_device = self.scene_device[
                "tables_material_refractive_index"
            ].device
            if (
                random_tape.values.device != simulation_device
                or random_tape.global_photon_ids.device != simulation_device
            ):
                raise ValueError("random_tape must be on the simulation device")
            if int(random_tape.values.shape[0]) != nphotons:
                raise ValueError(
                    "full-simulation tape mode requires one row per source photon"
                )
            if nphotons > np.iinfo(np.int32).max:
                raise ValueError(
                    "full-simulation tape rows currently require nphotons <= int32 max"
                )
            expected_ids = torch.arange(
                nphotons, dtype=torch.int64, device=self.device
            )
            if not torch.equal(random_tape.global_photon_ids, expected_ids):
                raise ValueError(
                    "full-simulation tape rows must be ordered by global photon "
                    "IDs 0..nphotons-1"
                )
            if tape_audit is None:
                tape_audit = allocate_torch_audit(
                    nphotons, device=self.device
                )
            elif not isinstance(tape_audit, TorchTapeAudit):
                raise TypeError("tape_audit must be a TorchTapeAudit")
            max_tape_interactions = int(random_tape.spec.max_interactions)
            if tape_certificate is None:
                tape_certificate = allocate_torch_certificate(
                    nphotons, max_tape_interactions, device=self.device
                )
            elif not isinstance(
                tape_certificate, TorchInteractionCertificate
            ):
                raise TypeError(
                    "tape_certificate must be a TorchInteractionCertificate"
                )
            certificate_words = tape_certificate.words
            if (
                not isinstance(certificate_words, torch.Tensor)
                or not certificate_words.is_cuda
                or certificate_words.device != simulation_device
                or certificate_words.dtype != torch.int32
                or certificate_words.shape
                != (nphotons, max_tape_interactions)
                or not certificate_words.is_contiguous()
            ):
                raise ValueError(
                    "tape certificate must be contiguous CUDA int32 "
                    "[row, interaction] on the simulation device"
                )
            if int(random_tape.spec.draws_per_interaction) > int(
                CERTIFICATE_DRAW_MASK
            ):
                raise ValueError("certificate draw counts must fit 28 bits")
            if state_certificate is not None:
                if not isinstance(
                    state_certificate, TorchStateCertificate
                ):
                    raise TypeError(
                        "state_certificate must be a TorchStateCertificate"
                    )
                state_certificate_words = state_certificate.words
                if (
                    not isinstance(state_certificate_words, torch.Tensor)
                    or not state_certificate_words.is_cuda
                    or state_certificate_words.device != simulation_device
                    or state_certificate_words.dtype != torch.int32
                    or state_certificate_words.shape != (
                        nphotons,
                        max_tape_interactions,
                        STATE_CERTIFICATE_WIDTH,
                    )
                    or not state_certificate_words.is_contiguous()
                ):
                    raise ValueError(
                        "state certificate must be contiguous CUDA int32 "
                        "[row, interaction, 15] on the simulation device"
                    )
            boundary_tape_trace = BoundaryTapeTrace.allocate(
                nphotons, self.device
            )

        torch.cuda.synchronize(self.device)
        started = _time.perf_counter()
        if self.chroma_mesh_box_compatibility:
            self.chroma_global_workspace.clear_sticky_overflow()
        else:
            self.pmt_workspace.clear_sticky_overflow()
        hit_times = []
        hit_channels = []
        final_states = [] if keep_final_states else None
        dense_rounds = 0
        reservoir_rounds = 0
        reservoir_photons = 0
        boundary_events = 0
        self.last_trace = []
        self.last_device_scheduler_syncs = 0
        tiles = 0
        available_memory_bytes = None
        if self.tile_size is None:
            available_memory_bytes = self.auto_memory_bytes
            if available_memory_bytes is None:
                available_memory_bytes = _cuda_available_memory_bytes(
                    torch, self.device
                )
        self.last_tile_plan = _simulation_tile_plan(
            nphotons,
            self.tile_size,
            available_memory_bytes=available_memory_bytes,
        )
        tile_capacity = self.last_tile_plan.tile_capacity
        use_reservoir = self.reservoir_rounds is not None and not keep_final_states
        if not use_reservoir:
            for first in range(0, nphotons, tile_capacity):
                count = min(tile_capacity, nphotons-first)
                tape_rows = (
                    torch.arange(
                        first, first + count,
                        dtype=torch.int32, device=self.device,
                    )
                    if use_random_tape else None
                )
                hits, state, rounds, events, trace = self._simulate_tile(
                    count, center, voxel_size, seed, first, max_steps,
                    random_tape=random_tape,
                    tape_audit=tape_audit,
                    tape_row_indices=tape_rows,
                    tape_trace=boundary_tape_trace,
                    tape_certificate=tape_certificate,
                    state_certificate=state_certificate,
                )
                hit_times.append(hits.time)
                hit_channels.append(hits.channel)
                dense_rounds += rounds
                boundary_events += events
                self.last_trace.extend(trace)
                tiles += 1
                if final_states is not None:
                    final_states.append(state)
        else:
            reservoir_state_chunks = []
            reservoir_id_chunks = []
            dense_limit = min(int(self.reservoir_rounds), int(max_steps))
            for first in range(0, nphotons, tile_capacity):
                count = min(tile_capacity, nphotons-first)
                state = self._new_state(
                    count, center, voxel_size, seed, first
                )
                tape_rows = (
                    torch.arange(
                        first, first + count,
                        dtype=torch.int32, device=self.device,
                    )
                    if use_random_tape else None
                )
                pending = torch.arange(
                    count, dtype=torch.int32, device=self.device
                )
                pending, rounds, events, trace = self._propagate_state(
                    state,
                    pending,
                    seed,
                    first,
                    max_steps,
                    dense_limit,
                    random_tape=random_tape,
                    tape_audit=tape_audit,
                    tape_row_indices=tape_rows,
                    tape_trace=boundary_tape_trace,
                    tape_certificate=tape_certificate,
                    state_certificate=state_certificate,
                )
                hits = self._state_hits(state)
                hit_times.append(hits.time)
                hit_channels.append(hits.channel)
                dense_rounds += rounds
                boundary_events += events
                self.last_trace.extend(trace)
                compacted = self._compact_reservoir_state(
                    state, pending, first
                )
                if compacted is not None:
                    compacted_state, compacted_ids = compacted
                    reservoir_state_chunks.append(compacted_state)
                    reservoir_id_chunks.append(compacted_ids)
                    reservoir_photons += int(compacted_ids.numel())
                tiles += 1

            if reservoir_state_chunks:
                if len(reservoir_state_chunks) == 1:
                    state = reservoir_state_chunks[0]
                    global_photon_ids = reservoir_id_chunks[0]
                else:
                    state = tuple(
                        torch.cat(parts, dim=0)
                        for parts in zip(*reservoir_state_chunks)
                    )
                    global_photon_ids = torch.cat(
                        reservoir_id_chunks, dim=0
                    ).contiguous()
                # Drop chunk containers before allocating boundary scratch for
                # the pooled drain.  The combined tensors (or the single
                # selected tuple) remain referenced by ``state``.
                del reservoir_state_chunks, reservoir_id_chunks
                if "compacted_state" in locals():
                    del compacted_state, compacted_ids, compacted
                pending = torch.arange(
                    reservoir_photons, dtype=torch.int32, device=self.device
                )
                reservoir_tape_rows = (
                    global_photon_ids.to(torch.int32).contiguous()
                    if use_random_tape else None
                )
                pending, rounds, events, trace = self._propagate_state(
                    state,
                    pending,
                    seed,
                    0,
                    max_steps,
                    max_steps,
                    global_photon_ids=global_photon_ids,
                    random_tape=random_tape,
                    tape_audit=tape_audit,
                    tape_row_indices=reservoir_tape_rows,
                    tape_trace=boundary_tape_trace,
                    tape_certificate=tape_certificate,
                    state_certificate=state_certificate,
                )
                if pending.numel():
                    state[4][pending.to(torch.int64)] |= (1 << 0)
                hits = self._state_hits(state)
                hit_times.append(hits.time)
                hit_channels.append(hits.channel)
                reservoir_rounds += rounds
                boundary_events += events
                self.last_trace.extend(trace)
        torch.cuda.synchronize(self.device)
        elapsed = _time.perf_counter() - started
        if self.chroma_mesh_box_compatibility:
            if self.chroma_global_workspace.sticky_overflowed():
                raise RuntimeError(
                    "Chroma global BVH traversal stack overflowed; event "
                    "output is invalid"
                )
        elif self.pmt_workspace.sticky_overflowed():
            raise RuntimeError(
                "PMT traversal stack overflowed; event output is invalid"
            )
        if hit_times:
            times = torch.cat(hit_times) if len(hit_times) > 1 else hit_times[0]
            channels = (
                torch.cat(hit_channels) if len(hit_channels) > 1 else hit_channels[0]
            )
        else:
            times = torch.empty(0, dtype=torch.float32, device=self.device)
            channels = torch.empty(0, dtype=torch.int32, device=self.device)
        flat_hits = FlatHits(times, channels)
        stats = SimulationStats(
            photons=nphotons,
            detections=len(flat_hits),
            tiles=tiles,
            boundary_rounds=dense_rounds + reservoir_rounds,
            boundary_events=boundary_events,
            elapsed_seconds=elapsed,
            dense_rounds=dense_rounds,
            reservoir_rounds=reservoir_rounds,
            reservoir_photons=reservoir_photons,
        )
        return TritonSimulationResult(
            flat_hits=flat_hits,
            stats=stats,
            final_states=None if final_states is None else tuple(final_states),
            tape_audit=tape_audit if use_random_tape else None,
            tape_certificate=tape_certificate if use_random_tape else None,
            state_certificate=state_certificate if use_random_tape else None,
            boundary_tape_trace=boundary_tape_trace,
        )


__all__ = [
    "AUTO_TILE_ALIGNMENT",
    "AUTO_TILE_BYTES_PER_PHOTON",
    "AUTO_TILE_RESERVE_BYTES",
    "AUTO_TILE_USABLE_FRACTION",
    "BoundaryTapeTrace",
    "FlatHits",
    "Reflect3WiresTritonSimulation",
    "SimulationTilePlan",
    "SimulationStats",
    "STATE_CERTIFICATE_WIDTH",
    "TAPE_PROCESS_BULK_ABSORB",
    "TAPE_PROCESS_BULK_SCATTER",
    "TAPE_PROCESS_DIELECTRIC_REFLECT",
    "TAPE_PROCESS_DIELECTRIC_TRANSMIT",
    "TAPE_PROCESS_SURFACE_ABSORB",
    "TAPE_PROCESS_SURFACE_DETECT",
    "TAPE_PROCESS_SURFACE_DIFFUSE",
    "TAPE_PROCESS_SURFACE_SPECULAR",
    "TAPE_PROCESS_UNSET",
    "TritonSimulationResult",
]
