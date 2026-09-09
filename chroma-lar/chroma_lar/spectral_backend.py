"""Calibrated spectral transport on the existing analytic/instanced detector."""

from __future__ import annotations
from dataclasses import dataclass
import time

import numpy as np

from chroma.event import (
    NO_HIT,
    BULK_ABSORB,
    SURFACE_DETECT,
    SURFACE_ABSORB,
    SURFACE_REEMIT,
    NAN_ABORT,
)
from chroma.triton.optical_response import OpticalHits, validate_seed
from chroma.triton.photon_input import as_photon_batch
from .triton_scene.compiler import compile_reflect3wires_scene
from .triton_scene.detector_query import DetectorBoundaryQuery
from .spectral_model import CompiledSpectralModel
from .spectral_state import PhotonState, TransportSettings

STEP_LIMIT = 1 << 30
TERMINAL_FLAGS = NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | STEP_LIMIT | NAN_ABORT


@dataclass(frozen=True)
class FastSpectralResult:
    hits: OpticalHits
    photon_count: int
    steps: int  # Geometry rounds; per-photon interaction counts are final_state["steps"].
    step_limit_count: int
    scene_fingerprint: str
    diagnostics: dict
    final_state: dict[str, np.ndarray] | None = None


class SpectralDetectorSimulation:
    """Full wavelength-dependent optics; geometry acceleration changes no tables.

    The full scene retains both detector halves and the outer cavity. This
    source adapter accepts the original negative-x component and source boxes
    inside compiler-certified LAr regions, including the opposite component.
    PMT coating and glass remain explicit triangle boundaries.
    """

    def __init__(
        self,
        calibration,
        *,
        config_name="detector_config_reflect_reflect3wires",
        device="cuda",
        history_length=8,
        epochs_per_poll=2,
        block_size=128,
        fused_pmt=False,
        region_mode="automatic",
    ):
        import torch

        self.settings = TransportSettings(history_length, epochs_per_poll, block_size, fused_pmt)
        self.torch, self.device = torch, torch.device(device)
        self.calibration = calibration
        self.scene = compile_reflect3wires_scene(
            config_name, calibration=calibration, retain_all_geometry=True
        )
        self.lar_index = self.scene.tables.material_names.index(
            calibration.manifest["roles"]["lar"]
        )
        self.query = DetectorBoundaryQuery(
            self.scene,
            device=self.device,
            fused_pmt=fused_pmt,
            region_mode=region_mode,
            bulk_material=self.lar_index,
        )
        self.model = CompiledSpectralModel(self.scene, calibration, self.device)
        self.fingerprint = self.model.fingerprint
        self.channel_count = self.scene.total_reference_channels

    def _empty_state(self, count):
        return PhotonState.allocate(count, self.device)

    def _source(self, state, count, center, voxel_size, seed, photon_id_base, event_id, birth):
        if count == 0:
            return
        import triton
        from .spectral_kernels import source_kernel

        source_kernel[(triton.cdiv(count, 256),)](
            *state,
            *self.model.source,
            count,
            seed,
            photon_id_base,
            event_id,
            *map(float, center),
            float(voxel_size),
            float(birth),
            float(self.calibration.source.rise_time_ns),
            NC=len(self.calibration.source.fractions),
            BLOCK=256,
            enable_fp_fusion=False,
        )

    def _check_source_box(self, center, voxel_size):
        center = np.asarray(center, float)
        if (
            center.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(voxel_size)
            or voxel_size < 0
        ):
            raise ValueError("source center/voxel must be finite with nonnegative voxel size")
        lo, hi = (
            self.scene.reachability.source_component_bounds_min,
            self.scene.reachability.source_component_bounds_max,
        )
        if np.all(center - voxel_size / 2 > lo) and np.all(center + voxel_size / 2 < hi):
            return
        regions = () if self.query.regions is None else self.query.regions.regions
        for region in regions:
            if (
                region.material == self.lar_index
                and np.all(center - voxel_size / 2 > region.bounds.lower)
                and np.all(center + voxel_size / 2 < region.bounds.upper)
            ):
                return
        raise ValueError("source must be within the negative-x component or a certified LAr region")

    def simulate_voxel(
        self,
        count,
        center=(-1000.0, 0.0, 0.0),
        *,
        voxel_size=30.0,
        seed=1,
        event_id=0,
        birth=0.0,
        photon_id_base=0,
        max_steps=2048,
        keep_final_states=False,
        timings=None,
    ):
        seed = validate_seed(seed)
        if not isinstance(count, (int, np.integer)) or not 0 <= count < 2**31:
            raise ValueError("count must be a nonnegative integer less than 2**31")
        if (
            not isinstance(photon_id_base, (int, np.integer))
            or photon_id_base < 0
            or photon_id_base + count > 2**63
        ):
            raise ValueError("invalid photon ID range")
        if (
            not isinstance(event_id, (int, np.integer))
            or not 0 <= event_id <= 2**32 - 1
            or not np.isfinite(birth)
        ):
            raise ValueError("event ID and birth time are invalid")
        self._check_source_box(center, voxel_size)
        self.query.select_bulk_region([center])
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        state = self._empty_state(count)
        self._source(state, count, center, voxel_size, seed, photon_id_base, event_id, birth)
        self.torch.cuda.synchronize(self.device)
        sourced = time.perf_counter()
        result = self._transport(state, seed, max_steps, keep_final_states)
        if timings is not None:
            timings.append(
                {
                    "photons": count,
                    "source_seconds": sourced - started,
                    "transport_output_seconds": time.perf_counter() - sourced,
                    "event_seconds": time.perf_counter() - started,
                }
            )
        return result

    def simulate(self, photons, *, seed=1, max_steps=2048, keep_final_states=False):
        batch = as_photon_batch(photons)
        # Selection is only a performance choice. Sample a bounded number of
        # representatives; unselected photons retain the exact boundary path.
        stride = max(1, batch.photon_count // 1024)
        self.query.select_bulk_region(batch.pos[::stride])
        lo, hi = (
            self.scene.reachability.source_component_bounds_min,
            self.scene.reachability.source_component_bounds_max,
        )
        valid_sources = np.all((batch.pos > lo) & (batch.pos < hi), axis=1)
        if not valid_sources.all() and self.query.regions is not None:
            for region in self.query.regions.regions:
                if region.material == self.lar_index:
                    valid_sources |= region.bounds.contains(batch.pos)
        if not valid_sources.all():
            raise ValueError(
                "input photons must start in the negative-x component or a certified LAr region"
            )
        if np.any(batch.last_hit_triangles != -1) or np.any(batch.flags & TERMINAL_FLAGS):
            raise ValueError("the detector source adapter requires fresh, live photons")
        if np.any(batch.wavelengths < self.calibration.wavelengths[0]) or np.any(
            batch.wavelengths > self.calibration.wavelengths[-1]
        ):
            raise ValueError("source wavelength is outside the calibration grid")
        state = self._empty_state(batch.photon_count)
        for name, value in (
            ("pos", batch.pos),
            ("direction", batch.direction),
            ("polarization", batch.polarization),
            ("times", batch.times),
            ("flags", batch.flags.view(np.int32)),
            ("photon_ids", batch.global_photon_ids),
            ("wavelengths", batch.wavelengths),
            ("event_indices", batch.event_indices.astype(np.int64)),
        ):
            getattr(state, name).copy_(
                self.torch.from_numpy(np.array(value, copy=True)).to(self.device)
            )
        for value in (state.last_instance, state.last_hit, state.channels):
            value.fill_(-1)
        state.steps.zero_()
        return self._transport(state, validate_seed(seed), max_steps, keep_final_states)

    def simulate_depositions(
        self,
        positions,
        energy_mev,
        *,
        times=0.0,
        event_indices=0,
        quenching=1.0,
        seed=1,
        photon_id_base=0,
        max_photons=100_000_000,
        max_steps=2048,
        keep_final_states=False,
        timings=None,
    ):
        seed = validate_seed(seed)
        self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        positions, births, events, counts = self.calibration.source.sample_depositions(
            positions,
            energy_mev,
            times=times,
            event_indices=event_indices,
            quenching=quenching,
            seed=seed,
            max_photons=max_photons,
        )
        total = int(counts.sum())
        if (
            total >= 2**31
            or not isinstance(photon_id_base, (int, np.integer))
            or photon_id_base < 0
            or photon_id_base + total > 2**63
        ):
            raise ValueError("photon population/ID range exceeds the GPU source bounds")
        for position, n in zip(positions, counts):
            if n:
                self._check_source_box(position, 0.0)
        self.query.select_bulk_region(positions, weights=counts)
        state = self._empty_state(total)
        first = 0
        for position, birth, event, n in zip(positions, births, events, counts):
            n = int(n)
            self._source(
                state.slice(first, first + n),
                n,
                position,
                0.0,
                seed,
                photon_id_base + first,
                int(event),
                float(birth),
            )
            first += n
        self.torch.cuda.synchronize(self.device)
        sourced = time.perf_counter()
        result = self._transport(state, seed, max_steps, keep_final_states)
        if timings is not None:
            timings.append(
                {
                    "photons": total,
                    "source_seconds": sourced - started,
                    "transport_output_seconds": time.perf_counter() - sourced,
                    "event_seconds": time.perf_counter() - started,
                }
            )
        return result

    def _transport(self, state: PhotonState, seed, max_steps, keep_final_states):
        import triton
        from chroma.triton.transport import DeviceQueue
        from .spectral_kernels import bulk_epoch, boundary_step

        if not isinstance(max_steps, (int, np.integer)) or not 1 <= max_steps < 2**22:
            raise ValueError("max_steps must be an integer in [1, 2**22)")
        torch = self.torch
        count = len(state.times)
        queues = [DeviceQueue.allocate(count, device=self.device) for _ in range(2)]
        boundary = DeviceQueue.allocate(count, device=self.device)
        current = queues[0]
        current.buffer.copy_(torch.arange(count, dtype=torch.int32, device=self.device))
        current.count.fill_(count)
        capacity = count
        rounds = 0
        self.query.begin_event()
        grid = self.model.optics.wavelength_grid
        while capacity:
            boundary.reset()
            while True:
                for _ in range(self.settings.epochs_per_poll):
                    output = queues[1] if current is queues[0] else queues[0]
                    output.reset()
                    bulk_epoch[(triton.cdiv(capacity, self.settings.block_size),)](
                        *state,
                        current.buffer,
                        current.count,
                        capacity,
                        output.buffer,
                        output.count,
                        boundary.buffer,
                        boundary.count,
                        self.model.properties.absorption_length,
                        self.model.properties.scattering_length,
                        self.model.properties.group_velocity,
                        seed,
                        max_steps,
                        *map(float, self.query.safe_lower),
                        *map(float, self.query.safe_upper),
                        float(grid.start),
                        float(grid.step),
                        LAR=self.lar_index,
                        NW=grid.count,
                        HISTORY=self.settings.history_length,
                        BLOCK=self.settings.block_size,
                        enable_fp_fusion=False,
                    )
                    current = output
                nboundary, ncontinuing = torch.cat((boundary.count, current.count)).cpu().tolist()
                if not 0 <= nboundary + ncontinuing <= count:
                    raise RuntimeError("spectral queue partition overflow")
                if not ncontinuing:
                    break
                capacity = ncontinuing
            if not nboundary:
                break
            resolved = self.query.resolve(state, boundary, nboundary)
            hit, analytic = resolved.hit, resolved.analytic
            current = queues[0].reset()
            boundary_step[(triton.cdiv(nboundary, self.settings.block_size),)](
                *state,
                boundary.buffer,
                boundary.count,
                nboundary,
                current.buffer,
                current.count,
                *hit,
                self.query.scene_device["pmt_scene_material1_index"],
                *self.model.properties,
                seed,
                max_steps,
                float(grid.start),
                float(grid.step),
                NW=grid.count,
                BLOCK=self.settings.block_size,
                analytic_index=analytic.index,
                analytic_primitive=analytic.primitive_index,
                analytic_outward=analytic.outward_normal,
                wire_geometry=self.model.wire_geometry,
                box_bounds=self.model.box_bounds,
                PROJECT_ANALYTIC=True,
                enable_fp_fusion=False,
            )
            capacity = current.size()
            rounds += 1
        torch.cuda.synchronize(self.device)
        self.query.check_event()
        return self._collect_result(state, rounds, keep_final_states)

    def _collect_result(self, state: PhotonState, rounds, keep_final_states):
        """Audit every photon before compacting detected hits onto the host."""
        torch = self.torch
        h = state.flags
        diagnostic = {
            name: int(((h & mask) != 0).sum().item())
            for name, mask in (
                ("detected", SURFACE_DETECT),
                ("reemitted", SURFACE_REEMIT),
                ("escaped", NO_HIT),
                ("step_limit", STEP_LIMIT),
                ("aborted", NAN_ABORT),
            )
        }
        diagnostic["unfinished"] = int(((h & TERMINAL_FLAGS) == 0).sum().item())
        diagnostic["nonfinite"] = sum(
            int((~torch.isfinite(value)).sum().item())
            for value in (
                state.pos,
                state.direction,
                state.polarization,
                state.times,
                state.wavelengths,
            )
        )
        if (
            diagnostic["nonfinite"]
            or diagnostic["unfinished"]
            or diagnostic["aborted"]
            or diagnostic["step_limit"]
        ):
            bad = ((h & (STEP_LIMIT | NAN_ABORT)) != 0) | ((h & TERMINAL_FLAGS) == 0)
            for value in (state.pos, state.direction, state.polarization):
                bad |= ~torch.isfinite(value).all(1)
            bad |= ~torch.isfinite(state.times) | ~torch.isfinite(state.wavelengths)
            sample = torch.nonzero(bad).flatten()[:8]
            self.last_failure = {
                name: value[sample].cpu().numpy().tolist()
                for name, value in (
                    ("position", state.pos),
                    ("direction", state.direction),
                    ("photon_ids", state.photon_ids),
                    ("last_instance", state.last_instance),
                    ("last_triangle", state.last_hit),
                    ("steps", state.steps),
                )
            }
            self.last_failure["diagnostics"] = diagnostic
            raise RuntimeError(f"invalid spectral transport result: {diagnostic}")
        selected = torch.nonzero((h & SURFACE_DETECT) != 0).flatten()
        hits = OpticalHits(
            state.times[selected].cpu().numpy(),
            state.channels[selected].cpu().numpy(),
            state.photon_ids[selected].cpu().numpy(),
            state.event_indices[selected].cpu().numpy(),
            state.wavelengths[selected].cpu().numpy(),
        )
        final = state.to_host() if keep_final_states else None
        return FastSpectralResult(
            hits,
            len(state.times),
            rounds,
            diagnostic["step_limit"],
            self.fingerprint,
            diagnostic,
            final,
        )
