"""End-to-end scintillation, spectral propagation, PMT response, and waveforms."""

from __future__ import annotations

import numpy as np

from .optical_readout import OpticalEventResult, OpticalReadout
from chroma.triton.spectral import SpectralSimulation

__all__ = ["OpticalEventResult", "OpticalSimulation", "FastOpticalSimulation"]


class OpticalSimulation:
    def __init__(self, geometry, calibration, *, backend="triton", device=None, tile_size=65536):
        self.calibration = calibration
        self.transport = SpectralSimulation(
            geometry,
            wavelengths=calibration.wavelengths,
            backend=backend,
            device=device,
            tile_size=tile_size,
        )
        self.readout = OpticalReadout(calibration, self.transport.scene.host.channel_count)

    def simulate_photons(
        self,
        photons,
        *,
        event_indices=None,
        seed=1,
        max_steps=1000,
        photon_id_base=0,
        allow_step_limit=False,
    ):
        events = photons.event_indices if hasattr(photons, "event_indices") else photons.evidx
        if event_indices is None:
            event_indices = np.unique(events)
        elif not np.isin(events, event_indices).all():
            raise ValueError(
                "event_indices must include every input photon event, including undetected events"
            )
        result = self.transport.simulate(
            photons, seed=seed, max_steps=max_steps, photon_id_base=photon_id_base
        )
        if result.step_limit_count and not allow_step_limit:
            raise RuntimeError(
                f"{result.step_limit_count} photons reached max_steps; increase the limit or explicitly allow truncation"
            )
        return self.readout.apply(
            result,
            event_indices,
            seed=seed,
            max_steps=max_steps,
            backend=self.transport.backend,
            photon_count=len(result.final_state["times"]),
        )

    def simulate_depositions(
        self,
        positions,
        energy_mev,
        *,
        times=0.0,
        event_indices=0,
        quenching=1.0,
        seed=1,
        max_steps=1000,
        max_photons=10_000_000,
    ):
        photons = self.calibration.source.from_depositions(
            positions,
            energy_mev,
            times=times,
            event_indices=event_indices,
            quenching=quenching,
            seed=seed,
            max_photons=max_photons,
        )
        # Deposition events with zero emitted/detected photons still get a waveform.
        events = np.unique(np.broadcast_to(event_indices, (len(positions),)))
        return self.simulate_photons(photons, event_indices=events, seed=seed, max_steps=max_steps)


class FastOpticalSimulation:
    """Calibrated full optical pipeline using analytic/instanced GPU transport.

    Optical hits, photoelectrons and all requested event waveforms are returned
    on the host. Full terminal photon states are an optional debugging output.
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
    ):
        from .spectral_backend import SpectralDetectorSimulation

        self.calibration = calibration
        self.transport = SpectralDetectorSimulation(
            calibration,
            config_name=config_name,
            device=device,
            history_length=history_length,
            epochs_per_poll=epochs_per_poll,
            block_size=block_size,
            fused_pmt=fused_pmt,
        )
        self.device = device
        self.readout = OpticalReadout(calibration, self.transport.channel_count, device=device)

    def _response(self, result, event_indices, seed, max_steps, timings=None):
        return self.readout.apply(
            result,
            event_indices,
            seed=seed,
            max_steps=max_steps,
            backend="triton_spectral_detector",
            photon_count=result.photon_count,
            diagnostics=result.diagnostics,
            timings=timings,
        )

    def simulate_voxel(
        self,
        count,
        center=(-1000.0, 0.0, 0.0),
        *,
        voxel_size=30.0,
        event_id=0,
        event_indices=None,
        seed=1,
        max_steps=2048,
        keep_final_states=False,
        timings=None,
    ):
        events = [event_id] if event_indices is None else event_indices
        if event_id not in events:
            raise ValueError("event_indices must contain the photon source event")
        result = self.transport.simulate_voxel(
            count,
            center,
            voxel_size=voxel_size,
            event_id=event_id,
            seed=seed,
            max_steps=max_steps,
            keep_final_states=keep_final_states,
            timings=timings,
        )
        return self._response(result, events, seed, max_steps, timings)

    def simulate_photons(
        self, photons, *, event_indices=None, seed=1, max_steps=2048, keep_final_states=False
    ):
        events = photons.event_indices if hasattr(photons, "event_indices") else photons.evidx
        if event_indices is None:
            event_indices = np.unique(events)
        elif not np.isin(events, event_indices).all():
            raise ValueError("event_indices must contain every input photon event")
        result = self.transport.simulate(
            photons, seed=seed, max_steps=max_steps, keep_final_states=keep_final_states
        )
        return self._response(result, event_indices, seed, max_steps)

    def simulate_depositions(
        self,
        positions,
        energy_mev,
        *,
        times=0.0,
        event_indices=0,
        quenching=1.0,
        seed=1,
        max_steps=2048,
        max_photons=100_000_000,
        photon_id_base=0,
        keep_final_states=False,
        timings=None,
    ):
        result = self.transport.simulate_depositions(
            positions,
            energy_mev,
            times=times,
            event_indices=event_indices,
            quenching=quenching,
            seed=seed,
            max_steps=max_steps,
            max_photons=max_photons,
            photon_id_base=photon_id_base,
            keep_final_states=keep_final_states,
            timings=timings,
        )
        events = np.unique(np.broadcast_to(event_indices, (len(positions),)))
        return self._response(result, events, seed, max_steps, timings)
