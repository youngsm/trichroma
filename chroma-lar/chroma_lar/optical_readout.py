"""Shared PMT response, digitization and result provenance for optical engines."""

from dataclasses import dataclass
import json
import time
from typing import Protocol

import numpy as np
from chroma.triton.optical_response import OpticalHits, Photoelectrons, Waveforms, digitize


class OpticalTransportResult(Protocol):
    """The result fields required by readout, independent of transport storage."""

    hits: OpticalHits
    scene_fingerprint: str
    step_limit_count: int


@dataclass(frozen=True)
class OpticalEventResult:
    transport: OpticalTransportResult
    photoelectrons: Photoelectrons
    waveforms: Waveforms
    metadata: dict

    def save(self, path):
        """Portable NPZ output with explicit calibration and scene provenance."""
        pe, hits, wf = self.photoelectrons, self.transport.hits, self.waveforms
        np.savez_compressed(
            path,
            waveform_events=wf.event_indices,
            sample_times_ns=wf.sample_times,
            waveforms_adc=wf.samples,
            pe_times_ns=pe.times,
            pe_charge=pe.charges,
            pe_channels=pe.channels,
            pe_events=pe.event_indices,
            pe_photon_ids=pe.photon_ids,
            optical_times_ns=hits.times,
            optical_channels=hits.channels,
            optical_wavelengths_nm=hits.wavelengths,
            optical_photon_ids=hits.photon_ids,
            optical_events=hits.event_indices,
            metadata_json=json.dumps(self.metadata, sort_keys=True),
        )


class OpticalReadout:
    """Turn optical hits into host-readable PE and event waveforms.

    device=None selects the CPU digitizer; a CUDA device selects its GPU
    implementation. PMT response, event membership and metadata are shared.
    Rebuild the pipeline when changing calibration.
    """

    def __init__(self, calibration, channel_count: int, *, device=None):
        self.calibration = calibration
        self.channel_count = channel_count
        self.device = device

    def apply(
        self,
        result: OpticalTransportResult,
        event_indices,
        *,
        seed,
        max_steps,
        backend,
        photon_count,
        diagnostics=None,
        timings=None
    ):
        started = time.perf_counter()
        pe = self.calibration.response.apply(result.hits, seed=seed)
        responded = time.perf_counter()
        if self.device is None:
            digitizer, options = digitize, {}
        else:
            from chroma.triton.digitizer_kernels import digitize_gpu

            digitizer, options = digitize_gpu, {"device": self.device}
        waveforms = digitizer(
            pe,
            event_indices=event_indices,
            channel_count=self.channel_count,
            seed=seed,
            **options,
            **self.calibration.digitizer
        )
        completed = time.perf_counter()
        if timings is not None:
            timings.append(
                {
                    "pmt_response_seconds": responded - started,
                    "digitization_seconds": completed - responded,
                }
            )
        metadata = {
            "schema_version": 1,
            "calibration_fingerprint": self.calibration.fingerprint,
            "calibration_status": self.calibration.manifest["status"],
            "calibration_provenance": self.calibration.manifest["provenance"],
            "calibration_files": self.calibration.files,
            "scene_fingerprint": result.scene_fingerprint,
            "backend": backend,
            "seed": int(seed),
            "max_steps": int(max_steps),
            "rng": "philox4x32-10-id64-stream32-seed64-v1",
            "photons": photon_count,
            "detected_photons": len(result.hits),
            "photoelectrons": len(pe.times),
            "step_limit_count": result.step_limit_count,
            "pe_before_sample_window": int(np.count_nonzero(pe.times < waveforms.sample_times[0])),
            "pe_after_sample_window": int(np.count_nonzero(pe.times > waveforms.sample_times[-1])),
            "units": {
                "position": "mm",
                "wavelength": "nm",
                "time": "ns",
                "charge": "PE",
                "waveform": "ADC",
            },
        }
        if diagnostics is not None:
            metadata["transport_diagnostics"] = diagnostics
        return OpticalEventResult(result, pe, waveforms, metadata)
