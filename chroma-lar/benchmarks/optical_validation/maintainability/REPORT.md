# Optical engine maintainability refactor

Separated device table compilation, named state contracts, geometry queries, transport scheduling and shared readout. The fast path no longer constructs the legacy monochromatic transport engine. Public simulation entry points and NPZ output fields are retained. The source/bulk/surface physics kernels were unchanged.

See [architecture and package feasibility](../../../docs/optical_architecture.md).

## Verification

- **167 tests passed**, with zero failures, errors or skips ([log](tests_final.log)).
- **749,196 photons, three seeds, 81 arrays/metadata records:** exact equality to the saved pre-refactor outputs. Includes terminal positions, directions, polarization, times, wavelengths, flags, IDs, hits, PE, noisy ADC waveforms and metadata ([record](equivalence.json), [reproduction script](../../check_optical_refactor.py)).
- **CUDA memcheck: zero errors** across the 16 fast optical tests and the device geometry parity test ([log](memcheck.log)).
- All five large benchmark events retained exactly the same photon counts, detections, PE counts, diagnostics and waveform dimensions as the earlier implementation.

## Local A100 throughput

Same Poisson source, full detector, synthetic calibration, nonzero electronics noise and ADC; source generation through CPU-readable hit/PE/waveform output. Setup, JIT warmup, file writes and optional terminal-state downloads are excluded.

| Implementation | Sustained M photons/s | Slowest repeat | Fastest repeat |
|---|---:|---:|---:|
| Before refactor (earlier run) | 25.648 | 24.105 | 26.936 |
| After refactor | 25.498 | 23.905 | 26.427 |

Five measured events: 149,998,442 photons in 5.882764 s. These are separate measurements, not an interleaved performance experiment. Current runtime hashes match [the new benchmark record](performance_30m.json).

The baseline capture was made before editing runtime sources. The final comparison was rerun after the refactor. The first regression run found a stale renamed-table reference in a test; that reference was corrected before the passing final suite.

This establishes regression evidence for these configurations, not universal correctness or measured detector accuracy. The [original full validation report](../FAST_FULL_REPORT.md) retains the native-CUDA discrepancy and model limitations. No packaging migration was performed.
