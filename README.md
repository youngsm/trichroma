# TriChroma

Triton optical photon transport, detector acceleration, and calibrated optical
readout, consolidated from the Chroma-Lite and Chroma-LAr working trees.

The full detector pipeline measured **25.50 million input photons/s sustained**
on an NVIDIA A100 40 GB across five approximately 30M-photon events. This includes
scintillation generation, wavelength-dependent transport, effective TPB
reemission, PMT collection/TTS/charge, and CPU-readable waveforms with electronics
noise and ADC processing. Construction, JIT warmup, file writes, and optional
downloads of every terminal photon are excluded.

The calibration used for validation is synthetic. This repository preserves the
documented numerical fixes, comparison discrepancies, and implementation limits.
The current fast engine is specialized for the included detector. Full
`chroma.sim.Simulation` compatibility and a general primitive/region compiler are
future work; existing Triton entry points are explicit.

## Quick start

Use a Linux CUDA environment. The recorded runs used Python 3.10.20,
PyTorch 2.5.0+cu124, Triton 3.1.0, and an A100-SXM4-40GB.

```bash
git clone https://github.com/youngsm/trichroma.git
cd trichroma
python -m pip install -r requirements-triton.txt
export PYTHONPATH="$PWD/chroma-lite:$PWD/chroma-lar"
```

This snapshot retains the two source directories so existing imports, scripts,
and recorded source hashes remain valid. The Triton path does not require
PyCUDA. The nested legacy `setup.py` files describe their original distributions;
the source-path setup above is the supported route for this consolidated snapshot.

Run the complete synthetic optical example:

```bash
python chroma-lar/benchmarks/run_optical_simulation.py \
  --demo --backend triton --output /tmp/optical-demo.npz
```

Run the accelerated full detector benchmark:

```bash
python chroma-lar/benchmarks/benchmark_fast_full.py \
  --calibration chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration_noise.json \
  --depositions --counts 30000000 --repeats 5 \
  --output /tmp/trichroma-throughput.json
```

See the [optical API and CLI guide](chroma-lar/docs/full_optical_simulation.md)
for energy-deposition and prepared-photon inputs, calibration files, and outputs.

## Validation

```bash
python scripts/check_optics.py --output-dir /tmp/trichroma-checks
python chroma-lar/benchmarks/check_optical_refactor.py compare
```

The first command requires a local CUDA GPU and runs the focused optical,
geometry, source, readout, and scheduling suite. The second checks exact
equivalence to the included pre-refactor baseline for 749,196 photons over three
seeds, including full terminal states, hits, photoelectrons, and noisy ADC waveforms.

- [Maintainability refactor: 167 tests, zero memcheck errors, and throughput](chroma-lar/benchmarks/optical_validation/maintainability/REPORT.md)
- [Full detector validation, analytic boundary fix, and native CUDA comparison](chroma-lar/benchmarks/optical_validation/FAST_FULL_REPORT.md)
- [Earlier generic mesh comparison and unresolved mesh-path issues](chroma-lar/benchmarks/optical_validation/REPORT.md)
- [Consolidated checkout verification](docs/consolidation_validation.md)

## Code organization

| Location | Contents |
|---|---|
| `chroma-lite/chroma/triton/` | General optical tables, photon/state adapters, physics kernels, mesh transport, and readout implementations |
| `chroma-lar/chroma_lar/` | Calibration, optical pipeline, compiled spectral model, named device state, and transport scheduling |
| `chroma-lar/chroma_lar/triton_scene/` | Detector compiler, analytic intersections, shared-mesh instances, and geometry queries |
| `chroma-lar/benchmarks/` | Benchmark/comparison tools, recorded reports, and regression fixtures |
| `chroma-lite/test/`, `chroma-lar/test/` | Reference, GPU, and integration tests |

[Architecture and extension points](chroma-lar/docs/optical_architecture.md)
describe the current component boundaries, detector-specific assumptions, and
the work required for a standalone Chroma-compatible package. The proposed
general primitive compiler and automatic homogeneous-region discovery have
not been implemented in this snapshot.

## Provenance and artifacts

This repository imports the source snapshots as ordinary files, without nested
Git repositories or submodules. [The consolidation manifest](docs/consolidation_manifest.json)
records the upstream repository revisions, hashes of copied files, and an
inventory of omitted large generated photon/BVH outputs. Source changes and
previous working directories were preserved. The required regression baseline,
small fixtures, calibration files, logs, and reports are included. Larger
generated outputs can be recreated with the accompanying benchmark scripts.

Chroma originated with Anthony LaTorre and Stanley Seibert. The Chroma components
retain their [original license](chroma-lite/LICENSE.txt). This snapshot includes
the local work based on [youngsm/chroma-lite](https://github.com/youngsm/chroma-lite)
and [youngsm/chroma-lar](https://github.com/youngsm/chroma-lar).
