# TriChroma

Triton optical photon transport, detector acceleration, and calibrated optical
readout, consolidated from the Chroma-Lite and Chroma-LAr working trees.

The full detector pipeline with automatically discovered bulk regions measured
**25.46 million input photons/s sustained**
on an NVIDIA A100 40 GB across five approximately 30M-photon events. This includes
scintillation generation, wavelength-dependent transport, effective TPB
reemission, PMT collection/TTS/charge, and CPU-readable waveforms with electronics
noise and ADC processing. Construction, JIT warmup, file writes, and optional
downloads of every terminal photon are excluded.

The calibration used for validation is synthetic. This repository preserves the
documented numerical fixes, comparison discrepancies, and implementation limits.
The current fast engine is specialized for the included detector. Full
`chroma.sim.Simulation` compatibility remains future work; this experimental
branch adds a primitive region compiler, with explicit Triton entry points.

## Quick start

Use a Linux CUDA environment. The recorded runs used Python 3.10.20,
PyTorch 2.5.0+cu124, Triton 3.1.0, and an A100-SXM4-40GB.

```bash
git clone https://github.com/youngsm/trichroma.git
cd trichroma
git switch feature/primitive-regions
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
- [Primitive compiler, current detector comparisons and browser rendering](chroma-lar/benchmarks/optical_validation/primitive_regions/REPORT.md)
- [Original Chroma raw-word parity and its precise limits](docs/legacy_parity.md)

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
the work required for a standalone Chroma-compatible package. This experimental
branch adds [automatic primitive-region discovery](docs/primitive_regions.md),
a Theia-like water Cherenkov fixture, and a [Jupyter detector viewer](docs/jupyter_viewer.md).
The region compiler now feeds the fast LAr path; the water optical example uses
the existing general mesh transport. General fast boundary lowering remains
future work.

For wavelength-dependent physics, open [photon_camera.ipynb](notebooks/photon_camera.ipynb)
with the **TriChroma (GPU, pimm-bench)** kernel. Its WebGPU camera renders a
dispersive prism, fluorescent glass, Rayleigh scattering, and a TPB-coated PMT
with an optional cutaway. Choose the photon count and orbit the scene; the
browser simulates optical photons and forms an illuminated camera image.
The [camera guide](docs/photon_camera.md) describes the physical models and
rendering approximations. The portable bundle runs over localhost or HTTPS
without a Python rendering server.

The separate [optical diagnostics notebook](notebooks/optical_showcase.ipynb)
provides spectra, timing distributions and selected trajectories for inspecting
individual interactions. See the [diagnostics guide](docs/optical_showcase.md).

The [WebGPU viewer prototype](docs/webgpu_viewer.md) renders geometry in the
browser, including the 49,684-PMT scene, using shared meshes and WGSL traversal.
Its camera-ray performance is separate from optical photon transport.

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
