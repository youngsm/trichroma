# TriChroma

A Triton implementation of [Chroma](https://github.com/youngsm/chroma-lite)'s
optical photon simulation that needs no PyCUDA. It installs two packages:

- `chroma`: Chroma's API (geometry, detector, event, loader, ...) at the
  usual import paths, so existing code such as
  [chroma-lar](https://github.com/youngsm/chroma-lar) runs unchanged;
- `trichroma`: the Triton implementation behind `chroma.sim.Simulation`.

On an A100 (40 GB) with LAr detectors built by chroma-lar, against CUDA
Chroma on the same GPU:

| Workload | CUDA Chroma | TriChroma |
|---|---|---|
| 30M prepared photons, one weighted `simulate()` call (LUT style) | 0.55M photons/s | 11.7M photons/s (23M with opt-in roulette) |
| waveform-map macro, photons drawn on the GPU | 0.29 s per voxel (numpy photons) | 3.1 ms per voxel |
| pixel-detector quantile LUT, 30M photons per voxel, GPU photons | ~42 s per voxel (numpy photons) | 0.22 s per voxel |

CUDA Chroma's rates include its wire defects, which end many histories early
and lose 15% of the light on these detectors (see the design notes).
Scripts that generate photons with numpy on the CPU gain only 2-5x until the
photons are drawn on the GPU (`trichroma.sources`); the macro numbers use
the chroma-lar branch
[`trichroma-fast-waveform`](https://github.com/youngsm/chroma-lar/tree/trichroma-fast-waveform).

An exact mode (`CHROMA_TRITON_TAPE=replay:<dir>`) replays random-number tapes
recorded from CUDA Chroma and reproduces its outputs bit for bit, bugs
included.

## Install

Linux with a CUDA GPU; the validated environment is Python 3.10, PyTorch
2.5.0+cu124 and Triton 3.1.0 (`requirements-triton.txt` pins it).

```bash
git clone https://github.com/youngsm/trichroma.git
cd trichroma
python -m pip install -e .            # numpy, scipy, torch, triton
python -m pip install -e '.[cuda]'    # optional: the original CUDA backend and tape recording
export TRITON_CACHE_DIR=/lscratch/$USER/triton-cache   # compiled kernels (default: ~/.triton)
```

## Use

`chroma.sim.Simulation` is the Triton implementation when PyCUDA is not
installed; `CHROMA_BACKEND=triton` or `cuda` chooses explicitly.

```python
from chroma.sim import Simulation
from chroma_lar.geometry import build_detector_from_config

detector = build_detector_from_config("detector_config_reflect_reflect3wires")
sim = Simulation(detector, seed=1)
for event in sim.simulate(photons, keep_flat_hits=True, max_steps=1000):
    ...
```

Photons can also be drawn on the GPU and never visit host memory:

```python
from chroma.event import Event
from trichroma import sources

photons = sources.photon_bombs(200_000, voxel_centres, voxel_size=30, wavelength=450)
events = (Event(photons_beg=photons[i * 200_000:(i + 1) * 200_000]) for i in range(len(voxel_centres)))
for event in sim.simulate(events, keep_flat_hits=True, photons_per_batch=5_000_000):
    ...
```

The first run on a new detector compiles its kernels (tens of seconds, up to
a minute or two), then they come from `TRITON_CACHE_DIR`.

## Documentation

- [Design](docs/design.md): environment variables, the API contract, every
  deliberate difference from CUDA Chroma, the engine and its throughput.
- [Bitwise mode](docs/bitwise_mode.md): recording and replaying CUDA Chroma's
  random numbers.
- [Earlier Triton work](docs/earlier_triton_work.md), before the drop-in
  backend.
- [The original Chroma README](docs/chroma/README.md), documentation and
  container recipes.

## Tests and benchmarks

```bash
python -m pytest                      # unit, integration and bitwise tests; GPU tests skip without a GPU
python -m trichroma.tape.verify all --fixture synthetic \
    --cuda-python /path/to/pycuda/env/bin/python \
    --triton-python "$(which python)" --work /lscratch/$USER/tape-check
```

The second records fixtures with CUDA Chroma and checks that the Triton
replay matches every output word (the LAr fixtures also need chroma-lar).
The original Chroma test suite is `tests/cuda_backend` (run it by name, with
PyCUDA; many of its tests failed before TriChroma).
`benchmarks/` has the throughput benchmark behind the table above
(`simulate_throughput.py`, either backend), an engine profiler and a
statistical CUDA/Triton comparison; they build chroma-lar detectors.

## Layout

| Location | Contents |
|---|---|
| `src/chroma/` | Chroma's API, and the original CUDA backend (`sim_cuda.py`, `gpu/`, `cuda/`, `camera.py`; needs `[cuda]`) |
| `src/trichroma/simulation.py` | The Triton `Simulation`: batching, events, hits, DAQ, pipelining |
| `src/trichroma/engine/` | Scene compiler, SAH trees, fused transport kernel, physics, exact mode |
| `src/trichroma/sources.py` | Photon sources drawn on the GPU |
| `src/trichroma/tape/` | Bitwise mode: tape format, CUDA-side recorder, replay checks, fixtures |
| `tests/` | `unit/`, `integration/`, `bitwise/` (recorded tapes), `cuda_backend/` (original Chroma suite) |
| `benchmarks/` | Throughput, profiling and backend comparison scripts |
| `docs/` | Design notes and validation records |
| `bin/` | The original Chroma command-line tools |

## Provenance

Modules from the earlier detector-specific Triton work are at commit
[`c951290`](https://github.com/youngsm/trichroma/tree/c951290/src/trichroma/chroma/triton);
the chroma-lar copy this repository once carried is at
[`78e4b1a`](https://github.com/youngsm/trichroma/tree/78e4b1ab4b64bf5be9f78d848c8490170dae8d68/chroma-lar).
[The consolidation manifest](docs/consolidation_manifest.json) records the
upstream revisions the repository started from.

Chroma originated with Anthony LaTorre and Stanley Seibert. The Chroma
components retain their [original license](LICENSE.txt). This repository
includes the local work based on
[youngsm/chroma-lite](https://github.com/youngsm/chroma-lite).
