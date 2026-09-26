# TriChroma

A Triton implementation of [Chroma](https://github.com/youngsm/chroma-lite)'s
optical photon simulation that needs no PyCUDA. `src/trichroma` holds a `chroma`
package: Chroma's geometry, detector, event and loader APIs, with a Triton
engine behind `chroma.sim.Simulation` when `CHROMA_BACKEND=triton` is set.
Existing Chroma code such as [chroma-lar](https://github.com/youngsm/chroma-lar)
runs on it without changes.

On an A100 (40 GB), for LAr detectors built by chroma-lar:

- the chroma-lar waveform map (`pyrat macros/waveform_map_pyrat.py`, 720
  voxels x 200K photons) takes 8 s end to end instead of 55 s;
- the pixel-detector quantile LUT
  (`macros/waveform_map_pyrat_quantile.py`, 30M photons per voxel) takes
  0.22 s per voxel, against the ~18 s estimated for CUDA Chroma;
- the transport engine alone reaches 90-140M photons/s on large batches.

The macro speedups also need the chroma-lar branch
[`trichroma-fast-waveform`](https://github.com/youngsm/chroma-lar/tree/trichroma-fast-waveform),
which draws photons on the GPU.

An exact mode (`CHROMA_TRITON_TAPE=replay:<dir>`) replays random-number tapes
recorded from CUDA Chroma and reproduces its outputs bit for bit, bugs
included.

## Quick start

Linux with a CUDA GPU; the recorded runs used Python 3.10, PyTorch
2.5.0+cu124 and Triton 3.1.0.

```bash
git clone https://github.com/youngsm/trichroma.git
cd trichroma
python -m pip install -r requirements-triton.txt
export PYTHONPATH="$PWD/src/trichroma:$PYTHONPATH"
export CHROMA_BACKEND=triton
export TRITON_CACHE_DIR=/lscratch/$USER/triton-cache   # compiled kernels (default: ~/.triton)
```

Then use `chroma` as before, e.g. with chroma-lar on the path:

```python
from chroma_lar.geometry import build_detector_from_config
from chroma.sim import Simulation

detector = build_detector_from_config("detector_config_reflect_reflect3wires")
sim = Simulation(detector, seed=1)
for event in sim.simulate(photons, keep_flat_hits=True, max_steps=1000):
    ...
```

The first run on a new detector compiles the kernels (tens of seconds); they
are cached in `TRITON_CACHE_DIR`. `src/trichroma/setup.py` is Chroma's
original one (it still lists PyCUDA); use the path setup above.

## Documentation

- [Drop-in design](docs/triton_dropin_design.md): environment variables, the
  API contract, every deliberate difference from CUDA Chroma, the engine and
  its throughput.
- [Bitwise tape mode](docs/triton_legacy_tape.md): recording and replaying
  CUDA Chroma's random numbers.
- [Earlier Triton work](src/trichroma/chroma/triton/README.md) (before the
  drop-in backend).

## Validation

```bash
python -m chroma.triton.legacy.verify all --fixture synthetic \
    --cuda-python /path/to/cuda/env/bin/python \
    --triton-python "$(which python)" --work /lscratch/$USER/tape-check
python -m pytest src/trichroma/test/test_legacy_tape.py src/trichroma/test/test_legacy_exact.py
python scripts/check_optics.py --output-dir /tmp/trichroma-checks
```

The first records fixtures with CUDA Chroma (PyCUDA environment) and checks
that the Triton replay matches every output word; the LAr fixtures need
chroma-lar on `PYTHONPATH`. `scripts/profile_engine.py` and
`scripts/compare_backends.py` (statistical CUDA/Triton comparison) also build
chroma-lar detectors.

## Code organization

| Location | Contents |
|---|---|
| `src/trichroma/chroma/` | Chroma's API (geometry, detector, event, loader, ...) and the original CUDA backend (`sim_cuda.py`, `cuda/`, `gpu/`) |
| `src/trichroma/chroma/triton/compat/` | The Triton `Simulation` (batching, events, hits, pipelining) |
| `src/trichroma/chroma/triton/engine/` | Production engine: scene compiler, SAH trees, fused transport kernel, physics, DAQ, exact mode |
| `src/trichroma/chroma/triton/legacy/` | Tape recording (CUDA side), replay and bitwise verification |
| `src/trichroma/test/` | Tests and recorded tape fixtures |
| `scripts/` | Engine profiling, backend comparison and the optics regression runner |
| `docs/` | Design notes and validation records |

## Provenance

The chroma-lar copy that earlier versions of this repository carried,
including its benchmarks, reports and the detector-specific Triton pipeline,
is at commit
[`78e4b1a`](https://github.com/youngsm/trichroma/tree/78e4b1ab4b64bf5be9f78d848c8490170dae8d68/chroma-lar);
chroma-lar itself lives at [youngsm/chroma-lar](https://github.com/youngsm/chroma-lar).
[The consolidation manifest](docs/consolidation_manifest.json) records the
upstream revisions the repository started from.

Chroma originated with Anthony LaTorre and Stanley Seibert. The Chroma
components retain their [original license](src/trichroma/LICENSE.txt). This
repository includes the local work based on
[youngsm/chroma-lite](https://github.com/youngsm/chroma-lite).
