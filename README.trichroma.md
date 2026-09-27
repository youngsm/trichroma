# TriChroma

A Triton backend for [Chroma](https://github.com/youngsm/chroma-lite)'s
optical photon simulation that needs no PyCUDA. Chroma (chroma-lite) keeps the
API and the CUDA backend; with TriChroma installed, `chroma.sim.Simulation`
runs on Triton, so existing code such as
[chroma-lar](https://github.com/youngsm/chroma-lar) runs unchanged.

![LArTPC photons/s versus photons per launch, TriChroma and CUDA Chroma](docs/img/lar_vs_chroma.png)

On chroma-lar's LArTPC it transports up to 176M photons/s on an A100, 15-75x
CUDA Chroma depending on the photons per launch
([performance](docs/performance.md)). An exact mode
(`CHROMA_TRITON_TAPE=replay:<dir>`) reproduces recorded CUDA Chroma runs bit
for bit.

## Install

Linux with a CUDA GPU; the validated environment is Python 3.10, PyTorch
2.5.0+cu124 and Triton 3.1.0 (`requirements-triton.txt` pins it).

```bash
pip install 'git+https://github.com/youngsm/chroma-lite.git'   # Chroma's API ('[cuda]' adds the CUDA backend)
git clone https://github.com/youngsm/trichroma.git
pip install -e trichroma                                        # numpy, scipy, torch, triton
export TRITON_CACHE_DIR=/lscratch/$USER/triton-cache            # compiled kernels (default: ~/.triton)
```

## Use

```python
from chroma.sim import Simulation
from chroma_lar.geometry import build_detector_from_config

detector = build_detector_from_config("detector_config_reflect_reflect3wires")
sim = Simulation(detector, seed=1)
for event in sim.simulate(photons, keep_flat_hits=True, max_steps=1000):
    ...
```

`chroma.sim.Simulation` is TriChroma when PyCUDA is not installed;
`CHROMA_BACKEND=triton` or `cuda` chooses explicitly, so both backends can be
installed side by side. For speed, draw photons on the GPU with
`trichroma.sources`. The first run on a new detector compiles its kernels, and
later runs load them from `TRITON_CACHE_DIR`.

`trichroma.utils` compares two runs: `compare_events` bitwise (exact mode
against the CUDA run it replays, or TriChroma against itself), and
`compare_statistics` within statistical errors (production against CUDA
Chroma).

## Documentation

- [Performance](docs/performance.md): throughput and comparisons with CUDA
  Chroma.
- [Design](docs/design.md): environment variables, the API contract, the
  engine, the repository layout.
- [Bitwise mode](docs/bitwise_mode.md): recording and replaying CUDA Chroma
  runs.
- [Exact vs production](docs/exact_vs_production.md): every difference
  between the two modes, and the evidence that production is unbiased.

Run the tests with `python -m pytest`; GPU tests skip without a GPU.

Chroma originated with Anthony LaTorre and Stanley Seibert; its components
keep their [original license](LICENSE.txt). Earlier work and provenance:
[docs/earlier_triton_work.md](docs/earlier_triton_work.md).
