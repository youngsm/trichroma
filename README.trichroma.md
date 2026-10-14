# TriChroma

TriChroma runs [Chroma](https://github.com/youngsm/chroma-lite)'s optical
photon simulation on [Triton](https://triton-lang.org) instead of PyCUDA. Your
Chroma code stays the same: `chroma.sim.Simulation` simply runs on TriChroma,
15 to 75 times faster.

![Photons per second against photons per launch, TriChroma and CUDA Chroma, on a liquid-argon TPC](docs/img/lar_vs_chroma.png)

## Install

You need Linux and an NVIDIA GPU. Install Chroma (the chroma-lite
distribution) first, then TriChroma:

```bash
pip install 'git+https://github.com/youngsm/chroma-lite.git'
git clone https://github.com/youngsm/trichroma.git
pip install -e trichroma
```

The tested versions are pinned in `requirements-triton.txt` (Python 3.10,
PyTorch 2.5, Triton 3.1).

## Use

Nothing changes in your scripts:

```python
from chroma.sim import Simulation

sim = Simulation(detector, seed=1)       # any Chroma Detector
for event in sim.simulate(photons, keep_flat_hits=True):
    ...
```

When PyCUDA is not installed, `Simulation` is TriChroma. With both installed,
choose with `CHROMA_BACKEND=triton` or `CHROMA_BACKEND=cuda`.

The first run on a new detector compiles GPU kernels, which takes a minute or
so; later runs reuse them.

## Documentation

- [User guide](docs/guide.md): switching backends, what changes in your
  results, options, getting full speed.
- [How TriChroma scales](docs/how_it_scales.md): the ideas behind the speed,
  without code.
- [Performance](docs/performance.md): measured throughput on three detectors.
- [Correctness](docs/correctness.md): how TriChroma's results are checked
  against CUDA Chroma.
- [Reference](docs/reference/): every difference from CUDA Chroma, the
  bit-exact replay mode, and the engine's internals.

Run the tests with `python -m pytest`; tests that need a GPU are skipped
without one.

Chroma originated with Anthony LaTorre and Stanley Seibert; its components
keep their [original license](LICENSE.txt).
