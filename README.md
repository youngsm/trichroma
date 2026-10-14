# TriChroma

TriChroma runs [Chroma](docs/chroma/README.md)'s optical photon simulation on
[Triton](https://triton-lang.org) instead of PyCUDA. Your Chroma code stays the
same: `chroma.sim.Simulation` simply runs on TriChroma, 15 to 75 times faster.

![Photons per second against photons per launch, TriChroma and CUDA Chroma, on a liquid-argon TPC](docs/img/lar_vs_chroma.png)

## Install

You need Linux and an NVIDIA GPU. One package provides Chroma and its Triton
engine:

```bash
pip install 'git+https://github.com/youngsm/trichroma.git'
# or, with uv
uv pip install 'git+https://github.com/youngsm/trichroma.git'
```

For the original PyCUDA backend add the `cuda` extra,
`pip install 'trichroma[cuda] @ git+https://github.com/youngsm/trichroma.git'`;
it compiles PyCUDA and needs the CUDA toolkit. The tested versions are pinned in
`requirements-triton.txt` (Python 3.10, PyTorch 2.5, Triton 3.1).

If an environment has the older `Chroma` (chroma-lite) or `trichroma`
packages, uninstall both first: they wrote the same `chroma/` directory.

## Use

Nothing changes in your scripts:

```python
from chroma.sim import Simulation

sim = Simulation(detector, seed=1)       # any Chroma Detector
for event in sim.simulate(photons, keep_flat_hits=True):
    ...
```

`Simulation` runs on Triton by default. `CHROMA_BACKEND=cuda` selects the
original CUDA implementation (with the `cuda` extra installed).

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

## Chroma

TriChroma is a fork of Chroma by Anthony LaTorre and Stanley Seibert, by way
of Ben Land's version, extended with wavelength shifters, dichroic filters and
analytic wire planes. Chroma's own README and documentation, including the
whitepaper, are in [docs/chroma](docs/chroma/README.md).

Chroma's components keep their [original license](LICENSE.txt).
