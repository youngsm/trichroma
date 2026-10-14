# Merge TriChroma into chroma-lite as `chroma.triton`, rename the repo TriChroma

Date: 2026-10-14. Branch: `merge-trichroma` of chroma-lite.

## Goal

One repository and one distribution provide Chroma's API and both of its
backends. The Triton engine becomes the subpackage `chroma.triton` and the
default backend; PyCUDA stays an optional extra. The repository
`youngsm/chroma-lite` becomes `youngsm/trichroma`, and the standalone TriChroma
repository is archived.

## Why

TriChroma cannot run without Chroma, and the two already change together: the
backend switch lives in chroma-lite (`chroma/backend.py`, `chroma/sim.py`) and
the engine in TriChroma; CUDA Chroma records random-number tapes
(`chroma/sim_cuda.py`) that TriChroma replays; on 2026-09-27 the
`CHROMA_TRITON_TAPE` change and the DAQ CDF fix each landed in both repositories
on the same day. Nothing pins the two to matching revisions, and the bitwise
guarantee (record with CUDA, replay with Triton) can only be tested with both.

## Layout of the merged repository

| From TriChroma | To |
|---|---|
| `src/trichroma/` | `chroma/triton/` |
| `tests/{unit,integration,bitwise}/`, `tests/conftest.py` | `tests/{unit,integration,bitwise}/`, `tests/conftest.py` |
| `benchmarks/` | `benchmarks/` |
| `docs/` (guide, how it scales, performance, correctness, reference, history, img) | `docs/` |
| `requirements-triton.txt` | `requirements-triton.txt` |
| `README.md` | `README.md` (replaces chroma-lite's, see Documentation) |

| From chroma-lite | To |
|---|---|
| `test/` (the original CUDA suite) | `tests/cuda_backend/` (where its `conftest.py` already says it lives; still skipped without PyCUDA) |
| `doc/` (Sphinx docs, Chroma whitepaper) | `docs/chroma/` |
| `README.md` (Chroma's history, containers) | `docs/chroma/README.md` |
| `chroma/`, `bin/`, `installation/`, `images/`, `LICENSE.txt` | unchanged |

The TriChroma history is kept: its commits are rewritten with
`git filter-repo` so the files sit at their new paths, then merged into
chroma-lite with `--allow-unrelated-histories`. `git log chroma/triton/...`
shows the engine's whole history. Rewritten commits get new hashes; the
archived standalone repository keeps the original ones.

## Code changes

- Every `trichroma` import becomes `chroma.triton` (in `chroma.triton` itself,
  `chroma/sim.py`, `chroma/sim_cuda.py`'s tape hook, tests and benchmarks).
  No `trichroma` compatibility shim.
- `chroma/triton/__init__.py` drops the "install chroma-lite first" check.
  Importing `chroma.triton` still imports no torch, Triton or PyCUDA, so
  `import chroma` stays light and works on machines without a GPU.
- Nothing executable lives directly in `chroma/`: a script there would put
  `chroma/` on `sys.path`, where `chroma/triton` could shadow the `triton`
  library.
- `chroma.backend.default_backend()` returns `triton`. `CHROMA_BACKEND=cuda`
  selects CUDA Chroma (needs the `cuda` extra). The environment variables
  `CHROMA_BACKEND`, `CHROMA_TRITON` and `CHROMA_TRITON_TAPE` keep their names
  and values.
- The PyCUDA-only tape modules (`chroma.triton.tape.record_cuda`, `probes`)
  are imported only by the CUDA backend when recording, as now.
- Benchmarks and validation scripts that build chroma-lar detectors keep
  importing `chroma_lar`; they are not packaged and are not dependencies.

## Packaging

`pyproject.toml` (setuptools) replaces `setup.py`.

- Distribution name `trichroma`, version `0.6.0` (after chroma-lite's 0.5 and
  TriChroma's 0.1.0). Import name stays `chroma`.
- Packages: `chroma` and its subpackages including `chroma.triton`; package
  data `chroma/cuda/*.cu`, `chroma/cuda/*.h`; scripts `bin/*`.
- `requires-python = ">=3.9"`.
- Dependencies: `numpy`, `scipy`, `torch`, `triton`.
- Extras: `cuda = ["pycuda", "pytools==2022.1.2"]`;
  `viz = ["pygame", "pyzmq", "uncertainties", "matplotlib"]`; `test = ["pytest"]`.
- pytest: `testpaths = ["tests"]`, marker `gpu`. `python -m pytest` on a CPU
  node passes, with GPU tests and the CUDA suite skipped.
- An environment that has the old `Chroma` and `trichroma` distributions must
  uninstall both before installing this one, because both wrote `chroma/` (the
  README says so).

## Documentation

- `README.md` is TriChroma's new README, changed for the merged project: one
  install (`pip install 'git+https://github.com/youngsm/trichroma.git'`, add
  `[cuda]` for CUDA Chroma), Triton as the default backend, module paths
  `chroma.triton.*`, and a short Chroma lineage section (LaTorre and Seibert's
  Chroma, Ben Land's fork, this fork's wavelength shifters, dichroic filters
  and wire planes) that links to `docs/chroma/`.
- `docs/guide.md`, `docs/reference/` and `docs/correctness.md`: the
  default-backend rule, the install steps and `trichroma.*` module names are
  updated; nothing else is rewritten.
- `docs/history/` stays as written: it records what happened, including old
  module names and commit hashes.

## Repository renames on GitHub

Done last, after the branch is merged to the default branch and the user confirms:

1. Rename `youngsm/trichroma` to `youngsm/trichroma-standalone`, point its
   README to the new home, and archive it.
2. Rename `youngsm/chroma-lite` to `youngsm/trichroma` and update its
   description. GitHub redirects `chroma-lite` URLs (git, pip, API). The repo
   remains a fork of `BenLand100/chroma`.
3. The default branch `master` is renamed `main`. GitHub retargets open pull
   requests and redirects web links, but git references to `@master` (old
   container recipes, pinned pip URLs) stop resolving, and local clones need
   `git branch -m master main`.

Afterwards, local clones of the old TriChroma repository (under
`/sdf/group/neutrino/youngsam/sim/triton_chroma/`) must point their `origin`
at `trichroma-standalone`: the name `trichroma` then belongs to the merged
repository, whose history does not contain the old hashes.

## Testing

1. `python -m pytest` on a CPU node: passes, GPU and CUDA-suite tests skipped.
2. `python -m pytest` on a turing and an ampere node, in a uv environment
   with the merged package: TriChroma's unit, integration and bitwise tests
   pass as they do in the standalone repository at `6da1da3`.
3. With the `cuda` extra on one GPU node: `CHROMA_BACKEND=cuda` runs, and a
   tape recorded by CUDA Chroma replays bitwise with the Triton backend
   (`tests/bitwise`).
4. A fresh uv environment installs the merged checkout
   (`uv pip install /sdf/data/neutrino/youngsam/trichroma-merge`) without
   compiling anything; `import chroma` on a CPU node imports no torch.

## Out of scope

chroma-lar's changes (its own spec, revised to depend on this), the NERSC
Shifter image, rewriting `docs/history/`, and the old Geant4 container recipes
in `installation/`.
