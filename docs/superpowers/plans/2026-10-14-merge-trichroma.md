# Merge TriChroma into chroma-lite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One repository and one distribution (`trichroma`, import name `chroma`) that provides Chroma's API with the Triton engine as `chroma.triton` and the default backend.

**Architecture:** TriChroma's history is rewritten with `git filter-repo` so its files land at their merged paths, then merged into chroma-lite's `merge-trichroma` branch with `--allow-unrelated-histories`. A scripted rewrite turns `trichroma` module paths into `chroma.triton`; a few package-name-sensitive spots are fixed by hand. `pyproject.toml` replaces `setup.py`; the backend default flips to `triton`.

**Tech Stack:** Python 3.10/3.11, setuptools, uv, pytest, PyTorch 2.5.1 + Triton 3.1.0, PyCUDA (optional), git-filter-repo, apptainer, Slurm on S3DF.

**Spec:** `/sdf/data/neutrino/youngsam/trichroma-merge/docs/superpowers/specs/2026-10-14-merge-trichroma-design.md`

## Global Constraints

- Work only in `/sdf/data/neutrino/youngsam/trichroma-merge` (chroma-lite clone, branch `merge-trichroma`). Builds, environments and caches live under `/sdf/data/neutrino/youngsam/`, never `/sdf/group/`.
- Commits are authored by the user's git identity with no Claude attribution or `Co-Authored-By` lines.
- Import name stays `chroma`; the engine is `chroma.triton`; no `trichroma` compatibility shim.
- Distribution name `trichroma`, version `0.6.0`, `requires-python = ">=3.9"`.
- Core dependencies: `numpy`, `scipy`, `torch`, `triton`. Extras: `cuda = ["pycuda", "pytools==2022.1.2"]`, `viz = ["pygame", "pyzmq", "uncertainties", "matplotlib"]`, `test = ["pytest"]`.
- `chroma.backend.default_backend()` returns `"triton"`; `CHROMA_BACKEND=cuda` selects CUDA Chroma. `CHROMA_BACKEND`, `CHROMA_TRITON`, `CHROMA_TRITON_TAPE` keep their names and values.
- `import chroma` imports no `torch`, `triton` or `pycuda`.
- Nothing executable lives directly in `chroma/` (a script there would put `chroma/` on `sys.path` and `chroma/triton` would shadow the `triton` library).
- `docs/history/` is not rewritten.
- Environment variables for every shell in this plan:
  ```bash
  export UV_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/uv
  export TRITON_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/triton
  export APPTAINER_CACHEDIR=/sdf/data/neutrino/youngsam/.apptainer_cache
  export APPTAINER_TMPDIR=/sdf/data/neutrino/youngsam/tmp/apptainer
  ```
- GPU jobs: `srun --account=neutrino:cider-nu --partition=ampere|turing --gpus=1 --cpus-per-gpu=4 --mem=32G --time=2:00:00`.
- Pushing the feature branch `merge-trichroma` is pre-approved. Merging into the default branch, renaming `master` to `main`, and the GitHub repository renames happen only after the user says yes at that step.

## Review Focus

- A user who sets `CHROMA_BACKEND=cuda` without the `cuda` extra should get a clear ImportError naming the extra, not a PyCUDA traceback. Test: Task 5 Step 1 `test_cuda_backend_without_pycuda_says_which_extra`.
- An environment that still has the old `Chroma` or `trichroma` distribution installed should not silently import stale code. Test: Task 6 Step 1 `test_chroma_and_engine_come_from_the_same_tree`.
- Running `python -m chroma.triton.tape.verify ...` (the documented command) should work after the rename. Test: Task 4 Step 1 `test_verify_module_runs_as_main`.
- `import chroma` on a CPU login node must not import torch or fail without a GPU. Test: Task 6 Step 1 `test_import_chroma_imports_no_torch`.
- Docs must not tell readers to install chroma-lite first or to `pip install -e trichroma`. Test: Task 7 Step 1 `test_docs_have_no_two_package_install`.

---

### Task 1: Rewrite TriChroma's history onto the merged paths

**Files:**
- Create: `/sdf/data/neutrino/youngsam/trichroma-filtered/` (a fresh clone; not part of the repo)

**Interfaces:**
- Produces: a git repository at `/sdf/data/neutrino/youngsam/trichroma-filtered`, branch `main`, whose tree has `chroma/triton/`, `tests/`, `benchmarks/`, `docs/`, `requirements-triton.txt`, `LICENSE.txt`, `README.trichroma.md`, `.gitignore.trichroma`, `pyproject.trichroma.toml`, and no `src/`.

- [ ] **Step 1: Install git-filter-repo**

```bash
uv tool install git-filter-repo
git filter-repo --version
```
Expected: a version string (for example `2.47.0`).

- [ ] **Step 2: Fresh clone of TriChroma**

```bash
git clone https://github.com/youngsm/trichroma.git /sdf/data/neutrino/youngsam/trichroma-filtered
git -C /sdf/data/neutrino/youngsam/trichroma-filtered log --oneline -1
```
Expected: `6da1da3 Merge pull request #1 from youngsm/docs-rewrite` (or newer; record the hash for the merge message).

- [ ] **Step 3: Rewrite paths**

```bash
git -C /sdf/data/neutrino/youngsam/trichroma-filtered filter-repo --force \
  --path-rename src/trichroma/:chroma/triton/ \
  --path-rename README.md:README.trichroma.md \
  --path-rename .gitignore:.gitignore.trichroma \
  --path-rename pyproject.toml:pyproject.trichroma.toml
```

- [ ] **Step 4: Verify the layout**

```bash
cd /sdf/data/neutrino/youngsam/trichroma-filtered
git ls-files | cut -d/ -f1-2 | sort -u
test ! -e src && echo "no src/"
git log --oneline -- chroma/triton/simulation.py | wc -l
```
Expected: entries start with `chroma/triton`, `tests/`, `benchmarks/`, `docs/`, plus the renamed top-level files; `no src/`; a commit count above 1 (history kept).

### Task 2: Merge the rewritten history into `merge-trichroma`

**Files:**
- Modify: repository history of `/sdf/data/neutrino/youngsam/trichroma-merge` (branch `merge-trichroma`)

**Interfaces:**
- Consumes: Task 1's repository.
- Produces: branch `merge-trichroma` containing both trees; `chroma/triton/` present next to the rest of `chroma/`.

- [ ] **Step 1: Fetch and merge**

```bash
cd /sdf/data/neutrino/youngsam/trichroma-merge
git switch merge-trichroma
git remote add trichroma-filtered /sdf/data/neutrino/youngsam/trichroma-filtered
git fetch trichroma-filtered
git merge --allow-unrelated-histories --no-ff trichroma-filtered/main \
  -m "Merge TriChroma (youngsm/trichroma 6da1da3) as chroma.triton" \
  -m "TriChroma's history is rewritten so its files sit at their merged paths: src/trichroma -> chroma/triton; tests, benchmarks and docs at the top level."
```
Expected: the merge completes without conflicts (`LICENSE.txt` is byte-identical in both; the other colliding files were renamed in Task 1). If git reports a conflict, stop and report it.

- [ ] **Step 2: Verify**

```bash
git ls-files chroma/triton | head -5
git log --oneline -- chroma/triton/engine/core.py | tail -2
git remote remove trichroma-filtered
```
Expected: `chroma/triton/__init__.py` etc. listed; the oldest commits of `core.py` are TriChroma's.

### Task 3: Arrange the merged tree

**Files:**
- Move: `test/` → `tests/cuda_backend/`
- Move: `doc/` → `docs/chroma/`
- Move: `README.md` → `docs/chroma/README.md`
- Move: `README.trichroma.md` → `README.md`
- Modify: `.gitignore` (union of both; delete `.gitignore.trichroma`)
- Modify: `tests/cuda_backend/conftest.py` (docstring path)

**Interfaces:**
- Produces: the layout in the spec's table. Later tasks edit `README.md` (TriChroma's text) and `docs/`.

- [ ] **Step 1: Move files**

```bash
cd /sdf/data/neutrino/youngsam/trichroma-merge
mkdir -p tests docs/chroma
git mv test tests/cuda_backend
git mv doc docs/chroma/doc
git mv README.md docs/chroma/README.md
git mv README.trichroma.md README.md
```

`doc/` moves to `docs/chroma/doc/` (the Sphinx sources keep their own `source/` layout; TriChroma's `.gitignore.trichroma` already ignores `docs/chroma/doc/build/`).

- [ ] **Step 2: Merge `.gitignore`**

Replace `.gitignore` with the union of both files, keeping each pattern once:

```gitignore
__pycache__/
*.py[cod]
*~
*.so
*.d
*.root
*.lprof
*.simg
.coverage
.DS_Store
.env
.vscode/
.venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.egg-info/
build/
dist/
distribute*.gz
distribute*.egg
docs/chroma/doc/build/
node_modules
package-lock.json
```

```bash
git rm -q .gitignore.trichroma
```

- [ ] **Step 3: Fix the CUDA suite's docstring path**

In `tests/cuda_backend/conftest.py`, the docstring already says `python -m pytest tests/cuda_backend`; change only the install hint `(``pip install .[cuda]``)` to `(``pip install '.[cuda]'``)` so the shell does not glob it. Check:

```bash
grep -n "tests/cuda_backend\|\[cuda\]" tests/cuda_backend/conftest.py
```
Expected: both lines present.

- [ ] **Step 4: Fix links broken by the moves**

`docs/chroma/README.md` now sits next to `docs/chroma/doc/`, so its `doc/source/chroma.pdf` link still resolves. Its image links point at the top-level `images/` folder: change every `](images/` to `](../../images/`. Then check every relative link:

```bash
cd docs/chroma && for f in $(grep -o '](\([^)#]*\))' README.md | sed 's/](\(.*\))/\1/' | grep -v '^http'); do test -e "$f" || echo "BROKEN $f"; done; cd ../..
```
Expected: no `BROKEN` lines.

- [ ] **Step 5: Commit**

```bash
git add -A .gitignore tests docs README.md
git commit -q -m "Arrange the merged tree: tests/, docs/, TriChroma's README" -m "chroma-lite's CUDA test suite moves to tests/cuda_backend (where its conftest says it lives), its Sphinx docs and README to docs/chroma/; TriChroma's README becomes the repository's."
```

### Task 4: Rename `trichroma` modules to `chroma.triton`

**Files:**
- Create (outside the repo): `/sdf/data/neutrino/youngsam/trichroma-merge-tools/rename_modules.py`
- Modify: every `.py` under `chroma/triton/`, `tests/{unit,integration,bitwise}/`, `tests/conftest.py`, `benchmarks/`
- Modify: `chroma/sim.py`, `chroma/sim_cuda.py`, `chroma/backend.py` (docstrings and imports)
- Modify: `chroma/triton/__init__.py`, `chroma/triton/tape/record_cuda.py:125-140`
- Test: `tests/unit/test_layout.py`

**Interfaces:**
- Produces: modules `chroma.triton.simulation` (`Simulation`, `pick_seed`), `chroma.triton.sources`, `chroma.triton.utils` (`compare_events`, `compare_statistics`, `two_proportion_z`, `ks_test`, `event_statistics_arrays`), `chroma.triton.tape.verify` (CLI `python -m chroma.triton.tape.verify`).

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_layout.py`:

```python
"""The engine lives at chroma.triton; nothing imports the old trichroma package."""

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_no_trichroma_imports_left():
    pattern = re.compile(r"^\s*(from|import)\s+trichroma\b", re.M)
    offenders = []
    for base in ("chroma", "tests", "benchmarks"):
        for path in (ROOT / base).rglob("*.py"):
            if pattern.search(path.read_text()):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_engine_modules_import():
    import chroma.triton.options
    import chroma.triton.sources
    import chroma.triton.utils

    assert chroma.triton.utils.two_proportion_z(10, 100, 10, 100) == 0.0


def test_verify_module_runs_as_main():
    env = dict(os.environ, CHROMA_BACKEND="triton")
    result = subprocess.run([sys.executable, "-m", "chroma.triton.tape.verify", "--help"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "python -m chroma.triton.tape.verify" in result.stdout
```

- [ ] **Step 2: Run them to see them fail**

Use the CPU environment (created here, reused later):

```bash
cd /sdf/data/neutrino/youngsam/trichroma-merge
uv venv --python 3.10 /sdf/data/neutrino/youngsam/envs/trichroma-merge
source /sdf/data/neutrino/youngsam/envs/trichroma-merge/bin/activate
uv pip install -e . numpy scipy torch==2.5.1 pytest
python -m pytest tests/unit/test_layout.py -q
```
Expected: FAIL (`trichroma` imports present; `chroma.triton.tape.verify --help` prints `python -m trichroma...` or fails to import).

(`setup.py` still drives this editable install; `chroma.triton` is found because `find_packages()` picks up `chroma/triton`. Task 6 replaces it.)

- [ ] **Step 3: Write the rename script**

`/sdf/data/neutrino/youngsam/trichroma-merge-tools/rename_modules.py`:

```python
"""Rewrite trichroma module paths to chroma.triton in the merged tree.

Usage: python rename_modules.py <repo root>
"""

import pathlib
import re
import sys

RULES = [
    (re.compile(r"src/trichroma/"), "chroma/triton/"),
    (re.compile(r"\bfrom trichroma import\b"), "from chroma.triton import"),
    # trichroma.<module>, but not URLs such as github.com/youngsm/trichroma.git
    (re.compile(r"(?<![/\w-])trichroma\.(?!git\b)"), "chroma.triton."),
]
TARGETS = ["chroma/triton", "tests/unit", "tests/integration", "tests/bitwise", "tests/conftest.py",
           "benchmarks", "chroma/sim.py", "chroma/sim_cuda.py", "chroma/backend.py"]


def rewrite(path):
    text = path.read_text()
    new = text
    for pattern, replacement in RULES:
        new = pattern.sub(replacement, new)
    if new != text:
        path.write_text(new)
        print("rewrote", path)


def main(root):
    root = pathlib.Path(root)
    for target in TARGETS:
        path = root / target
        files = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for f in files:
            rewrite(f)


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 4: Run it and fix the remaining spots by hand**

```bash
python /sdf/data/neutrino/youngsam/trichroma-merge-tools/rename_modules.py /sdf/data/neutrino/youngsam/trichroma-merge
grep -rn "trichroma" chroma tests/unit tests/integration tests/bitwise tests/conftest.py benchmarks --include=*.py
```

Fix every remaining hit:

1. `chroma/triton/__init__.py`: replace the whole file with

```python
"""The Triton backend of Chroma's optical photon simulation (TriChroma).

With the Triton backend (the default, or ``CHROMA_BACKEND=triton``),
``chroma.sim.Simulation`` is :class:`chroma.triton.simulation.Simulation`.

* :mod:`chroma.triton.simulation` -- the ``Simulation`` class (batching, events,
  hits, DAQ, pipelining).
* :mod:`chroma.triton.engine` -- the transport engine: scene compiler, trees,
  fused transport kernel, physics, exact mode.
* :mod:`chroma.triton.sources` -- photon sources drawn on the GPU.
* :mod:`chroma.triton.tape` -- bitwise mode: record CUDA Chroma's random numbers
  and replay them exactly.

Importing this package does not import torch, Triton or PyCUDA.
"""
```

2. `chroma/triton/tape/record_cuda.py`, `_source_hashes` (the engine is now inside `chroma`):

```python
def _source_hashes():
    import chroma

    root = os.path.dirname(os.path.abspath(chroma.__file__))
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "_build_ext")]
        for name in sorted(filenames):
            if name.endswith((".py", ".h", ".cu")):
                path = os.path.join(dirpath, name)
                with open(path, "rb") as f:
                    out[os.path.relpath(path, os.path.dirname(root))] = hashlib.sha256(f.read()).hexdigest()
    return root, out
```

3. `chroma/sim.py`: replace the module with

```python
"""``chroma.sim.Simulation`` resolves to the backend selected by ``CHROMA_BACKEND``
(unset: ``triton``).

The original PyCUDA implementation lives unchanged in :mod:`chroma.sim_cuda`
(``pip install 'trichroma[cuda]'``); the Triton implementation lives in
:mod:`chroma.triton.simulation` and does not import PyCUDA. See
:mod:`chroma.backend`.
"""

from chroma.backend import backend_name as _backend_name

if _backend_name() == "triton":
    from chroma.triton.simulation import Simulation, pick_seed  # noqa: F401
else:
    try:
        from chroma.sim_cuda import Simulation, pick_seed  # noqa: F401
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".")[0] not in ("pycuda", "pytools"):
            raise
        raise ImportError("CHROMA_BACKEND=cuda needs PyCUDA: pip install 'trichroma[cuda]'") from error

__all__ = ["Simulation", "pick_seed"]
```

4. Leftovers that are prose about the project (`TriChroma`), URLs (`github.com/youngsm/trichroma`), or test function names (`test_trichroma_sources_simulate`) stay. `trichroma` inside `chroma/triton/tape/fixtures.py`'s docstring reference `(trichroma commit 78e4b1a)` stays (history).

Re-run the grep; every remaining hit must be one of the kinds in item 4.

Then read the prose the regex touched, looking for a sentence-ending `trichroma.` that became `chroma.triton.` (a lost full stop) or a distribution name turned into a module path:

```bash
git diff --word-diff=plain -- chroma tests benchmarks | grep -n "{+chroma.triton" | grep -v "import\|:mod:\|python -m" | head -40
```
Fix any such sentence by hand.

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/unit/test_layout.py -q
python -m pytest tests/unit -q
```
Expected: `test_layout.py` passes. The rest of `tests/unit` passes except `test_backend.py::test_default_backend_follows_pycuda` (fixed in Task 5); note any other failure and fix it before committing.

- [ ] **Step 6: Commit**

```bash
git add -A chroma tests benchmarks
git commit -q -m "Rename the trichroma package to chroma.triton" -m "Module paths, the tape recorder's source hashes (the engine is inside chroma now) and chroma.sim's import; prose, URLs and history keep the TriChroma name."
```

### Task 5: Triton is the default backend

**Files:**
- Modify: `chroma/backend.py:1-45`
- Modify: `tests/unit/test_backend.py`

**Interfaces:**
- Produces: `chroma.backend.default_backend() -> "triton"`; `backend_name(environ=None) -> "cuda" | "triton"` unchanged otherwise.

- [ ] **Step 1: Update the tests first**

In `tests/unit/test_backend.py`, replace `test_default_backend_follows_pycuda` with:

```python
def test_default_backend_is_triton_even_with_pycuda(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args):
        return object() if name == "pycuda" else real_find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    default_backend.cache_clear()
    try:
        assert default_backend() == "triton"
        assert backend_name({}) == "triton"
        assert backend_name({"CHROMA_BACKEND": ""}) == "triton"
    finally:
        default_backend.cache_clear()


def test_cuda_backend_without_pycuda_says_which_extra():
    import importlib.util
    import os
    import subprocess
    import sys

    if importlib.util.find_spec("pycuda") is not None:
        pytest.skip("PyCUDA is installed")
    env = dict(os.environ, CHROMA_BACKEND="cuda")
    result = subprocess.run([sys.executable, "-c", "from chroma.sim import Simulation"],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "trichroma[cuda]" in result.stderr
```

Keep `import importlib.util` at the top of the file (the first test uses it).

- [ ] **Step 2: Run to see the first fail**

```bash
python -m pytest tests/unit/test_backend.py -q
```
Expected: `test_default_backend_is_triton_even_with_pycuda` FAILS (`assert 'cuda' == 'triton'`); `test_cuda_backend_without_pycuda_says_which_extra` passes (Task 4's `chroma/sim.py`).

- [ ] **Step 3: Change the default**

In `chroma/backend.py`, change the module docstring's sentence
`Unset, it is ``cuda`` when PyCUDA is installed and ``triton`` otherwise.`
to
`Unset, it is ``triton``; ``cuda`` needs the ``cuda`` extra (PyCUDA).`
and the `triton` bullet's `:mod:`trichroma.simulation`` (already rewritten to `chroma.triton.simulation` by Task 4). Replace `default_backend`:

```python
@functools.lru_cache(maxsize=None)
def default_backend():
    """The backend used when ``CHROMA_BACKEND`` is unset: ``triton``."""
    return "triton"
```

Remove `import importlib.util` from `chroma/backend.py` if it is now unused.

- [ ] **Step 4: Run the tests**

```bash
python -m pytest tests/unit -q
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add chroma/backend.py tests/unit/test_backend.py
git commit -q -m "Make Triton the default backend" -m "CHROMA_BACKEND=cuda now opts into CUDA Chroma, with an ImportError naming the cuda extra when PyCUDA is missing."
```

### Task 6: Package as `trichroma` with pyproject.toml

**Files:**
- Create: `pyproject.toml`
- Delete: `setup.py`, `pyproject.trichroma.toml`
- Test: `tests/unit/test_packaging.py`

**Interfaces:**
- Produces: installable distribution `trichroma==0.6.0` with extras `cuda`, `viz`, `test`; scripts `chroma-sim`, `chroma-cam`, `chroma-geo`, `chroma-bvh`, `chroma-server`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_packaging.py`:

```python
"""The trichroma distribution provides chroma and chroma.triton from one tree."""

import os
import subprocess
import sys
from importlib import metadata


def test_distribution_is_trichroma():
    dist = metadata.distribution("trichroma")
    assert dist.version == "0.6.0"
    requires = " ".join(dist.requires or [])
    for name in ("numpy", "scipy", "torch", "triton"):
        assert name in requires
    assert 'extra == "cuda"' in requires


def test_chroma_and_engine_come_from_the_same_tree():
    import chroma
    import chroma.triton

    chroma_dir = os.path.dirname(os.path.abspath(chroma.__file__))
    assert os.path.dirname(os.path.dirname(os.path.abspath(chroma.triton.__file__))) == chroma_dir
    for old in ("Chroma",):
        try:
            metadata.distribution(old)
        except metadata.PackageNotFoundError:
            continue
        raise AssertionError("the old %r distribution is still installed: uninstall it" % old)


def test_import_chroma_imports_no_torch():
    code = ("import sys, chroma\n"
            "heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'triton', 'pycuda'))\n"
            "assert not heavy, heavy\n")
    env = {k: v for k, v in os.environ.items() if k != "CHROMA_BACKEND"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_cuda_kernels_are_package_data():
    import chroma

    cuda_dir = os.path.join(os.path.dirname(chroma.__file__), "cuda")
    assert any(name.endswith(".cu") for name in os.listdir(cuda_dir))
```

- [ ] **Step 2: Run to see them fail**

```bash
python -m pytest tests/unit/test_packaging.py -q
```
Expected: `test_distribution_is_trichroma` FAILS (`PackageNotFoundError: trichroma`), and `test_chroma_and_engine_come_from_the_same_tree` FAILS (the `Chroma` distribution from `setup.py` is installed).

- [ ] **Step 3: Write `pyproject.toml`, delete the old files**

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "trichroma"
version = "0.6.0"
description = "Chroma's optical photon simulation on Triton (and on CUDA, with the cuda extra)"
readme = "README.md"
license = {file = "LICENSE.txt"}
requires-python = ">=3.9"
dependencies = [
    "numpy",
    "scipy",
    "torch",
    "triton",
]

[project.optional-dependencies]
# The original PyCUDA backend (CHROMA_BACKEND=cuda).
cuda = ["pycuda", "pytools==2022.1.2"]
# The event viewer (chroma.camera), chroma-server and the histogram helpers.
viz = ["pygame", "pyzmq", "uncertainties", "matplotlib"]
test = ["pytest"]

[project.urls]
Homepage = "https://github.com/youngsm/trichroma"

[tool.setuptools]
script-files = ["bin/chroma-sim", "bin/chroma-cam", "bin/chroma-geo", "bin/chroma-bvh", "bin/chroma-server"]

[tool.setuptools.packages.find]
include = ["chroma", "chroma.*"]

[tool.setuptools.package-data]
chroma = ["cuda/*.cu", "cuda/*.h"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "gpu: needs a CUDA GPU",
]
```

```bash
git rm -q setup.py pyproject.trichroma.toml
```

- [ ] **Step 4: Reinstall and run the tests**

```bash
uv pip uninstall Chroma
uv pip install -e '.[test]'
python -m pytest tests/unit/test_packaging.py -q
python -m pytest -q
```
Expected: `test_packaging.py` passes; the full suite passes on this CPU node with GPU tests and `tests/cuda_backend` skipped. Record the pass/skip counts.

Then a fresh, non-editable install from the checkout, timed, must compile nothing (no PyCUDA, no build of C extensions):

```bash
rm -rf /sdf/data/neutrino/youngsam/envs/trichroma-fresh
uv venv --python 3.10 /sdf/data/neutrino/youngsam/envs/trichroma-fresh
time uv pip install --python /sdf/data/neutrino/youngsam/envs/trichroma-fresh/bin/python /sdf/data/neutrino/youngsam/trichroma-merge
/sdf/data/neutrino/youngsam/envs/trichroma-fresh/bin/python -c "import sys, chroma, chroma.sim; print(chroma.sim.Simulation.__module__, 'torch' in sys.modules)"
```
Expected: the install reports only wheel downloads/installs; the last line prints `chroma.triton.simulation False`. Record the install time.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/unit/test_packaging.py
git commit -q -m "Package Chroma and its Triton engine as the trichroma distribution" -m "pyproject.toml replaces setup.py: torch and triton are core dependencies, PyCUDA and the viewer are extras, the import name stays chroma."
```

### Task 7: Documentation for the merged project

**Files:**
- Modify: `README.md`, `docs/guide.md`, `docs/reference/*.md`, `docs/correctness.md`, `docs/how_it_scales.md`, `docs/performance.md`
- Test: `tests/unit/test_docs.py`

**Interfaces:**
- Consumes: Task 6's install command and Task 5's default.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_docs.py`:

```python
"""Docs describe one install and Triton as the default (docs/history excepted)."""

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
CURRENT = [ROOT / "README.md"] + [p for p in (ROOT / "docs").rglob("*.md")
                                   if "history" not in p.parts and "chroma" not in p.parts[-2:]]

STALE = [
    "install Chroma (the chroma-lite distribution) first",
    "pip install -e trichroma",
    "git clone https://github.com/youngsm/trichroma.git",
    "when PyCUDA is installed",
    "Chroma itself (geometry, materials, detectors, events) still comes from chroma-lite",
    "from trichroma",
    "import trichroma",
    "python -m trichroma",
]


def test_docs_have_no_two_package_install():
    found = []
    for path in CURRENT:
        text = path.read_text()
        for phrase in STALE:
            if phrase.lower() in text.lower():
                found.append("%s: %r" % (path.relative_to(ROOT), phrase))
    assert found == []


def test_readme_links_resolve():
    import re

    text = (ROOT / "README.md").read_text()
    for target in re.findall(r"\]\(([^)#]+)\)", text):
        if not target.startswith("http"):
            assert (ROOT / target).exists(), target
```

- [ ] **Step 2: Run to see it fail**

```bash
python -m pytest tests/unit/test_docs.py -q
```
Expected: FAIL listing README/guide phrases.

- [ ] **Step 3: Apply the mechanical renames to the current docs**

```bash
python - <<'EOF'
import pathlib, re
root = pathlib.Path("/sdf/data/neutrino/youngsam/trichroma-merge")
rules = [(re.compile(r"src/trichroma/"), "chroma/triton/"),
         (re.compile(r"\bfrom trichroma import\b"), "from chroma.triton import"),
         (re.compile(r"(?<![/\w-])trichroma\.(?!git\b)"), "chroma.triton.")]
for p in [root / "README.md"] + sorted((root / "docs").rglob("*.md")):
    rel = p.relative_to(root).parts
    if "history" in rel or rel[:2] == ("docs", "chroma"):
        continue
    t = p.read_text(); n = t
    for pat, rep in rules: n = pat.sub(rep, n)
    if n != t: p.write_text(n); print("rewrote", p)
EOF
```

- [ ] **Step 4: Hand edits**

`README.md` - replace the `## Install` and the first paragraph of `## Use` with:

````markdown
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
````

and, under `## Use`, replace the paragraph that begins "When PyCUDA is not installed" with:

```markdown
`Simulation` runs on Triton by default. `CHROMA_BACKEND=cuda` selects the
original CUDA implementation (with the `cuda` extra installed).
```

Add at the end of `README.md`, before the license sentence:

```markdown
## Chroma

TriChroma is a fork of Chroma by Anthony LaTorre and Stanley Seibert, by way
of Ben Land's version, extended with wavelength shifters, dichroic filters and
analytic wire planes. Chroma's own README and documentation, including the
whitepaper, are in [docs/chroma](docs/chroma/README.md).
```

`docs/guide.md` - replace the intro paragraph and `## Choosing the backend` body:

```markdown
TriChroma is Chroma with a second implementation of `chroma.sim.Simulation`,
written in Triton, in the package `chroma.triton`. Geometry, materials,
detectors and events are Chroma's. This guide covers what you need to use it
day to day.

## Choosing the backend

`chroma.sim.Simulation` is one of two implementations:

- **TriChroma** (`chroma.triton`), the default;
- **CUDA Chroma**, Chroma's original PyCUDA code (install the `cuda` extra).

Set `CHROMA_BACKEND=cuda` or `CHROMA_BACKEND=triton` to choose. The choice is
fixed when `chroma.sim` is first imported.
```

In the rest of `docs/guide.md`, `docs/correctness.md` and `docs/reference/`, edit any remaining sentence that says the CUDA backend is used when PyCUDA is installed, or that tells the reader to install chroma-lite and TriChroma separately, to the one-package wording above. Leave `docs/history/` unchanged.

Read the regex's prose changes for lost full stops (a sentence that ended in `trichroma.`):

```bash
git diff --word-diff=plain -- README.md docs | grep -n "{+chroma.triton" | grep -v "import\|:mod:\|python -m\|chroma.triton.[a-z_]*(" | head -40
```
Fix any such sentence by hand.

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/unit/test_docs.py -q
python -m pytest -q
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add README.md docs tests/unit/test_docs.py
git commit -q -m "Docs: one package, Triton by default, chroma.triton module paths" -m "The README gains a Chroma lineage section pointing to docs/chroma; docs/history stays as written."
```

### Task 8: GPU validation on ampere and turing

**Files:**
- Create (outside the repo): `/sdf/data/neutrino/youngsam/trichroma-merge-tools/gpu_validate.sh`
- Create: `/sdf/data/neutrino/youngsam/containers/pytorch-2.5.1-cu124-devel.sif`
- Create: `/sdf/data/neutrino/youngsam/envs/trichroma-merge-cuda` (venv inside the container, with the `cuda` extra)

**Interfaces:**
- Consumes: the merged branch from Tasks 1-7.
- Produces: pass/fail of the full test suite on both GPUs, of the CUDA suite with `CHROMA_BACKEND=cuda`, and of the bitwise tape round trip; a results note for the user.

- [ ] **Step 1: Pull the CUDA devel image (for PyCUDA, which compiles against the CUDA toolkit)**

```bash
mkdir -p $APPTAINER_TMPDIR /sdf/data/neutrino/youngsam/containers
apptainer pull /sdf/data/neutrino/youngsam/containers/pytorch-2.5.1-cu124-devel.sif \
  docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

- [ ] **Step 2: Build the CUDA-extra environment inside it**

```bash
apptainer exec -B /sdf /sdf/data/neutrino/youngsam/containers/pytorch-2.5.1-cu124-devel.sif bash -lc '
  export UV_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/uv
  ~/.local/bin/uv venv --system-site-packages --python /opt/conda/bin/python /sdf/data/neutrino/youngsam/envs/trichroma-merge-cuda
  ~/.local/bin/uv pip install --python /sdf/data/neutrino/youngsam/envs/trichroma-merge-cuda/bin/python \
      -e "/sdf/data/neutrino/youngsam/trichroma-merge[cuda,test]"
  /sdf/data/neutrino/youngsam/envs/trichroma-merge-cuda/bin/python -c "import pycuda, torch, triton; print(torch.__version__, triton.__version__)"'
```
Expected: `2.5.1+cu124 3.1.0` (PyCUDA builds from source here; that is the slow, optional path).

- [ ] **Step 3: Write the GPU validation script**

`/sdf/data/neutrino/youngsam/trichroma-merge-tools/gpu_validate.sh`:

```bash
#!/bin/bash
# Usage (on a GPU node): gpu_validate.sh <label>
set -u
LABEL=$1
OUT=/sdf/data/neutrino/youngsam/trichroma-merge-tools/results/$LABEL
mkdir -p "$OUT"
export UV_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/uv
export TRITON_CACHE_DIR=/sdf/data/neutrino/youngsam/.cache/triton
REPO=/sdf/data/neutrino/youngsam/trichroma-merge
SIF=/sdf/data/neutrino/youngsam/containers/pytorch-2.5.1-cu124-devel.sif
cd "$REPO"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"

# 1. Triton backend, full suite, uv environment (no PyCUDA)
/sdf/data/neutrino/youngsam/envs/trichroma-merge/bin/python -m pytest -q -rs \
  > "$OUT/pytest_triton.txt" 2>&1; echo "pytest_triton exit $?" | tee -a "$OUT/summary.txt"

# 2. Same suite with PyCUDA installed (default must still be triton), then the CUDA suite
apptainer exec --nv -B /sdf "$SIF" bash -lc "cd $REPO && \
  /sdf/data/neutrino/youngsam/envs/trichroma-merge-cuda/bin/python -m pytest -q -rs tests/unit tests/integration tests/bitwise" \
  > "$OUT/pytest_with_pycuda.txt" 2>&1; echo "pytest_with_pycuda exit $?" | tee -a "$OUT/summary.txt"
apptainer exec --nv -B /sdf "$SIF" bash -lc "cd $REPO && CHROMA_BACKEND=cuda \
  /sdf/data/neutrino/youngsam/envs/trichroma-merge-cuda/bin/python -m pytest -q -rs tests/cuda_backend" \
  > "$OUT/pytest_cuda_backend.txt" 2>&1; echo "pytest_cuda_backend exit $?" | tee -a "$OUT/summary.txt"

# 3. Bitwise: record with CUDA Chroma, replay with chroma.triton, compare
apptainer exec --nv -B /sdf "$SIF" bash -lc "cd $REPO && \
  /sdf/data/neutrino/youngsam/envs/trichroma-merge-cuda/bin/python -m chroma.triton.tape.verify all \
    --fixture synthetic --work $OUT/tapes --transparency" \
  > "$OUT/verify_synthetic.txt" 2>&1; echo "verify_synthetic exit $?" | tee -a "$OUT/summary.txt"
```

```bash
chmod +x /sdf/data/neutrino/youngsam/trichroma-merge-tools/gpu_validate.sh
```

- [ ] **Step 4: Run on ampere and turing**

```bash
srun --account=neutrino:cider-nu --partition=ampere --gpus=1 --cpus-per-gpu=4 --mem=32G --time=2:00:00 \
  /sdf/data/neutrino/youngsam/trichroma-merge-tools/gpu_validate.sh ampere
srun --account=neutrino:cider-nu --partition=turing --gpus=1 --cpus-per-gpu=4 --mem=32G --time=2:00:00 \
  /sdf/data/neutrino/youngsam/trichroma-merge-tools/gpu_validate.sh turing
```

- [ ] **Step 5: Check results**

```bash
cat /sdf/data/neutrino/youngsam/trichroma-merge-tools/results/*/summary.txt
tail -3 /sdf/data/neutrino/youngsam/trichroma-merge-tools/results/*/pytest_triton.txt
```

Expected:
- `pytest_triton` and `pytest_with_pycuda` exit 0 on both GPUs, with no GPU test skipped for lack of a GPU.
- `pytest_cuda_backend`: the counts the conftest documents (17 pass, 31 fail, 5 import errors) or better; record the counts, do not try to fix the old CUDA suite.
- `verify_synthetic` exit 0 and a report of identical outputs.

For the baseline, run step 1's pytest command in the standalone repository at `6da1da3` (`/sdf/data/neutrino/youngsam/trichroma-src`, with its own `uv venv` and `pip install -e . 'git+https://github.com/youngsm/chroma-lite.git'`) on one GPU if any merged-branch test fails, to tell regressions from pre-existing failures. Report any regression to the user before continuing.

### Task 9: Push the branch (pre-approved)

chroma-lar's plan needs the branch on GitHub to resolve `git+https://github.com/youngsm/chroma-lite.git@merge-trichroma` and to build its container. The user pre-approved this push; push only if Task 8 passed, otherwise report first.

- [ ] **Step 1: Push**

```bash
git -C /sdf/data/neutrino/youngsam/trichroma-merge push -u origin merge-trichroma
```

### Task 10: Rename master to main, merge, rename the repositories (user approval, after chroma-lar validation)

Run only after chroma-lar's plan has validated against `merge-trichroma` and the user confirms each step.

- [ ] **Step 0: Rename the default branch `master` to `main` (after yes)**

```bash
gh api -X POST repos/youngsm/chroma-lite/branches/master/rename -f new_name=main
gh api repos/youngsm/chroma-lite --jq .default_branch
cd /sdf/data/neutrino/youngsam/trichroma-merge
git branch -m master main
git fetch origin
git branch -u origin/main main
git remote set-head origin -a
```
Expected: `main`. Then search the user's repos on this machine for `chroma-lite` references pinned to `@master` or `refs/heads/master` and list them for the user:

```bash
grep -rn "chroma-lite[^ ]*master" /sdf/data/neutrino/youngsam/chroma-lar-trichroma /sdf/group/neutrino/youngsam/sim/chroma-lar --include=Dockerfile --include=Singularity --include=*.py --include=*.toml --include=*.txt 2>/dev/null
```

- [ ] **Step 1: Merge to main (after yes)**

Open a PR from `merge-trichroma` to `main` on `youngsm/chroma-lite` (body without Claude attribution), or, if the user prefers, merge locally with `git merge --no-ff` and push `main`.

- [ ] **Step 2: Rename the old TriChroma repository and archive it (after yes)**

```bash
gh repo rename trichroma-standalone --repo youngsm/trichroma --yes
gh repo edit youngsm/trichroma-standalone --description "Archived: TriChroma now lives in youngsm/trichroma (chroma.triton)"
gh repo archive youngsm/trichroma-standalone --yes
```
(Before archiving, commit a README note in `trichroma-standalone` pointing to `https://github.com/youngsm/trichroma`, if the user wants one.)

- [ ] **Step 3: Rename chroma-lite (after yes)**

```bash
gh repo rename trichroma --repo youngsm/chroma-lite --yes
gh repo edit youngsm/trichroma --description "Chroma's optical photon simulation on Triton (TriChroma), with the original CUDA backend"
gh api repos/youngsm/chroma-lite --jq .full_name
```
Expected: `youngsm/trichroma` (the old name redirects).

- [ ] **Step 4: Re-point local clones**

```bash
git -C /sdf/data/neutrino/youngsam/trichroma-merge remote set-url origin https://github.com/youngsm/trichroma.git
for d in trichroma trichroma-legacy trichroma-primitives; do
  git -C /sdf/group/neutrino/youngsam/sim/triton_chroma/$d remote set-url origin https://github.com/youngsm/trichroma-standalone.git
done
```
(These clones belong to the old TriChroma repository; tell the user before touching them.)
