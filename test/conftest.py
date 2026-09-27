"""The original Chroma test suite (CUDA backend). Run it explicitly with
``python -m pytest tests/cuda_backend`` where PyCUDA is installed
(``pip install .[cuda]``); it is skipped otherwise.

Its state predates TriChroma: on the validated environment (PyCUDA 2025.1,
CUDA 12.4) 17 tests pass, 31 fail and 5 modules do not import (nose-style
yield tests, the Geant4 generators chroma-lite dropped, ROOT I/O, stale
cache/BVH expectations and kernels that no longer compile)."""

import importlib.util

if importlib.util.find_spec("pycuda") is None:
    collect_ignore_glob = ["*.py"]
