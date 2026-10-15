"""The original Chroma test suite (CUDA backend). It is opt-in: run it with
``CHROMA_CUDA_SUITE=1 CHROMA_BACKEND=cuda python -m pytest tests/cuda_backend``
where PyCUDA is installed (``pip install '.[cuda]'``); a plain ``python -m pytest``
skips it.

Its state predates TriChroma: on PyTorch 2.5.1 / CUDA 12.4 with PyCUDA 2026.1
(A100 and RTX 2080 Ti, ``--continue-on-collection-errors``) 15 tests pass, 26
fail, 2 are skipped and 9 modules do not import (nose-style yield tests, the
Geant4 generators chroma-lite dropped, ROOT I/O, stale cache/BVH expectations
and kernels that no longer compile)."""

import importlib.util
import os

if not os.environ.get("CHROMA_CUDA_SUITE") or importlib.util.find_spec("pycuda") is None:
    collect_ignore_glob = ["*.py"]
