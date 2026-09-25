"""Bitwise legacy mode support: native RNG tapes recorded from CUDA Chroma.

* :mod:`chroma.triton.legacy.tape` -- tape format, writer and reader.
* :mod:`chroma.triton.legacy.record_cuda` -- the CUDA-backend recorder
  installed by :mod:`chroma.sim_cuda` when ``CHROMA_TRITON_TAPE`` is set.
  Imports PyCUDA; never imported by the Triton backend.
* :mod:`chroma.triton.legacy.scene` -- the exact words ``GPUGeometry`` and
  ``GPUDetector`` upload, reproduced with NumPy (no PyCUDA).
* :mod:`chroma.triton.legacy.reference` -- a minimal replay harness that
  drives :mod:`chroma.triton.engine.exact` from a tape (a test tool, not an
  engine).
* :mod:`chroma.triton.legacy.verify` -- ``python -m chroma.triton.legacy.verify``.

Importing this package does not import PyCUDA, torch or Triton.
"""
