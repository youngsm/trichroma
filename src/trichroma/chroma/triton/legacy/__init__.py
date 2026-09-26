"""Bitwise legacy mode support: native RNG tapes recorded from CUDA Chroma.

* :mod:`chroma.triton.legacy.tape` -- tape format, writer and reader.
* :mod:`chroma.triton.legacy.record_cuda` -- the CUDA-backend recorder
  installed by :mod:`chroma.sim_cuda` when ``CHROMA_TRITON_TAPE`` is set.
  Imports PyCUDA; never imported by the Triton backend.
* :mod:`chroma.triton.legacy.scene` -- the exact words ``GPUGeometry`` and
  ``GPUDetector`` upload, reproduced with NumPy (no PyCUDA).
* :mod:`chroma.triton.legacy.reference` -- the reference loop: the engine's
  exact kernels driven launch by launch from the host (an independent check
  of the scheduling and the timing baseline; needs no detector).
* :mod:`chroma.triton.legacy.engine` -- ``LegacyEngine``: historical alias of
  ``ProductionEngine(..., tape=...)``. The replay itself is the exact mode of
  the production engine (:mod:`chroma.triton.engine.exact_mode`), which the
  compatibility layer builds for ``CHROMA_TRITON_TAPE=replay:<dir>``.
* :mod:`chroma.triton.legacy.fixtures` -- detectors and photon sources of the
  bitwise fixtures (LAr ones need ``chroma_lar``).
* :mod:`chroma.triton.legacy.probes` -- CUDA probe kernels compiled from the
  original headers (unit ground truth for ``test/test_legacy_exact.py``).
* :mod:`chroma.triton.legacy.verify` -- ``python -m chroma.triton.legacy.verify``.

Reference: ``docs/triton_legacy_tape.md``.

Importing this package does not import PyCUDA, torch or Triton.
"""
