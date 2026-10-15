"""Bitwise mode: native RNG tapes recorded from CUDA Chroma, replayed exactly.

* :mod:`chroma.triton.tape.format` -- tape format, writer and reader.
* :mod:`chroma.triton.tape.record_cuda` -- the CUDA-backend recorder
  installed by :mod:`chroma.sim_cuda` when ``CHROMA_TRITON_TAPE`` is set.
  Imports PyCUDA; never imported by the Triton backend.
* :mod:`chroma.triton.tape.scene` -- the exact words ``GPUGeometry`` and
  ``GPUDetector`` upload, reproduced with NumPy (no PyCUDA).
* :mod:`chroma.triton.tape.reference` -- the reference loop: the engine's
  exact kernels driven launch by launch from the host (an independent check
  of the scheduling and the timing baseline; needs no detector).
* The replay itself is the exact mode of the production engine
  (:mod:`chroma.triton.engine.exact_mode`), which :mod:`chroma.triton.simulation`
  builds for ``CHROMA_TRITON_TAPE=replay:<dir>``.
* :mod:`chroma.triton.tape.fixtures` -- detectors and photon sources of the
  bitwise fixtures (LAr ones need ``chroma_lar``).
* :mod:`chroma.triton.tape.probes` -- CUDA probe kernels compiled from the
  original headers (unit ground truth for ``tests/bitwise/test_exact.py``).
* :mod:`chroma.triton.tape.verify` -- ``python -m chroma.triton.tape.verify``.

Reference: ``docs/reference/exact_mode.md``.

Importing this package does not import PyCUDA, torch or Triton.
"""
