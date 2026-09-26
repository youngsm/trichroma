"""Bitwise mode: native RNG tapes recorded from CUDA Chroma, replayed exactly.

* :mod:`trichroma.tape.format` -- tape format, writer and reader.
* :mod:`trichroma.tape.record_cuda` -- the CUDA-backend recorder
  installed by :mod:`chroma.sim_cuda` when ``CHROMA_TRITON_TAPE`` is set.
  Imports PyCUDA; never imported by the Triton backend.
* :mod:`trichroma.tape.scene` -- the exact words ``GPUGeometry`` and
  ``GPUDetector`` upload, reproduced with NumPy (no PyCUDA).
* :mod:`trichroma.tape.reference` -- the reference loop: the engine's
  exact kernels driven launch by launch from the host (an independent check
  of the scheduling and the timing baseline; needs no detector).
* The replay itself is the exact mode of the production engine
  (:mod:`trichroma.engine.exact_mode`), which :mod:`trichroma.simulation`
  builds for ``CHROMA_TRITON_TAPE=replay:<dir>``.
* :mod:`trichroma.tape.fixtures` -- detectors and photon sources of the
  bitwise fixtures (LAr ones need ``chroma_lar``).
* :mod:`trichroma.tape.probes` -- CUDA probe kernels compiled from the
  original headers (unit ground truth for ``tests/bitwise/test_exact.py``).
* :mod:`trichroma.tape.verify` -- ``python -m trichroma.tape.verify``.

Reference: ``docs/bitwise_mode.md``.

Importing this package does not import PyCUDA, torch or Triton.
"""
