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
