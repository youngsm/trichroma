"""TriChroma: a Triton backend for Chroma's optical photon simulation.

Chroma's API (``import chroma``) comes from chroma-lite. With the Triton
backend (the default when PyCUDA is not installed, or
``CHROMA_BACKEND=triton``), ``chroma.sim.Simulation`` is
:class:`trichroma.simulation.Simulation`.

* :mod:`trichroma.simulation` -- the ``Simulation`` class (batching, events,
  hits, DAQ, pipelining).
* :mod:`trichroma.engine` -- the transport engine: scene compiler, trees,
  fused transport kernel, physics, exact mode.
* :mod:`trichroma.sources` -- photon sources drawn on the GPU.
* :mod:`trichroma.tape` -- bitwise mode: record CUDA Chroma's random numbers
  and replay them exactly.

Importing this package does not import torch, Triton or PyCUDA.
"""

import importlib.util as _util

if _util.find_spec("chroma") is None:
    raise ImportError("TriChroma is a backend for Chroma: install chroma-lite first "
                      "(pip install 'git+https://github.com/youngsm/chroma-lite.git')")

__version__ = "0.1.0"
