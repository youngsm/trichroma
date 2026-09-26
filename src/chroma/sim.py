"""``chroma.sim.Simulation`` resolves to the backend selected by ``CHROMA_BACKEND``
(unset: ``cuda`` when PyCUDA is installed, ``triton`` otherwise).

The original PyCUDA implementation lives unchanged in :mod:`chroma.sim_cuda`;
the Triton implementation lives in :mod:`trichroma.simulation` and
does not import PyCUDA. See :mod:`chroma.backend`.
"""

from chroma.backend import backend_name as _backend_name

if _backend_name() == "triton":
    from trichroma.simulation import Simulation, pick_seed  # noqa: F401
else:
    from chroma.sim_cuda import Simulation, pick_seed  # noqa: F401

__all__ = ["Simulation", "pick_seed"]
