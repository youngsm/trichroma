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
