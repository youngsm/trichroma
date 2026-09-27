"""Backend selection for :class:`chroma.sim.Simulation`.

``CHROMA_BACKEND`` chooses the implementation at import time of
:mod:`chroma.sim`:

* ``cuda``: the original PyCUDA implementation.
* ``triton``: the Triton implementation, :mod:`trichroma.simulation`, which
  does not import PyCUDA.

Unset, it is ``cuda`` when PyCUDA is installed and ``triton`` otherwise.

``CHROMA_TRITON_TAPE`` selects the bitwise legacy mode of the Triton backend
(and the matching schedule hook of the CUDA backend):

* unset or ``off``: production physics with counter-based random numbers.
* ``canonical``: original arithmetic, XORWOW streams and launch schedule, with
  surviving photons re-queued in ascending order (a legal original schedule).
* ``record:<dir>``: CUDA backend only; record the actual schedule to ``<dir>``.
* ``replay:<dir>``: Triton backend only; replay a recorded schedule.
"""

import functools
import importlib.util
import os
from dataclasses import dataclass

BACKENDS = ("cuda", "triton")


@functools.lru_cache(maxsize=None)
def default_backend():
    """``cuda`` when PyCUDA is installed, ``triton`` otherwise."""
    return "cuda" if importlib.util.find_spec("pycuda") is not None else "triton"


def backend_name(environ=None):
    """Return the selected backend name, validating ``CHROMA_BACKEND``."""
    environ = os.environ if environ is None else environ
    value = environ.get("CHROMA_BACKEND", "").strip().lower() or default_backend()
    if value not in BACKENDS:
        raise ValueError("CHROMA_BACKEND must be one of %s, not %r" % (", ".join(BACKENDS), value))
    return value


@dataclass(frozen=True)
class TapeMode:
    """Parsed ``CHROMA_TRITON_TAPE`` setting."""

    mode: str = "off"  # off | canonical | record | replay
    directory: str = None

    @property
    def enabled(self):
        return self.mode != "off"


def tape_mode(environ=None):
    """Parse ``CHROMA_TRITON_TAPE``."""
    environ = os.environ if environ is None else environ
    value = environ.get("CHROMA_TRITON_TAPE", "").strip()
    if value.lower() in ("", "0", "off", "false", "no"):
        return TapeMode()
    if value.lower() in ("1", "on", "true", "yes", "canonical"):
        return TapeMode("canonical")
    kind, sep, directory = value.partition(":")
    kind = kind.strip().lower()
    if kind in ("record", "replay") and sep and directory.strip():
        return TapeMode(kind, directory.strip())
    raise ValueError(
        "CHROMA_TRITON_TAPE must be off, canonical, record:<dir> or replay:<dir>, not %r" % value
    )
