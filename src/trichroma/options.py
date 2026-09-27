"""``CHROMA_TRITON``: the Triton backend's options, a comma-separated list.

    CHROMA_TRITON=strict
    CHROMA_TRITON=legacy,roulette=0.05

Physics and exactness (all off by default):

* ``legacy``: keep CUDA Chroma's behaviour where production corrects it
  (docs/design.md lists what), with the production RNG, geometry and
  arithmetic: a statistical like-for-like comparison with CUDA Chroma.
* ``strict``: turn off the optimizations that can change a result at the
  float32 rounding level.
* ``roulette=<w>``: weighted mode only, Russian roulette below weight w
  (unbiased for every tally; not Chroma's weighted-mode semantics).

Debugging:

* ``legacy-wires``: CUDA Chroma's wire algorithm alone (``legacy`` implies it).
* ``wavefront``: the wavefront scheduler instead of the fused kernel.
* ``no-grid``: no empty-space grid for the wavefront scheduler.
* ``no-pipeline``: ``simulate`` reads, propagates and yields one batch at a
  time (the results are the same).

An unknown option raises ValueError.
"""

import os
from dataclasses import dataclass

FLAGS = ("legacy", "strict", "legacy-wires", "wavefront", "no-grid", "no-pipeline")


@dataclass(frozen=True)
class Options:
    legacy: bool = False
    strict: bool = False
    roulette: float = 0.0
    legacy_wires: bool = False
    wavefront: bool = False
    grid: bool = True
    pipeline: bool = True


def options(environ=None):
    """Parse ``CHROMA_TRITON`` (read at every call, so a change applies to the
    next engine or ``simulate`` call)."""
    environ = os.environ if environ is None else environ
    flags, roulette = set(), 0.0
    for item in environ.get("CHROMA_TRITON", "").split(","):
        item = item.strip().lower()
        if not item:
            continue
        key, sep, value = (part.strip() for part in item.partition("="))
        if key == "roulette" and sep:
            try:
                roulette = float(value)
            except ValueError:
                roulette = -1.0
            if not roulette >= 0.0:
                raise ValueError("CHROMA_TRITON: roulette=<w> needs a weight w >= 0, not %r" % value)
        elif key in FLAGS and not sep:
            flags.add(key)
        else:
            raise ValueError("CHROMA_TRITON: unknown option %r (options: %s, roulette=<w>)"
                             % (item, ", ".join(FLAGS)))
    return Options(legacy="legacy" in flags, strict="strict" in flags, roulette=roulette,
                   legacy_wires="legacy" in flags or "legacy-wires" in flags,
                   wavefront="wavefront" in flags, grid="no-grid" not in flags,
                   pipeline="no-pipeline" not in flags)
