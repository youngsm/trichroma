"""Historical entry point of the tape replay (kept as a thin alias).

The bitwise legacy mode is the exact mode of the production engine:
``ProductionEngine(detector, seed=, device=, tape=TapeMode('replay', dir))``
(:mod:`chroma.triton.engine.exact_mode`), which the compatibility layer builds
for ``CHROMA_TRITON_TAPE=replay:<dir>``.
"""


def LegacyEngine(detector, seed=None, device=None, nthreads_per_block=512, max_blocks=1024, tape=None,
                 photon_tracking=None, use_packed=None):
    """A :class:`chroma.triton.engine.core.ProductionEngine` in exact mode."""
    from chroma.triton.engine.core import ProductionEngine

    if tape is None or not tape.enabled:
        raise ValueError("LegacyEngine needs a TapeMode('replay', <dir>)")
    if seed is None:
        raise ValueError("LegacyEngine needs the Simulation's seed (it must match the tape)")
    return ProductionEngine(detector, seed=seed, device=device if device is not None else "cuda", tape=tape,
                            simulation=dict(nthreads_per_block=nthreads_per_block, max_blocks=max_blocks,
                                            photon_tracking=photon_tracking, use_packed=use_packed))
