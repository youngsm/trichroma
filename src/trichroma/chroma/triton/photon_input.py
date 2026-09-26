"""Adapters for arbitrary optical photon input and stable photon identity."""
from __future__ import annotations

import numpy as np

from .runtime import PhotonBatch


def as_photon_batch(photons, *, photon_id_base=0):
    """Accept a PhotonBatch or Chroma Photons without regenerating its source."""
    if isinstance(photons, PhotonBatch):
        batch = photons
    else:
        n = len(photons.pos)
        if not isinstance(photon_id_base, (int, np.integer)) or photon_id_base < 0 or photon_id_base + n > 2**63:
            raise ValueError("photon ID range is outside nonnegative int64")
        batch = PhotonBatch(
            pos=photons.pos, direction=photons.dir, polarization=photons.pol,
            wavelengths=photons.wavelengths, times=photons.t,
            last_hit_triangles=photons.last_hit_triangles, flags=photons.flags,
            weights=photons.weights, event_indices=photons.evidx,
            global_photon_ids=np.arange(photon_id_base, photon_id_base+n, dtype=np.int64),
            channels=photons.channel,
        )
    for name in ("pos", "direction", "polarization", "wavelengths", "times"):
        if not np.isfinite(getattr(batch, name)).all():
            raise ValueError(f"photon {name} must be finite")
    if np.any(batch.wavelengths <= 0):
        raise ValueError("photon wavelengths must be positive")
    for name in ("direction", "polarization"):
        if not np.allclose(np.linalg.norm(getattr(batch, name), axis=1), 1, atol=2e-5):
            raise ValueError(f"photon {name} vectors must be normalized")
    if np.any(np.abs(np.sum(batch.direction * batch.polarization, axis=1)) > 2e-5):
        raise ValueError("photon polarization must be transverse to direction")
    if np.any(batch.weights != 1):
        raise ValueError("spectral transport currently requires unweighted photons")
    return batch


def slice_batch(batch, start, stop):
    return PhotonBatch(**{name: getattr(batch, name)[start:stop] for name in (
        "pos", "direction", "polarization", "wavelengths", "times",
        "last_hit_triangles", "flags", "weights", "event_indices",
        "global_photon_ids", "channels")})
