"""Configurable LAr scintillation sources (mm, nm, ns, MeV).

No detector calibration is silently selected. Lifetimes, component fractions,
yield, and emission spectrum are supplied by the caller. Energy depositions
are external inputs; this module does not transport charged particles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chroma.triton.optical_response import TabulatedCDF, uniform, validate_seed
from chroma.triton.runtime import PhotonBatch


@dataclass(frozen=True)
class ScintillationSource:
    spectrum: TabulatedCDF
    lifetimes_ns: tuple
    fractions: tuple
    yield_per_mev: float
    rise_time_ns: float = 0.0
    provenance: str = "user supplied"

    def __post_init__(self):
        tau, weights = np.asarray(self.lifetimes_ns, float), np.asarray(self.fractions, float)
        if tau.ndim != 1 or len(tau) == 0 or weights.shape != tau.shape:
            raise ValueError("scintillation lifetimes and fractions must be matching vectors")
        if not np.isfinite(tau).all() or not np.isfinite(weights).all() or np.any(tau < 0) or np.any(weights < 0):
            raise ValueError("scintillation lifetimes and fractions must be finite and nonnegative")
        if not np.isclose(weights.sum(), 1, rtol=0, atol=1e-8):
            raise ValueError("scintillation fractions must sum to one")
        if not np.isfinite([self.yield_per_mev, self.rise_time_ns]).all() or self.yield_per_mev < 0 or self.rise_time_ns < 0:
            raise ValueError("yield and rise time must be finite and nonnegative")
        if self.spectrum.x[0] <= 0:
            raise ValueError("emission wavelengths must be positive")
        object.__setattr__(self, "lifetimes_ns", tuple(tau))
        object.__setattr__(self, "fractions", tuple(weights))

    def photons(self, positions, *, times=0.0, event_indices=0, seed=1, photon_id_base=0):
        seed = validate_seed(seed)
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("positions must be finite [N,3] coordinates")
        n = len(positions)
        if not isinstance(photon_id_base, (int, np.integer)) or photon_id_base < 0 or photon_id_base + n > 2**63:
            raise ValueError("photon ID range is outside nonnegative int64")
        ids = np.arange(photon_id_base, photon_id_base+n, dtype=np.int64)
        birth = np.broadcast_to(np.asarray(times, dtype=float), (n,)).copy()
        events = np.broadcast_to(np.asarray(event_indices), (n,))
        if not np.isfinite(birth).all() or np.any(events < 0) or np.any(events > np.iinfo(np.uint32).max) or np.any(events != events.astype(np.uint32)):
            raise ValueError("birth times must be finite and event indices nonnegative uint32 integers")
        def draw(stream):
            return uniform(ids, seed, 0x10000000 + stream)
        component = np.searchsorted(np.cumsum(self.fractions), draw(0), side="right")
        birth += -np.asarray(self.lifetimes_ns)[component] * np.log(draw(1))
        if self.rise_time_ns:
            birth -= self.rise_time_ns * np.log(draw(2))
        z = 2*draw(3)-1
        phi = 2*np.pi*draw(4)
        r = np.sqrt(np.maximum(0, 1-z*z))
        direction = np.column_stack((r*np.cos(phi), r*np.sin(phi), z))
        # Tangent frame with no pole singularity.
        tangent = np.column_stack((-np.sin(phi), np.cos(phi), np.zeros(n)))
        other = np.cross(direction, tangent)
        alpha = 2*np.pi*draw(5)
        polarization = np.cos(alpha)[:, None]*tangent + np.sin(alpha)[:, None]*other
        return PhotonBatch(
            pos=positions, direction=direction, polarization=polarization,
            wavelengths=self.spectrum.sample(draw(6)), times=birth,
            last_hit_triangles=np.full(n, -1, np.int32), flags=np.full(n, 1 << 11, np.uint32),
            weights=np.ones(n, np.float32), event_indices=events,
            global_photon_ids=ids, channels=np.zeros(n, np.uint32),
        )

    def from_depositions(self, positions, energy_mev, *, times=0.0, event_indices=0,
                         quenching=1.0, seed=1, photon_id_base=0, max_photons=10_000_000):
        """Poisson emission at deposition points with caller-supplied quenching.

        ``yield_per_mev`` and ``quenching`` must not both include the same
        recombination correction. Extended tracks can be supplied as steps.
        """
        positions, birth, events, counts = self.sample_depositions(positions, energy_mev, times=times,
            event_indices=event_indices, quenching=quenching, seed=seed, max_photons=max_photons)
        rows = np.repeat(np.arange(len(positions)), counts)
        return self.photons(positions[rows], times=birth[rows], event_indices=events[rows],
                            seed=seed, photon_id_base=photon_id_base)

    def sample_depositions(self, positions, energy_mev, *, times=0.0, event_indices=0,
                           quenching=1.0, seed=1, max_photons=10_000_000):
        """Validate deposition metadata and sample counts without expanding photons.

        CPU and GPU source paths share the same Poisson count law and seed.
        """
        seed = validate_seed(seed)
        if not isinstance(max_photons, (int, np.integer)) or max_photons < 0:
            raise ValueError("max_photons must be a nonnegative integer")
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
            raise ValueError("deposition positions must be finite [N,3]")
        n = len(positions)
        birth = np.broadcast_to(np.asarray(times, dtype=float), (n,))
        events = np.broadcast_to(np.asarray(event_indices), (n,))
        if (not np.isfinite(birth).all() or not np.isfinite(events).all()
                or np.any(events < 0) or np.any(events > np.iinfo(np.uint32).max)
                or np.any(events != events.astype(np.uint32))):
            raise ValueError("deposition times must be finite and event indices nonnegative uint32 integers")
        energy = np.broadcast_to(np.asarray(energy_mev, float), (n,))
        q = np.broadcast_to(np.asarray(quenching, float), (n,))
        if not np.isfinite(energy).all() or np.any(energy < 0) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("energies must be nonnegative and quenching in [0,1]")
        mean = energy * self.yield_per_mev * q
        if mean.sum() > max_photons:
            raise ValueError("expected emission exceeds max_photons; split the deposition batch")
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x10000010]))
        counts = rng.poisson(mean)
        if counts.sum() > max_photons:
            raise ValueError("sampled emission exceeds max_photons; split the deposition batch")
        return positions, birth, events, counts
