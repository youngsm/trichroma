"""Conservative discovery of homogeneous boxes, independent of detector names.

Certificates are an acceleration cover, not a partition of all physical space.
Dropping a cell only removes an optimization. No artificial boundary moves a
photon or consumes random numbers; uncertified flights use exact geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

import numpy as np

from .primitives import Bounds, PrimitiveScene


def subtract_box(cell, obstacle):
    """Partition the open part of cell outside obstacle into at most six boxes."""
    if not cell.overlaps(obstacle):
        return [cell]
    lo, hi = cell.lower.copy(), cell.upper.copy()
    cut_lo = np.maximum(lo, obstacle.lower)
    cut_hi = np.minimum(hi, obstacle.upper)
    pieces = []
    # Prefer the axis yielding the largest slabs, retaining large bulk regions
    # instead of fragmenting them through incidental obstacle ordering.
    widths = hi - lo
    possible = np.maximum(cut_lo - lo, hi - cut_hi) / widths
    for axis in np.argsort(-possible, kind="stable"):
        if lo[axis] < cut_lo[axis]:
            edge = hi.copy()
            edge[axis] = cut_lo[axis]
            pieces.append(Bounds(lo, edge))
            lo[axis] = cut_lo[axis]
        if hi[axis] > cut_hi[axis]:
            edge = lo.copy()
            edge[axis] = cut_hi[axis]
            pieces.append(Bounds(edge, hi))
            hi[axis] = cut_hi[axis]
    return pieces


def _rank(bounds):
    return (-bounds.volume, *bounds.lower, *bounds.upper)


@dataclass(frozen=True)
class HomogeneousRegion:
    bounds: Bounds
    material: int
    domain: str


@dataclass(frozen=True)
class RegionCompilation:
    regions: tuple[HomogeneousRegion, ...]
    rejected_domains: tuple[tuple[str, str], ...]
    dropped_cells: int
    fingerprint: str
    coarse_domains: tuple[str, ...] = ()

    def select(self, positions, *, material=None, weights=None):
        """Select one region for a batch without changing physical eligibility.

        Prefer the region containing most weighted source points. If none
        contains a point, choose the largest eligible region: every other
        photon still follows the exact boundary path.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape == (0,):
            positions = positions.reshape(0, 3)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("region selection positions must have shape [N,3]")
        weights = np.ones(len(positions)) if weights is None else np.asarray(weights)
        if (
            weights.shape != (len(positions),)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
        ):
            raise ValueError("region selection weights must be finite and nonnegative")
        if not np.isfinite(positions).all():
            raise ValueError("region selection positions must be finite")
        candidates = [r for r in self.regions if material is None or r.material == material]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda r: (float(weights[r.bounds.contains(positions)].sum()), r.bounds.volume),
        )

    def as_dict(self):
        return {
            "fingerprint": self.fingerprint,
            "dropped_cells": self.dropped_cells,
            "coarse_domains": list(self.coarse_domains),
            "rejected_domains": dict(self.rejected_domains),
            "regions": [
                {
                    "lower": r.bounds.lower.tolist(),
                    "upper": r.bounds.upper.tolist(),
                    "material": r.material,
                    "domain": r.domain,
                }
                for r in self.regions
            ],
        }


def _centered_clear_box(seed, blocker_lower, blocker_upper):
    """Contract a seed about its center until every obstacle lies outside it."""
    center = (seed.lower + seed.upper) / 2
    half = (seed.upper - seed.lower) / 2
    entry = np.max(np.maximum(blocker_lower - center, center - blocker_upper) / half, axis=1)
    factor = min(1.0, float(np.min(entry, initial=1.0)))
    if factor <= 0:
        return None
    half *= factor * (1.0 - 1e-12)
    return Bounds(center - half, center + half)


def compile_regions(
    scene: PrimitiveScene,
    *,
    max_regions=64,
    max_fragments=4096,
    max_subtractions=4096,
    min_extent=0.0,
):
    """Subtract conservative obstacles from certified interiors.

    A child interface must expose the domain material on its outside. Unknown
    or inconsistent outside materials disable that domain's optimization.
    Partially overlapping volume declarations are likewise left uncertified.
    Bounds are moved inward to float32 before they are emitted for GPU use.
    """
    for value in (max_regions, max_fragments, max_subtractions):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError("region/fragment limits must be positive integers")
    if not np.isfinite(min_extent) or min_extent < 0:
        raise ValueError("min_extent must be finite and nonnegative")
    obstacles = scene.bounded_obstacles()
    emitted, rejected, dropped, coarse = [], [], 0, []
    for domain in scene.volumes:
        others = [
            v
            for v in scene.volumes
            if v is not domain
            and domain.bounds.overlaps(v.bounds)
            and not v.encloses(domain.bounds)
        ]
        if any(not domain.encloses(v.bounds) for v in others):
            rejected.append((domain.name, "partially overlapping volume"))
            continue
        # Only directly nested children expose an interface to this domain.
        children = [
            v for v in others if not any(w is not v and w.encloses(v.bounds) for w in others)
        ]
        exposed = [
            o
            for o in obstacles
            if domain.bounds.overlaps(o.bounds) and not any(v.encloses(o.bounds) for v in children)
        ]
        if any(v.material_outside != domain.material_inside for v in children) or any(
            o.exterior_material != domain.material_inside for o in exposed
        ):
            rejected.append((domain.name, "inconsistent or unknown exterior material"))
            continue
        blockers = [v.bounds for v in children] + [o.bounds for o in exposed]
        blocker_lower = np.asarray([b.lower for b in blockers]).reshape(-1, 3)
        blocker_upper = np.asarray([b.upper for b in blockers]).reshape(-1, 3)
        cells = list(domain.seed_cells())
        if len(blockers) > max_subtractions:
            # Dense sensor arrays need bounded compile cost. A contraction is
            # checked against ALL original obstacles in one vectorized pass.
            # Only the acceleration cover becomes coarser; no obstacle is lost.
            coarse.append(domain.name)
            cells = [
                clear
                for seed in cells
                if (clear := _centered_clear_box(seed, blocker_lower, blocker_upper)) is not None
            ]
        for blocker in (sorted(blockers, key=_rank) if len(blockers) <= max_subtractions else ()):
            if not cells:
                break
            lower = np.asarray([cell.lower for cell in cells])
            upper = np.asarray([cell.upper for cell in cells])
            overlaps = np.all((lower < blocker.upper) & (upper > blocker.lower), axis=1)
            if not overlaps.any():
                continue
            cells = [
                part
                for cell, hit in zip(cells, overlaps)
                for part in (subtract_box(cell, blocker) if hit else (cell,))
            ]
            if len(cells) > max_fragments:
                cells.sort(key=_rank)
                dropped += len(cells) - max_fragments
                cells = cells[:max_fragments]
        for cell in cells:
            interior = cell.float32_interior()
            if interior is None or np.any(interior.upper - interior.lower <= min_extent):
                dropped += 1
                continue
            # Audit the certificate against every original bound, independently
            # of how subtraction fragmented or limited the candidate list.
            blocked = np.any(
                np.all((interior.lower < blocker_upper) & (interior.upper > blocker_lower), axis=1)
            )
            if not domain.encloses(interior) or blocked:
                raise RuntimeError("region certificate intersects a physical boundary")
            emitted.append(HomogeneousRegion(interior, int(domain.material_inside), domain.name))
    emitted.sort(key=lambda r: (_rank(r.bounds), r.material, r.domain))
    dropped += max(0, len(emitted) - max_regions)
    emitted = emitted[:max_regions]
    payload = {
        "volumes": [
            (v.name, v.descriptor(), int(v.material_inside), int(v.material_outside))
            for v in scene.volumes
        ],
        "obstacles": [
            (
                o.name,
                o.bounds.lower.tolist(),
                o.bounds.upper.tolist(),
                None if o.exterior_material is None else int(o.exterior_material),
            )
            for o in obstacles
        ],
        "limits": [int(max_regions), int(max_fragments), int(max_subtractions), float(min_extent)],
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return RegionCompilation(tuple(emitted), tuple(rejected), dropped, fingerprint, tuple(coarse))
