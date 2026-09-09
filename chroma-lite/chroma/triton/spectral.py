"""Spectral optical transport with arbitrary sources and timed WLS surfaces.

The NumPy implementation is an independent, slow validation reference. The
Triton implementation uses the existing packed-BVH traversal and a spectral
interaction kernel. This path is separate from the 450 nm detector optimizer.
Supported optics: absorption, polarized Rayleigh, dielectric boundaries,
default surfaces, and effective wavelength-shifting coatings. Unsupported
analytic primitives, weighted photons, and other surface models are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from chroma.event import (NO_HIT, BULK_ABSORB, SURFACE_DETECT, SURFACE_ABSORB,
                          RAYLEIGH_SCATTER, REFLECT_DIFFUSE, REFLECT_SPECULAR,
                          SURFACE_REEMIT, SURFACE_TRANSMIT)
from .bvh import PackedBVH, nearest_hit_cpu, nearest_hit_bvh_cpu
from .boundary import offset_boundary_points
from .optical_response import OpticalHits, TabulatedCDF, uniform, validate_seed
from .photon_input import as_photon_batch, slice_batch
from .physics import fresnel_step, rayleigh_scatter
from .scene import compile_host_scene

C_MM_PER_NS = 299.792458
STEP_LIMIT = np.uint32(1 << 30)
TERMINAL = NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | int(STEP_LIMIT) | (1 << 31)


def group_velocity(wavelengths, refractive_index):
    """Derive c / (n - wavelength * dn/dwavelength), in mm/ns."""
    wl, n = np.asarray(wavelengths, float), np.asarray(refractive_index, float)
    if wl.ndim != 1 or len(wl) < 2 or n.shape[-1:] != wl.shape or np.any(np.diff(wl) <= 0):
        raise ValueError("group velocity requires an increasing wavelength grid and matching n")
    ng = n - wl * np.gradient(n, wl, axis=-1, edge_order=2 if len(wl) > 2 else 1)
    if not np.isfinite(ng).all() or np.any(ng <= 0):
        raise ValueError("refractive-index table gives nonpositive/nonfinite group index")
    return (C_MM_PER_NS / ng).astype(np.float32)


def sample_spectral_property(obj, name, grid):
    """Interpolate a finite, ordered two-column property table onto a grid."""
    table = np.asarray(getattr(obj, name), float)
    if table.ndim != 2 or table.shape[1] != 2 or len(table) < 1:
        raise ValueError(f"{name} must be a two-column property table")
    if not np.isfinite(table).all() or np.any(np.diff(table[:, 0]) <= 0):
        raise ValueError(f"{name} must be finite and have increasing wavelengths")
    return np.interp(grid, table[:, 0], table[:, 1]).astype(np.float32)


@dataclass(frozen=True)
class SpectralScene:
    host: object
    bvh: PackedBVH
    normals: np.ndarray
    velocities: np.ndarray
    time_offsets: np.ndarray
    time_x: np.ndarray
    time_cdf: np.ndarray
    time_pdf: np.ndarray
    reemit_to_material1: np.ndarray
    fingerprint: str

    @classmethod
    def compile(cls, geometry, *, wavelengths=None):
        # Choose a spectral grid explicitly in production calibration configs.
        host = compile_host_scene(geometry, wavelengths=wavelengths)
        if host.features.analytic_wireplane_count:
            raise ValueError("spectral transport requires meshed wires (analytic_wires=False)")
        if np.any(~np.isin(host.optics.surfaces.model[host.optics.surfaces.present], [0, 2])):
            raise ValueError("spectral transport supports only default and WLS surfaces")
        if host.optics.materials.component_offsets[-1] != 0:
            raise ValueError("bulk re-emission is not implemented in spectral transport; use effective WLS coatings")
        for sid in np.flatnonzero(host.optics.surfaces.present):
            if host.optics.surfaces.model[sid] == 2:
                if np.any(host.optics.surfaces.detect[sid] != 0):
                    raise ValueError("WLS surfaces cannot detect photons; use a separate photocathode surface")
                if np.any(host.optics.surfaces.reemit[sid] > 0):
                    cdf = host.optics.surfaces.reemission_cdf[sid]
                    if cdf[0] != 0 or cdf[-1] != 1:
                        raise ValueError("WLS CDF endpoints must be exactly 0 and 1 on the compiled grid")
            if host.optics.surfaces.model[sid] == 0 and np.any(host.optics.surfaces.detect[sid] > 0):
                selected = host.surface_index == sid
                if np.any(host.triangle_channel_index[selected] < 0):
                    raise ValueError("detecting surfaces must be assigned to PMT channels")
        grid = host.optics.wavelength_grid.values
        materials = geometry.unique_materials
        speeds = np.empty_like(host.optics.materials.refractive_index)
        for i, material in enumerate(materials):
            if getattr(material, "group_velocity", None) is not None:
                speeds[i] = sample_spectral_property(material, "group_velocity", grid)
            else:
                speeds[i] = group_velocity(grid, host.optics.materials.refractive_index[i])
        if not np.isfinite(speeds).all() or np.any(speeds <= 0):
            raise ValueError("group velocities must be finite and positive")
        normals = np.cross(host.vertices[host.triangles[:, 1]] - host.vertices[host.triangles[:, 0]],
                           host.vertices[host.triangles[:, 2]] - host.vertices[host.triangles[:, 0]])
        length = np.linalg.norm(normals, axis=1)
        if np.any(length <= 0):
            raise ValueError("spectral scene contains degenerate triangles")
        normals = (normals / length[:, None]).astype(np.float32)
        offsets, tx, ty, tp, sides = [0], [], [], [], []
        for i, surface in enumerate(geometry.unique_surfaces):
            distribution = getattr(surface, "reemission_time_cdf", None)
            if distribution is None:
                distribution = TabulatedCDF([0, 0], [0, 1], "instantaneous WLS")
            elif not isinstance(distribution, TabulatedCDF):
                table = np.asarray(distribution, float)
                distribution = TabulatedCDF(table[:, 0], table[:, 1])
            if distribution.x[0] < 0:
                raise ValueError("WLS delays must be nonnegative")
            if distribution.density is not None and np.any(np.diff(distribution.x.astype(np.float32)) <= 0):
                raise ValueError("WLS PDF time knots must remain distinct in float32")
            tx.extend(distribution.x)
            ty.extend(distribution.cdf)
            tp.extend(distribution.density if distribution.density is not None else np.full(len(distribution.x), -1.))
            offsets.append(len(tx))
            probability = float(getattr(surface, "reemission_to_material1", 0.5))
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("WLS escape probability must be in [0,1]")
            sides.append(probability)
        layer_offsets = host.bvh.layer_offsets
        layer_counts = tuple(np.diff((*layer_offsets, len(host.bvh.nodes))))
        bvh = PackedBVH(host.bvh.nodes, host.vertices[host.triangles], host.bvh.world_origin,
                        host.bvh.world_scale, layer_offsets, layer_counts,
                        degree=max(2, int((host.bvh.nodes[:, 3] >> np.uint32(28)).max())))
        arrays = [normals, speeds, np.asarray(offsets, np.int32), np.asarray(tx, np.float32),
                  np.asarray(ty, np.float32), np.asarray(tp, np.float32), np.asarray(sides, np.float32)]
        if any(not np.isfinite(array).all() for array in arrays):
            raise ValueError("spectral scene contains values outside finite float32 range")
        digest = hashlib.sha256(host.fingerprint.encode())
        for array in arrays:
            digest.update(array.tobytes())
            array.flags.writeable = False
        return cls(host, bvh, *arrays, digest.hexdigest())


@dataclass(frozen=True)
class SpectralResult:
    hits: OpticalHits
    final_state: dict
    steps: int
    scene_fingerprint: str

    @property
    def step_limit_count(self):
        return int(np.count_nonzero(self.final_state["flags"] & STEP_LIMIT))


def _interp(rows, index, wavelength, grid):
    f = np.clip((wavelength - grid.start) / grid.step, 0, grid.count - 1)
    lo = np.minimum(f.astype(np.int64), grid.count - 2)
    alpha = f - lo
    left, right = rows[index, lo], rows[index, lo+1]
    # Equal infinite lengths are legitimate (no process), avoid inf-inf.
    with np.errstate(invalid="ignore"):
        value = (1-alpha)*left + alpha*right
    return np.where(alpha <= 0, left, np.where(alpha >= 1, right,
                    np.where(left == right, left, value)))


def _hemisphere(normal, u_cos, u_phi, *, lambert=False):
    """Direction about a supplied hemisphere normal."""
    ref = np.zeros_like(normal)
    ref[:, 2] = 1
    pole = np.abs(normal[:, 2]) > .9
    ref[pole] = [0, 1, 0]
    tangent = np.cross(ref, normal)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    other = np.cross(normal, tangent)
    cosine = np.sqrt(u_cos) if lambert else u_cos
    sine = np.sqrt(np.maximum(0, 1-cosine*cosine))
    phi = 2*np.pi*u_phi
    return (cosine[:, None]*normal + sine[:, None] *
            (np.cos(phi)[:, None]*tangent + np.sin(phi)[:, None]*other))


def _polarization(direction, u):
    ref = np.zeros_like(direction)
    ref[:, 2] = 1
    ref[np.abs(direction[:, 2]) > .9] = [0, 1, 0]
    tangent = np.cross(ref, direction)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    other = np.cross(direction, tangent)
    return np.cos(2*np.pi*u)[:, None]*tangent + np.sin(2*np.pi*u)[:, None]*other


def _state(batch):
    return {name: np.array(getattr(batch, source), copy=True) for name, source in (
        ("pos", "pos"), ("direction", "direction"), ("polarization", "polarization"),
        ("wavelengths", "wavelengths"), ("times", "times"), ("flags", "flags"),
        ("last_hit", "last_hit_triangles"), ("photon_ids", "global_photon_ids"),
        ("event_indices", "event_indices"))} | {"channels": np.full(batch.photon_count, -1, np.int32)}


def _finish(state, steps, fingerprint):
    for field in ("pos", "direction", "polarization", "wavelengths", "times"):
        if not np.isfinite(state[field]).all():
            raise RuntimeError(f"spectral transport produced nonfinite {field}; check input scales and optical tables")
    detected = (state["flags"] & SURFACE_DETECT) != 0
    hits = OpticalHits(state["times"][detected], state["channels"][detected],
                       state["photon_ids"][detected], state["event_indices"][detected],
                       state["wavelengths"][detected])
    return SpectralResult(hits, state, steps, fingerprint)


def propagate_reference(scene, batch, *, seed=1, max_steps=1000, use_bvh=False):
    """Independent CPU optics with brute-force or validated CPU BVH intersections."""
    intersect = nearest_hit_bvh_cpu if use_bvh else nearest_hit_cpu
    state = _state(batch)
    host, optics = scene.host, scene.host.optics
    mat, surfaces, grid = optics.materials, optics.surfaces, optics.wavelength_grid
    steps = 0
    for step in range(max_steps):
        active = np.flatnonzero((state["flags"] & TERMINAL) == 0)
        if not len(active):
            break
        steps = step+1
        # Parallel ray/triangle candidates intentionally produce infinities;
        # nearest_hit_cpu masks them using its determinant acceptance test.
        with np.errstate(invalid="ignore", divide="ignore"):
            result = intersect(scene.bvh, state["pos"][active], state["direction"][active],
                                     last_hit=state["last_hit"][active], high_precision=True)
        missed = result.triangle_ids < 0
        state["flags"][active[missed]] |= NO_HIT
        a = active[~missed]
        if not len(a):
            continue
        triangles = result.triangle_ids[~missed]
        distance = result.distances[~missed]
        normal = scene.normals[triangles]
        toward_outside = np.sum(state["direction"][a] * normal, axis=1) > 0
        m1, m2 = host.material1_index[triangles], host.material2_index[triangles]
        incident = np.where(toward_outside, m1, m2)
        other = np.where(toward_outside, m2, m1)
        incoming_normal = np.where(toward_outside[:, None], -normal, normal)
        wl = state["wavelengths"][a]
        ids = state["photon_ids"][a]
        draws = {}
        def draw(slot):
            if slot not in draws:
                draws[slot] = uniform(ids, seed, step*32+slot)
            return draws[slot]
        absorption = _interp(mat.absorption_length, incident, wl, grid)
        scattering = _interp(mat.scattering_length, incident, wl, grid)
        da, ds = -absorption*np.log(draw(0)), -scattering*np.log(draw(1))
        absorbed = (da <= ds) & (da <= distance)
        scattered = (ds < da) & (ds <= distance)
        travel = np.minimum(distance, np.minimum(da, ds))
        state["pos"][a] += travel[:, None]*state["direction"][a]
        state["times"][a] += travel / _interp(scene.velocities, incident, wl, grid)
        state["flags"][a[absorbed]] |= BULK_ABSORB
        state["last_hit"][a[absorbed | scattered]] = -1
        if np.any(scattered):
            ix = a[scattered]
            d, p = rayleigh_scatter(state["direction"][ix], state["polarization"][ix],
                                    draw(2)[scattered], draw(3)[scattered],
                                    draw(4)[scattered], draw(5)[scattered])
            state["direction"][ix], state["polarization"][ix] = d, p
            state["flags"][ix] |= RAYLEIGH_SCATTER
        boundary = ~(absorbed | scattered)
        state["last_hit"][a[boundary]] = triangles[boundary]
        surface_ids = host.surface_index[triangles]
        # Each surface group includes a possible absent dielectric surface (-1).
        for sid in np.unique(surface_ids[boundary]):
            loc = np.flatnonzero(boundary & (surface_ids == sid))
            ix = a[loc]
            incoming = incoming_normal[loc]
            passed = np.ones(len(loc), bool)
            if sid >= 0 and surfaces.present[sid]:
                model = int(surfaces.model[sid])
                def prop(name):
                    return _interp(getattr(surfaces, name), sid, wl[loc], grid)
                absorb, diffuse, specular = prop("absorb"), prop("reflect_diffuse"), prop("reflect_specular")
                detect = prop("detect") if model == 0 else np.zeros(len(loc))
                u = draw(6)[loc]
                killed = u < absorb
                detected = (u >= absorb) & (u < absorb+detect)
                diff = (u >= absorb+detect) & (u < absorb+detect+diffuse)
                spec = (u >= absorb+detect+diffuse) & (u < absorb+detect+diffuse+specular)
                passed = ~(killed | detected | diff | spec)
                if model == 2:
                    reemit = killed & (draw(7)[loc] < prop("reemit"))
                    ri = ix[reemit]
                    if len(ri):
                        cdf = TabulatedCDF(grid.values, surfaces.reemission_cdf[sid])
                        state["wavelengths"][ri] = cdf.sample(draw(8)[loc][reemit])
                        begin, end = scene.time_offsets[sid:sid+2]
                        pdf = scene.time_pdf[begin:end]
                        dt = TabulatedCDF(scene.time_x[begin:end], scene.time_cdf[begin:end],
                                          density=pdf if pdf[0] >= 0 else None)
                        state["times"][ri] += dt.sample(draw(9)[loc][reemit])
                        into1 = draw(10)[loc][reemit] < scene.reemit_to_material1[sid]
                        outnormal = normal[loc][reemit] * np.where(into1[:, None], -1, 1)
                        d = _hemisphere(outnormal, draw(11)[loc][reemit], draw(12)[loc][reemit])
                        state["direction"][ri] = d
                        state["polarization"][ri] = _polarization(d, draw(13)[loc][reemit])
                        state["flags"][ri] |= SURFACE_REEMIT
                    killed &= ~reemit
                state["flags"][ix[killed]] |= SURFACE_ABSORB
                if np.any(detected):
                    channels = host.triangle_channel_index[triangles[loc][detected]]
                    if np.any(channels < 0):
                        raise ValueError("detecting surface is not assigned to a PMT channel")
                    state["channels"][ix[detected]] = channels
                    state["flags"][ix[detected]] |= SURFACE_DETECT
                if np.any(diff):
                    d = _hemisphere(incoming[diff], draw(14)[loc][diff], draw(15)[loc][diff], lambert=True)
                    state["direction"][ix[diff]] = d
                    state["polarization"][ix[diff]] = _polarization(d, draw(16)[loc][diff])
                    state["flags"][ix[diff]] |= REFLECT_DIFFUSE
                if np.any(spec):
                    n = incoming[spec]
                    d, p = state["direction"][ix[spec]], state["polarization"][ix[spec]]
                    state["direction"][ix[spec]] = d - 2*np.sum(d*n, axis=1)[:, None]*n
                    state["polarization"][ix[spec]] = p - 2*np.sum(p*n, axis=1)[:, None]*n
                    state["flags"][ix[spec]] |= REFLECT_SPECULAR
            if np.any(passed):
                pi = ix[passed]
                pl = loc[passed]
                fresnel = fresnel_step(state["direction"][pi], state["polarization"][pi], incoming[passed],
                                       _interp(mat.refractive_index, incident[pl], wl[pl], grid),
                                       _interp(mat.refractive_index, other[pl], wl[pl], grid),
                                       draw(17)[pl], draw(18)[pl])
                state["direction"][pi], state["polarization"][pi] = fresnel.direction, fresnel.polarization
                state["flags"][pi] |= np.where(fresnel.reflected, REFLECT_SPECULAR, SURFACE_TRANSMIT).astype(np.uint32)
        continuing = boundary & ((state["flags"][a] & TERMINAL) == 0)
        ix = a[continuing]
        if len(ix):
            state["pos"][ix] = offset_boundary_points(
                state["pos"][ix], scene.bvh.triangle_vertices[triangles[continuing]],
                state["direction"][ix])
    remaining = (state["flags"] & TERMINAL) == 0
    state["flags"][remaining] |= STEP_LIMIT
    return _finish(state, steps, scene.fingerprint)


class SpectralSimulation:
    """General photon-source entry point for the supported spectral optics."""

    def __init__(self, geometry, *, wavelengths=None, backend="triton", device=None, tile_size=65536):
        self.scene = geometry if isinstance(geometry, SpectralScene) else SpectralScene.compile(geometry, wavelengths=wavelengths)
        if backend not in ("triton", "reference", "reference_bvh"):
            raise ValueError("backend must be 'triton', 'reference', or 'reference_bvh'")
        if not isinstance(tile_size, (int, np.integer)) or tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        self.backend, self.tile_size = backend, tile_size
        self._device = None
        if backend == "triton":
            from .spectral_kernels import DeviceSpectralScene
            self._device = DeviceSpectralScene(self.scene, device=device)

    def simulate(self, photons, *, seed=1, max_steps=1000, photon_id_base=0, timings=None):
        """Propagate photons; optional ``timings`` list records synchronized GPU tile stages."""
        seed = validate_seed(seed)
        batch = as_photon_batch(photons, photon_id_base=photon_id_base)
        if not isinstance(max_steps, (int, np.integer)) or not 1 <= max_steps < 2**22:
            raise ValueError("max_steps must be an integer in [1, 2**22)")
        grid = self.scene.host.optics.wavelength_grid.values
        if np.any((batch.wavelengths < grid[0]) | (batch.wavelengths > grid[-1])):
            raise ValueError("input photon wavelengths are outside the compiled spectral grid")
        if np.any(batch.flags & TERMINAL):
            raise ValueError("input photons must be live (no terminal history flags)")
        if np.any(batch.last_hit_triangles < -1) or np.any(batch.last_hit_triangles >= self.scene.host.triangle_count):
            raise ValueError("last-hit triangle is outside the scene")
        states, steps = [], 0
        for start in range(0, batch.photon_count, self.tile_size):
            tile = slice_batch(batch, start, start+self.tile_size)
            if self.backend in ("reference", "reference_bvh"):
                result = propagate_reference(self.scene, tile, seed=seed, max_steps=max_steps,
                                             use_bvh=self.backend == "reference_bvh")
            else:
                result = self._device.propagate(tile, seed=seed, max_steps=max_steps, timings=timings)
            states.append(result.final_state)
            steps = max(steps, result.steps)
        if not states:
            return _finish(_state(batch), 0, self.scene.fingerprint)
        state = {name: np.concatenate([s[name] for s in states]) for name in states[0]}
        return _finish(state, steps, self.scene.fingerprint)
