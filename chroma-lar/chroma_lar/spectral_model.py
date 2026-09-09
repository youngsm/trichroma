"""Compile calibrated physics into named, device-resident spectral tables.

Compilation validates the supported model and fingerprints effective tables
and full geometry. Rebuild the simulation when calibration data changes.
This layer has no transport queues, geometry queries, or readout policy.
"""

import hashlib

import numpy as np

from chroma.triton.optical_response import TabulatedCDF
from chroma.triton.optics import compile_optical_tables
from chroma.triton.spectral import group_velocity, sample_spectral_property
from .spectral_state import SpectralProperties, SourceProperties


class CompiledSpectralModel:
    def __init__(self, scene, calibration, device="cuda"):
        import torch

        self.scene = scene
        self.materials = [calibration.materials[name] for name in self.scene.tables.material_names]
        self.surfaces = [calibration.surfaces[name] for name in self.scene.tables.surface_names]
        self.optics = compile_optical_tables(
            self.materials, self.surfaces, wavelengths=calibration.wavelengths
        )
        m, s = self.optics.materials, self.optics.surfaces
        if m.component_offsets[-1] != 0 or np.any(~np.isin(s.model, [0, 2])):
            raise ValueError(
                "spectral transport supports default/WLS surfaces and no bulk re-emission"
            )
        for sid in np.flatnonzero(s.model == 2):
            if (
                np.any(s.detect[sid])
                or s.reemission_cdf[sid, 0] != 0
                or s.reemission_cdf[sid, -1] != 1
            ):
                raise ValueError(
                    "WLS requires a normalized spectral CDF and a separate detecting surface"
                )
            if np.any(self.scene.boxes.surface_index == sid) or np.any(
                self.scene.wires.surface_index == sid
            ):
                raise ValueError(
                    "this detector specialization supports WLS on PMT coating triangles"
                )
            selected = self.scene.pmt.scene_surface_index == sid
            if np.any(
                self.scene.pmt.scene_material1_index[selected]
                == self.scene.pmt.scene_material2_index[selected]
            ):
                raise ValueError("WLS coating requires distinct inside/outside materials")
        grid = calibration.wavelengths
        velocity = np.array(
            [
                (
                    sample_spectral_property(obj, "group_velocity", grid)
                    if getattr(obj, "group_velocity", None) is not None
                    else group_velocity(grid, m.refractive_index[i])
                )
                for i, obj in enumerate(self.materials)
            ],
            np.float32,
        )
        if not np.isfinite(velocity).all() or np.any(velocity <= 0):
            raise ValueError("group velocities must be finite and positive")
        offsets, xs, cs, ps, sides = [0], [], [], [], []
        for surface in self.surfaces:
            dist = getattr(surface, "reemission_time_cdf", TabulatedCDF([0, 0], [0, 1]))
            if dist.x[0] < 0 or (
                dist.density is not None and np.any(np.diff(dist.x.astype(np.float32)) <= 0)
            ):
                raise ValueError("invalid WLS time distribution")
            xs.extend(dist.x)
            cs.extend(dist.cdf)
            ps.extend(dist.density if dist.density is not None else np.full(len(dist.x), -1.0))
            offsets.append(len(xs))
            side = float(getattr(surface, "reemission_to_material1", 0.5))
            if not np.isfinite(side) or not 0 <= side <= 1:
                raise ValueError("invalid WLS escape probability")
            sides.append(side)
        arrays = (
            m.refractive_index,
            m.absorption_length,
            m.scattering_length,
            velocity,
            s.model,
            s.detect,
            s.absorb,
            s.reflect_diffuse,
            s.reflect_specular,
            s.reemit,
            s.reemission_cdf,
            np.asarray(offsets, np.int32),
            np.asarray(xs, np.float64),
            np.asarray(cs, np.float64),
            np.asarray(ps, np.float64),
            np.asarray(sides, np.float32),
        )
        self.properties = SpectralProperties(
            *(torch.from_numpy(np.array(a, copy=True)).to(device) for a in arrays)
        )
        wires = self.scene.wires
        self.wire_geometry = torch.from_numpy(
            np.column_stack(
                (
                    wires.origin,
                    wires.u,
                    wires.v,
                    wires.pitch,
                    wires.v0,
                    wires.radius,
                    wires.umin,
                    wires.umax,
                )
            )
        ).to(device)
        self.box_bounds = torch.from_numpy(
            np.stack((self.scene.boxes.bounds_min, self.scene.boxes.bounds_max), axis=1)
        ).to(device)
        distribution = calibration.source.spectrum
        self.source = SourceProperties(
            *(
                torch.from_numpy(np.asarray(a, dtype=dtype)).to(device)
                for a, dtype in (
                    ([0, len(distribution.x)], np.int32),
                    (distribution.x, np.float32),
                    (distribution.cdf, np.float32),
                    (
                        (
                            distribution.density
                            if distribution.density is not None
                            else np.full(len(distribution.x), -1.0)
                        ),
                        np.float32,
                    ),
                    (calibration.source.lifetimes_ns, np.float32),
                    (calibration.source.fractions, np.float64),
                )
            )
        )
        fingerprint = hashlib.sha256(calibration.fingerprint.encode())
        for name, value in sorted(self.scene.as_dict().items()):
            fingerprint.update(name.encode())
            fingerprint.update(
                value.tobytes() if isinstance(value, np.ndarray) else str(value).encode()
            )
        for value in (
            *arrays,
            grid,
            distribution.x,
            distribution.cdf,
            distribution.density if distribution.density is not None else [],
            calibration.source.lifetimes_ns,
            calibration.source.fractions,
            [calibration.source.rise_time_ns, calibration.source.yield_per_mev],
        ):
            value = np.ascontiguousarray(value)
            fingerprint.update(str((value.dtype, value.shape)).encode())
            fingerprint.update(value.tobytes())
        self.fingerprint = fingerprint.hexdigest()
