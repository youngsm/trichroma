"""Versioned calibration bundles for spectral optical simulation.

Every bundle declares whether it is measured, provisional, or synthetic.
Numeric constants and two-column CSV/inline tables are supported. A digest
includes both the manifest and referenced file contents. No calibration is
downloaded or selected implicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np

from chroma.geometry import Material, Surface, silly_unique
from chroma.triton.optical_response import PMTResponse, TabulatedCDF
from .generator.scintillation import ScintillationSource


@dataclass
class OpticalCalibration:
    manifest: dict
    wavelengths: np.ndarray
    materials: dict
    surfaces: dict
    source: ScintillationSource
    response: PMTResponse
    digitizer: dict
    fingerprint: str
    files: dict

    @classmethod
    def load(cls, path):
        path = Path(path)
        manifest_bytes = path.read_bytes()
        config = json.loads(manifest_bytes)
        if config.get("schema_version") != 1:
            raise ValueError("unsupported optical calibration schema_version")
        if config.get("status") not in ("measured", "provisional", "synthetic"):
            raise ValueError("calibration status must be measured, provisional, or synthetic")
        if not isinstance(config.get("provenance"), str) or not config["provenance"].strip():
            raise ValueError("calibration must identify its provenance")
        grid = config["wavelength_grid_nm"]
        start, stop, step = (float(grid[k]) for k in ("start", "stop", "step"))
        if not np.isfinite([start, stop, step]).all() or start <= 0 or stop <= start or step <= 0:
            raise ValueError("invalid wavelength grid")
        count = int(round((stop-start)/step))+1
        if count < 2 or count > 100_000 or not np.isclose(start+(count-1)*step, stop, rtol=0, atol=1.e-5):
            raise ValueError("wavelength grid must end on a grid point and contain <=100000 points")
        wavelengths = (start+np.arange(count)*step).astype(np.float32)
        digest = hashlib.sha256(manifest_bytes)
        files = {}

        def table(spec):
            if isinstance(spec, dict):
                if "file" in spec:
                    filename = (path.parent / spec["file"]).resolve()
                    contents = filename.read_bytes()
                    files[str(spec["file"])] = hashlib.sha256(contents).hexdigest()
                    digest.update(str(spec["file"]).encode())
                    digest.update(contents)
                    values = np.loadtxt(filename, delimiter=",", comments="#")
                else:
                    values = np.asarray(spec["values"], dtype=float)
            else:
                values = np.asarray(spec, dtype=float)
            if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
                raise ValueError("calibration table must contain at least two [x,y] rows")
            if not np.isfinite(values).all():
                raise ValueError("calibration table must be finite")
            return values

        def curve(spec):
            if np.isscalar(spec):
                value = float(spec)
                if not np.isfinite(value):
                    raise ValueError("calibration constants must be finite")
                return np.full(len(wavelengths), value)
            values = table(spec)
            if np.any(np.diff(values[:, 0]) <= 0):
                raise ValueError("property wavelengths must increase")
            if values[0, 0] > wavelengths[0] or values[-1, 0] < wavelengths[-1]:
                raise ValueError("optical property does not cover the compiled wavelength range")
            return np.interp(wavelengths, values[:, 0], values[:, 1])

        def distribution(spec):
            values = table(spec)
            provenance = spec.get("provenance", config["provenance"]) if isinstance(spec, dict) else config["provenance"]
            kind = spec.get("kind", "pdf") if isinstance(spec, dict) else "pdf"
            if kind == "pdf":
                return TabulatedCDF.from_pdf(values[:, 0], values[:, 1], provenance)
            if kind == "cdf":
                return TabulatedCDF(values[:, 0], values[:, 1], provenance)
            raise ValueError("distribution kind must be pdf or cdf")

        materials = {}
        for name, properties in config["materials"].items():
            material = Material(name)
            for field in ("refractive_index", "absorption_length", "scattering_length"):
                material.set(field, curve(properties[field]), wavelengths)
            if "group_velocity" in properties:
                material.set("group_velocity", curve(properties["group_velocity"]), wavelengths)
            materials[name] = material
        surfaces = {}
        for name, properties in config["surfaces"].items():
            model = properties.get("model", "default")
            if model not in ("default", "wls"):
                raise ValueError("calibration surface model must be default or wls")
            surface = Surface(name, model=2 if model == "wls" else 0)
            for field in ("detect", "absorb", "reflect_diffuse", "reflect_specular", "reemit"):
                surface.set(field, curve(properties.get(field, 0)), wavelengths)
            if model == "wls":
                spectrum = distribution(properties["emission_spectrum"])
                if spectrum.x[0] < start or spectrum.x[-1] > stop:
                    raise ValueError("WLS spectrum extends outside the compiled wavelength grid")
                surface.set("reemission_cdf", spectrum.evaluate(wavelengths), wavelengths)
                surface.reemission_time_cdf = distribution(properties["emission_time"])
                surface.reemission_to_material1 = properties.get("escape_to_material1", .5)
            surfaces[name] = surface
        scint = config["scintillation"]
        source = ScintillationSource(
            distribution(scint["spectrum"]), tuple(scint["lifetimes_ns"]), tuple(scint["fractions"]),
            scint["yield_per_mev"], scint.get("rise_time_ns", 0), config["provenance"])
        if source.spectrum.x[0] < start or source.spectrum.x[-1] > stop:
            raise ValueError("source spectrum extends outside the compiled wavelength grid")
        pmt = config.get("pmt_response", {})
        response = PMTResponse(tts_sigma=pmt.get("tts_sigma_ns", 0), transit_time=pmt.get("transit_time_ns", 0),
                               gain=pmt.get("gain", 1), collection_efficiency=pmt.get("collection_efficiency", 1),
                               time_cdf=distribution(pmt["time_distribution"]) if "time_distribution" in pmt else None,
                               charge_cdf=distribution(pmt["charge_distribution"]) if "charge_distribution" in pmt else None)
        digitizer = dict(config["digitizer"])
        pulse = table(digitizer.pop("pulse"))
        digitizer["pulse_times_ns"], digitizer["pulse_adc_per_pe"] = pulse[:, 0], pulse[:, 1]
        return cls(config, wavelengths, materials, surfaces, source, response, digitizer, digest.hexdigest(), files)

    def apply_to_geometry(self, geometry):
        """Replace all material/surface objects by name before flattening.

        Missing entries raise instead of inheriting the original monochromatic
        approximations. The caller owns the supplied mutable geometry.
        """
        if hasattr(geometry, "mesh"):
            raise ValueError("apply calibration before geometry.flatten()")
        replacements = []
        missing = set()
        if geometry.detector_material is not None and geometry.detector_material.name not in self.materials:
            missing.add(f"detector_material:{geometry.detector_material.name}")
        for solid in geometry.solids:
            for field in ("material1", "material2", "surface"):
                values = np.asarray(getattr(solid, field), dtype=object)
                mapping = self.surfaces if field == "surface" else self.materials
                updated = []
                for obj in values.flat:
                    if obj is None:
                        updated.append(None)
                    elif obj.name in mapping:
                        updated.append(mapping[obj.name])
                    else:
                        missing.add(f"{field}:{obj.name}")
                        updated.append(obj)
                replacements.append((solid, field, np.asarray(updated, dtype=object).reshape(values.shape)))
        wire_replacements = []
        for plane in getattr(geometry, "wireplanes", ()) or ():
            for field in ("material_inner", "material_outer", "surface"):
                obj = plane[field]
                mapping = self.surfaces if field == "surface" else self.materials
                if obj is not None and obj.name not in mapping:
                    missing.add(f"wire.{field}:{obj.name}")
                else:
                    wire_replacements.append((plane, field, None if obj is None else mapping[obj.name]))
        if missing:
            raise ValueError("calibration missing geometry optics: " + ", ".join(sorted(missing)))
        for solid, field, values in replacements:
            setattr(solid, field, values)
        for plane, field, value in wire_replacements:
            plane[field] = value
        # Geometry.flatten() builds its lookup tables from these per-solid
        # caches, not from the replaced per-triangle arrays.
        for solid in geometry.solids:
            solid.unique_materials = silly_unique(np.concatenate((solid.material1, solid.material2)))
            solid.unique_surfaces = silly_unique(solid.surface)
        if geometry.detector_material is not None:
            geometry.detector_material = self.materials[geometry.detector_material.name]
        return geometry

    def build_detector(self, config_path, **overrides):
        """Build the repository detector with a coated PMT outer window."""
        from .geometry.config_loader import build_detector_from_dict, load_config_from_file
        roles = self.manifest["roles"]
        config_file = Path(__file__).parent / "config" / (str(config_path)+".py")
        config = load_config_from_file(str(config_file if config_file.exists() else config_path))
        options = dict(flatten=False, target_material=self.materials[roles["lar"]],
                       pmt_glass_material=self.materials[roles["glass"]],
                       pmt_photocathode_surface=self.surfaces[roles["photocathode"]],
                       pmt_coating_surface=self.surfaces[roles["tpb"]])
        options.update(overrides)
        options["flatten"] = False
        config.update(options)
        if config.get("detector_type", "wire") == "wire":
            config["analytic_wires"] = bool(overrides.get("analytic_wires", False))
        geometry = build_detector_from_dict(config)
        return self.apply_to_geometry(geometry)
