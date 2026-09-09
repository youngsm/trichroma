"""Additional spectral and multi-interaction fixtures for native legacy parity."""

import numpy as np
from chroma.detector import Detector
from chroma.event import Photons
from chroma.geometry import Solid
from chroma.make import box
from optical_comparison_cases import GRID, material, plane, surface
from optical_comparison_cases import make_case as baseline_case


def make_case(name, count):
    if name in ("reflect3wires", "reflect3wires_vuv", "pixel_tpc", "pixel_tpc_vuv"):
        return detector_case(name, count)
    if name not in ("spectral_fresnel", "spectral_multibounce", "bulk_reemit", "analytic_wires"):
        return baseline_case(name, count)
    rng = np.random.default_rng(71983)
    wavelengths = rng.uniform(125.0, 495.0, count)
    pos = rng.uniform(-5.0, 5.0, (count, 3))
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    pol = np.cross(direction, rng.normal(size=(count, 3)))
    pol /= np.linalg.norm(pol, axis=1)[:, None]
    medium = material("dispersive-medium")
    medium.set("refractive_index", 1.15 + 0.015 / (GRID / 1000.0) ** 2, GRID)
    medium.set("scattering_length", 200.0 * (GRID / 450.0) ** 4, GRID)
    geometry = Detector(medium)
    steps = 128
    if name == "spectral_fresnel":
        steps = 1
        medium.set("scattering_length", 1.0e30)
        other = material("second-dispersive-medium")
        other.set("refractive_index", 1.6 + 0.025 / (GRID / 1000.0) ** 2, GRID)
        geometry.add_solid(Solid(plane(), medium, other))
        pos[:, 2] = -10.0
        direction[:, 2] = abs(direction[:, 2])
        pol = np.cross(direction, rng.normal(size=(count, 3)))
        pol /= np.linalg.norm(pol, axis=1)[:, None]
    elif name == "spectral_multibounce":
        medium.set("absorption_length", 200.0 + GRID * 2.0, GRID)
        wall = surface(
            "mixed-wall", absorb=0.1, detect=0.1, reflect_diffuse=0.45, reflect_specular=0.35
        )
        geometry.add_pmt(
            Solid(box(200.0, 200.0, 200.0), medium, medium, surface=wall), displacement=np.zeros(3)
        )
        glass = material("dispersive-glass")
        glass.set("refractive_index", 1.5 + 0.03 / (GRID / 1000.0) ** 2, GRID)
        geometry.add_solid(
            Solid(box(60.0, 70.0, 8.0), glass, medium), displacement=[0.0, 0.0, 25.0]
        )
        coat = surface(
            "spectral-coat", model=2, reemit=0.83, reflect_specular=0.05, reflect_diffuse=0.05
        )
        coat.set("absorb", np.where(GRID < 360.0, 0.85, 0.05), GRID)
        coat.set("reemission_cdf", np.clip((GRID - 390.0) / 90.0, 0.0, 1.0), GRID)
        geometry.add_solid(
            Solid(box(70.0, 60.0, 5.0), medium, medium, surface=coat),
            displacement=[0.0, 0.0, -25.0],
        )
    elif name == "analytic_wires":
        medium.set("scattering_length", 250.0)
        medium.set("absorption_length", 500.0)
        geometry.add_pmt(
            Solid(box(200.0, 200.0, 200.0), medium, medium, surface=surface("monitor", detect=1.0)),
            displacement=np.zeros(3),
        )
        metal = material("wire-metal", n=2.5)
        metal.set("absorption_length", 0.0001)
        reflector = surface("wire-reflector", absorb=0.2, reflect_specular=0.6, reflect_diffuse=0.2)
        geometry.wireplanes = [
            dict(
                origin=[0.0, 0.0, 0.0],
                u=[0.0, 1.0, 0.0],
                v=[0.8, 0.0, 0.6],
                pitch=3.0,
                radius=0.25,
                umin=-80.0,
                umax=80.0,
                vmin=-70.0,
                vmax=70.0,
                v0=0.17,
                surface=reflector,
                material_inner=metal,
                material_outer=medium,
                color=0xFFFFFFFF,
            )
        ]
        pos[:, 2] -= 30.0
    else:
        medium.set("absorption_length", 40.0 + 0.15 * GRID, GRID)
        for probability, length_factor, lower, upper, delay in (
            (0.85, 1.5, 390.0, 435.0, 7.0),
            (0.63, 3.0, 435.0, 485.0, 21.0),
        ):
            medium.comp_absorption_length.append(
                np.column_stack((GRID, (40.0 + 0.15 * GRID) * length_factor))
            )
            medium.comp_reemission_prob.append(
                np.column_stack((GRID, np.full(len(GRID), probability)))
            )
            medium.comp_reemission_wvl_cdf.append(
                np.column_stack((GRID, np.clip((GRID - lower) / (upper - lower), 0.0, 1.0)))
            )
            t = np.arange(0.0, 1000.0, 0.05)
            medium.comp_reemission_time_cdf.append(
                np.column_stack((t, np.clip(t / delay, 0.0, 1.0)))
            )
        geometry.add_pmt(
            Solid(box(200.0, 200.0, 200.0), medium, medium, surface=surface("monitor", detect=1.0)),
            displacement=np.zeros(3),
        )
    return (
        geometry,
        Photons(pos, direction, pol, wavelengths, t=rng.uniform(0.0, 12.0, count)),
        steps,
        {},
    )


def detector_case(name, count):
    """Full existing LAr detector geometry, using identical supplied photons.

    These fixtures retain the geometry database's original optical tables.
    VUV cases add an explicitly synthetic original-model WLS coating; no
    production-only TPB timing or PMT readout law enters an equivalence test.
    """
    from chroma_lar.geometry.config_loader import build_detector_from_config

    pixel = name.startswith("pixel")
    vuv = name.endswith("_vuv")
    config = "detector_config_pixel" if pixel else "detector_config_reflect_reflect3wires"
    options = {"flatten": False}
    if not pixel:
        options["analytic_wires"] = True
    if vuv:
        coat = surface("legacy-fixture-tpb", model=2, reemit=0.85)
        coat.set("absorb", np.where(GRID < 200.0, 1.0, 0.0), GRID)
        coat.set("reemission_cdf", np.clip((GRID - 410.0) / 80.0, 0.0, 1.0), GRID)
        options["pmt_coating_surface"] = coat
    geometry = build_detector_from_config(config, **options)
    rng = np.random.default_rng(1981)
    positions = rng.uniform(-15.0, 15.0, (count, 3))
    positions[:, 0] += np.where(np.arange(count) % 2, 1000.0, -1000.0)
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    polarization = np.cross(direction, rng.normal(size=(count, 3)))
    polarization /= np.linalg.norm(polarization, axis=1)[:, None]
    wavelengths = rng.uniform(120.0, 140.0, count) if vuv else rng.uniform(390.0, 495.0, count)
    photons = Photons(
        positions,
        direction,
        polarization,
        wavelengths,
        t=rng.uniform(0.0, 1500.0, count),
    )
    return geometry, photons, 512, {}
