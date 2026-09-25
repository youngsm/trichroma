"""Deterministic detectors and photon sources for bitwise CUDA/Triton comparisons.

Every fixture is rebuilt identically in the CUDA and the Triton process; the
tape additionally records the inputs, so a mismatch is reported instead of
silently comparing different runs.

* ``synthetic``: a small detector exercising every original optical path:
  default, complex (thin film, transmissive and opaque), WLS, dichroic and
  angular surfaces, a two-component bulk re-emitting medium with Rayleigh
  scattering, a dielectric block (Fresnel), an analytic wire plane and eight
  channels.
* ``reflect3wires``, ``reflect3wires_vuv``, ``pixel_vuv``: the chroma-lar LAr
  detectors built through the public loader (``flatten=True``, cached Chroma
  BVH), with the synthetic original-model TPB coating of
  ``chroma-lar/benchmarks/original_chroma_cases.py`` for the VUV variants.
  Requires ``chroma_lar`` on ``PYTHONPATH``.
"""

import numpy as np

from chroma.event import Photons

GRID = np.arange(120, 801, 5, dtype=np.float32)


def _material(name, n=1.0, absorption=1.0e30, scattering=1.0e30):
    from chroma.geometry import Material

    m = Material(name)
    m.set("refractive_index", n)
    m.set("absorption_length", absorption)
    m.set("scattering_length", scattering)
    return m


def _surface(name, model=0, **properties):
    from chroma.geometry import Surface

    s = Surface(name, model=model)
    for key, value in properties.items():
        s.set(key, value)
    return s


def synthetic_detector():
    """Small detector covering every original surface model and bulk process."""
    from chroma.detector import Detector
    from chroma.geometry import Solid, DichroicProps, AngularProps
    from chroma.make import box
    from chroma.loader import create_geometry_from_obj

    w = GRID.astype(np.float64)
    medium = _material("synthetic-medium")
    medium.set("refractive_index", 1.30 + 0.012 / (w / 1000.0) ** 2, GRID)
    medium.set("scattering_length", 400.0 * (w / 400.0) ** 4, GRID)
    medium.set("absorption_length", 250.0 + 0.8 * w, GRID)
    t = np.arange(0.0, 1000.0, 0.05)
    for probability, factor, lower, upper, delay in ((0.8, 1.6, 400.0, 440.0, 6.0),
                                                     (0.55, 2.7, 440.0, 520.0, 18.0)):
        medium.comp_absorption_length.append(np.column_stack((w, (250.0 + 0.8 * w) * factor)))
        medium.comp_reemission_prob.append(np.column_stack((w, np.full(len(w), probability))))
        medium.comp_reemission_wvl_cdf.append(np.column_stack((w, np.clip((w - lower) / (upper - lower), 0, 1))))
        medium.comp_reemission_time_cdf.append(np.column_stack((t, np.clip(t / delay, 0, 1))))
    detector = Detector(medium)

    wall = _surface("synthetic-wall", absorb=0.35, reflect_diffuse=0.35, reflect_specular=0.3)
    detector.add_solid(Solid(box(400.0, 400.0, 400.0), medium, medium, surface=wall))

    glass = _material("synthetic-glass")
    glass.set("refractive_index", 1.50 + 0.02 / (w / 1000.0) ** 2, GRID)
    glass.set("absorption_length", 2000.0)
    detector.add_solid(Solid(box(40.0, 60.0, 20.0), glass, medium), displacement=[0.0, 0.0, -60.0])

    sensor = _material("synthetic-sensor", n=1.6, absorption=0.5)
    default = _surface("synthetic-default", detect=0.3, absorb=0.2, reflect_diffuse=0.25, reflect_specular=0.15)
    detector.add_pmt(Solid(box(30.0, 30.0, 30.0), sensor, medium, surface=default), displacement=[120.0, 0.0, 0.0])

    film = _surface("synthetic-complex", model=1, detect=0.6, reflect_diffuse=0.3)
    film.set("eta", 2.0 + 0.001 * (w - 400.0), GRID)
    film.set("k", 1.2 + 0.0005 * (w - 400.0), GRID)
    film.thickness = 25.0
    film.transmissive = 1
    detector.add_pmt(Solid(box(30.0, 30.0, 30.0), sensor, medium, surface=film), displacement=[-120.0, 0.0, 0.0])

    opaque = _surface("synthetic-complex-opaque", model=1, detect=0.4, reflect_diffuse=0.5)
    opaque.set("eta", 1.7)
    opaque.set("k", 2.0)
    opaque.thickness = 12.0
    opaque.transmissive = 0
    detector.add_pmt(Solid(box(30.0, 30.0, 30.0), sensor, medium, surface=opaque), displacement=[0.0, 120.0, 0.0])

    wls = _surface("synthetic-wls", model=2, reemit=0.85, reflect_specular=0.05, reflect_diffuse=0.05)
    wls.set("absorb", np.where(w < 250.0, 0.9, 0.05), GRID)
    wls.set("reemission_cdf", np.clip((w - 410.0) / 80.0, 0.0, 1.0), GRID)
    detector.add_pmt(Solid(box(30.0, 30.0, 30.0), sensor, medium, surface=wls), displacement=[0.0, -120.0, 0.0])

    angles = np.radians(np.linspace(0.0, 90.0, 7))
    reflect = [np.column_stack((w, np.full(len(w), 0.1 + 0.1 * i))) for i in range(len(angles))]
    transmit = [np.column_stack((w, np.clip(0.8 - 0.12 * i + 0.0002 * (w - 400.0), 0.0, 0.9)))
                for i in range(len(angles))]
    dichroic = _surface("synthetic-dichroic", model=3)
    dichroic.dichroic_props = DichroicProps(angles, reflect, transmit)
    detector.add_pmt(Solid(box(30.0, 30.0, 30.0), sensor, medium, surface=dichroic), displacement=[0.0, 0.0, 120.0])

    aangles = np.radians(np.linspace(0.0, 90.0, 10))
    angular = _surface("synthetic-angular", model=4)
    angular.angular_props = AngularProps(aangles, np.linspace(0.7, 0.05, 10), np.linspace(0.1, 0.5, 10),
                                         np.linspace(0.1, 0.3, 10))
    detector.add_pmt(Solid(box(30.0, 30.0, 30.0), sensor, medium, surface=angular), displacement=[85.0, 85.0, 0.0])

    detect_all = _surface("synthetic-detector", detect=0.9, absorb=0.1)
    detector.add_pmt(Solid(box(30.0, 30.0, 30.0), sensor, medium, surface=detect_all), displacement=[-85.0, -85.0, 0.0])
    detector.add_pmt(Solid(box(30.0, 30.0, 30.0), sensor, medium, surface=default), displacement=[85.0, -85.0, 60.0])

    metal = _material("synthetic-wire-metal", n=2.5, absorption=0.0001)
    reflector = _surface("synthetic-wire-surface", absorb=0.2, reflect_specular=0.6, reflect_diffuse=0.2)
    # Keep the wire materials and surface in the solids' unique lists, as the
    # original GPUGeometry requires (its second wire block validates them).
    detector.add_solid(Solid(box(4.0, 4.0, 4.0), metal, medium, surface=reflector), displacement=[150.0, 150.0, 150.0])
    detector.wireplanes = [dict(origin=[0.0, 0.0, 60.0], u=[0.0, 1.0, 0.0], v=[0.8, 0.0, 0.6], pitch=3.0,
                                radius=0.25, umin=-80.0, umax=80.0, vmin=-70.0, vmax=70.0, v0=0.17,
                                surface=reflector, material_inner=metal, material_outer=medium,
                                color=0xFFFFFFFF)]
    # Consistent x/y CDF lengths (Detector.set_*_dist builds y one short).
    tx = np.linspace(-6.0, 6.0, 49)
    ty = np.cumsum(np.exp(-0.5 * (tx / 1.2) ** 2))
    detector.time_cdf = (tx, (ty - ty[0]) / (ty[-1] - ty[0]))
    qx = np.linspace(0.5, 1.5, 41)
    qy = np.cumsum(np.exp(-0.5 * ((qx - 1.0) / 0.1) ** 2))
    detector.charge_cdf = (qx, (qy - qy[0]) / (qy[-1] - qy[0]))
    return create_geometry_from_obj(detector)


def synthetic_photons(count, seed, wavelengths=(150.0, 600.0)):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-40.0, 40.0, (count, 3))
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    pol = np.cross(direction, rng.normal(size=(count, 3)))
    pol /= np.linalg.norm(pol, axis=1)[:, None]
    wl = rng.uniform(wavelengths[0], wavelengths[1], count)
    return Photons(pos, direction, pol, wl, t=rng.uniform(0.0, 20.0, count))


def lar_detector(name):
    """reflect3wires / reflect3wires_vuv / pixel_vuv through chroma-lar's loader."""
    from chroma_lar.geometry.config_loader import build_detector_from_config

    pixel = name.startswith("pixel")
    vuv = name.endswith("_vuv")
    options = {}
    if not pixel:
        options["analytic_wires"] = True
    if vuv:
        grid = np.arange(120, 501, dtype=np.float32)
        coat = _surface("legacy-fixture-tpb", model=2, reemit=0.85)
        coat.set("absorb", np.where(grid < 200.0, 1.0, 0.0), grid)
        coat.set("reemission_cdf", np.clip((grid - 410.0) / 80.0, 0.0, 1.0), grid)
        options["pmt_coating_surface"] = coat
    config = "detector_config_pixel" if pixel else "detector_config_reflect_reflect3wires"
    return build_detector_from_config(config, **options)


def lar_photons(name, count, seed):
    """The detector_case source of original_chroma_cases.py (two x = +-1000 mm points)."""
    vuv = name.endswith("_vuv")
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-15.0, 15.0, (count, 3))
    positions[:, 0] += np.where(np.arange(count) % 2, 1000.0, -1000.0)
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    polarization = np.cross(direction, rng.normal(size=(count, 3)))
    polarization /= np.linalg.norm(polarization, axis=1)[:, None]
    wavelengths = rng.uniform(120.0, 140.0, count) if vuv else rng.uniform(390.0, 495.0, count)
    return Photons(positions, direction, polarization, wavelengths, t=rng.uniform(0.0, 1500.0, count))


def split_events(photons, sizes):
    out, start = [], 0
    for size in sizes:
        out.append(photons[start:start + size])
        start += size
    assert start == len(photons)
    return out


# A run is (simulation kwargs, simulate kwargs, list of event photon sizes, source seed).
RUNS = {
    "synthetic": [
        dict(name="multi_launch", sim=dict(seed=20260925), simulate=dict(
            keep_photons_end=True, run_daq=True, max_steps=400, photons_per_batch=100000),
             events=[20000] * 10, seed=11),
        dict(name="weights", sim=dict(seed=7), simulate=dict(
            keep_photons_end=True, run_daq=True, max_steps=200, use_weights=True, photons_per_batch=50000),
             events=[10000, 15000, 25000], seed=12),
        dict(name="tracking", sim=dict(seed=99, photon_tracking=True), simulate=dict(
            keep_photons_end=True, run_daq=True, max_steps=60, photons_per_batch=4000),
             events=[1000, 1500, 1500], seed=13),
        dict(name="packed", sim=dict(seed=5, use_packed=True), simulate=dict(
            keep_photons_end=True, run_daq=True, max_steps=300, photons_per_batch=100000),
             events=[40000, 40000], seed=14),
        dict(name="small_threads", sim=dict(seed=3, nthreads_per_block=64, max_blocks=16), simulate=dict(
            keep_photons_end=True, run_daq=True, max_steps=300, photons_per_batch=30000),
             events=[3000] * 10, seed=15),
    ],
    "reflect3wires": [
        dict(name="visible", sim=dict(seed=1981), simulate=dict(
            keep_photons_end=True, run_daq=True, max_steps=512, photons_per_batch=200000),
             events=[50000] * 2, seed=1981),
    ],
    "reflect3wires_vuv": [
        dict(name="vuv", sim=dict(seed=1982), simulate=dict(
            keep_photons_end=True, run_daq=True, max_steps=512, photons_per_batch=200000),
             events=[50000] * 2, seed=1982),
    ],
    "pixel_vuv": [
        dict(name="vuv", sim=dict(seed=1983), simulate=dict(
            keep_photons_end=True, run_daq=True, max_steps=512, photons_per_batch=200000),
             events=[50000] * 2, seed=1983),
    ],
}


def build_detector(fixture):
    if fixture == "synthetic":
        return synthetic_detector()
    return lar_detector(fixture)


def make_events(fixture, run, scale=1.0):
    """List of chroma.event.Photons (one per event) for a run description."""
    sizes = [max(1, int(round(s * scale))) for s in run["events"]]
    total = int(sum(sizes))
    if fixture == "synthetic":
        photons = synthetic_photons(total, run["seed"])
    else:
        photons = lar_photons(fixture, total, run["seed"])
    return split_events(photons, sizes)
