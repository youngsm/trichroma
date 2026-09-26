"""The Simulation contract of the Triton backend, on a small detector that
exercises the production physics: dispersion, Rayleigh scattering, bulk
re-emission, default and WLS surfaces, an analytic wire plane and the DAQ.

Photon ids (the keys of the counter-based random numbers) are assigned in
input order, so every Event output must be the same bit for bit however the
input is batched, pipelined or supplied (host arrays or CUDA tensors).
"""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from chroma.event import Event, Photons

pytestmark = pytest.mark.gpu

GRID = np.arange(120, 801, 5, dtype=np.float32)
SIZES = [3000, 0, 1500, 6000, 1, 2500]


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


def _detector():
    from chroma.detector import Detector
    from chroma.geometry import Solid
    from chroma.loader import create_geometry_from_obj
    from chroma.make import box

    w = GRID.astype(np.float64)
    medium = _material("test-medium")
    medium.set("refractive_index", 1.30 + 0.012 / (w / 1000.0) ** 2, GRID)
    medium.set("scattering_length", 400.0 * (w / 400.0) ** 4, GRID)
    medium.set("absorption_length", 250.0 + 0.8 * w, GRID)
    t = np.arange(0.0, 1000.0, 0.05)
    medium.comp_absorption_length.append(np.column_stack((w, (250.0 + 0.8 * w) * 1.6)))
    medium.comp_reemission_prob.append(np.column_stack((w, np.full(len(w), 0.8))))
    medium.comp_reemission_wvl_cdf.append(np.column_stack((w, np.clip((w - 400.0) / 40.0, 0, 1))))
    medium.comp_reemission_time_cdf.append(np.column_stack((t, np.clip(t / 6.0, 0, 1))))
    detector = Detector(medium)

    wall = _surface("test-wall", absorb=0.35, reflect_diffuse=0.35, reflect_specular=0.3)
    detector.add_solid(Solid(box(400.0, 400.0, 400.0), medium, medium, surface=wall))
    glass = _material("test-glass")
    glass.set("refractive_index", 1.50 + 0.02 / (w / 1000.0) ** 2, GRID)
    glass.set("absorption_length", 2000.0)
    detector.add_solid(Solid(box(40.0, 60.0, 20.0), glass, medium), displacement=[0.0, 0.0, -60.0])

    sensor = _material("test-sensor", n=1.6, absorption=0.5)
    default = _surface("test-default", detect=0.3, absorb=0.2, reflect_diffuse=0.25, reflect_specular=0.15)
    wls = _surface("test-wls", model=2, reemit=0.85, reflect_specular=0.05, reflect_diffuse=0.05)
    wls.set("absorb", np.where(w < 250.0, 0.9, 0.05), GRID)
    wls.set("reemission_cdf", np.clip((w - 410.0) / 80.0, 0.0, 1.0), GRID)
    detect_all = _surface("test-detector", detect=0.9, absorb=0.1)
    for surface, position in ((default, [120.0, 0.0, 0.0]), (wls, [0.0, -120.0, 0.0]),
                              (detect_all, [-85.0, -85.0, 0.0]), (default, [85.0, -85.0, 60.0]),
                              (detect_all, [0.0, 120.0, 0.0])):
        detector.add_pmt(Solid(box(30.0, 30.0, 30.0), sensor, medium, surface=surface), displacement=position)

    metal = _material("test-wire-metal", n=2.5, absorption=0.0001)
    reflector = _surface("test-wire-surface", absorb=0.2, reflect_specular=0.6, reflect_diffuse=0.2)
    detector.add_solid(Solid(box(4.0, 4.0, 4.0), metal, medium, surface=reflector), displacement=[150.0, 150.0, 150.0])
    detector.wireplanes = [dict(origin=[0.0, 0.0, 60.0], u=[0.0, 1.0, 0.0], v=[0.8, 0.0, 0.6], pitch=3.0,
                                radius=0.25, umin=-80.0, umax=80.0, vmin=-70.0, vmax=70.0, v0=0.17,
                                surface=reflector, material_inner=metal, material_outer=medium,
                                color=0xFFFFFFFF)]
    tx = np.linspace(-6.0, 6.0, 49)
    ty = np.cumsum(np.exp(-0.5 * (tx / 1.2) ** 2))
    detector.time_cdf = (tx, (ty - ty[0]) / (ty[-1] - ty[0]))
    qx = np.linspace(0.5, 1.5, 41)
    qy = np.cumsum(np.exp(-0.5 * ((qx - 1.0) / 0.1) ** 2))
    detector.charge_cdf = (qx, (qy - qy[0]) / (qy[-1] - qy[0]))
    return create_geometry_from_obj(detector)


def _photons(count, seed):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    pol = np.cross(direction, rng.normal(size=(count, 3)))
    pol /= np.linalg.norm(pol, axis=1)[:, None]
    return Photons(rng.uniform(-40.0, 40.0, (count, 3)), direction, pol, rng.uniform(150.0, 600.0, count),
                   t=rng.uniform(0.0, 20.0, count))


@pytest.fixture(scope="module")
def detector():
    return _detector()


@pytest.fixture(scope="module")
def sources():
    return [_photons(n, 100 + i) for i, n in enumerate(SIZES)]


def _host_events(sources):
    return [Event(photons_beg=Photons(p.pos, p.dir, p.pol, p.wavelengths, p.t)) for p in sources]


def _device_events(sources):
    import torch

    def cuda(array, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(array), device="cuda").to(dtype)

    events = []
    for p in sources:
        n = len(p)
        events.append(Event(photons_beg=SimpleNamespace(
            pos=cuda(p.pos), dir=cuda(p.dir), pol=cuda(p.pol), wavelengths=cuda(p.wavelengths), t=cuda(p.t),
            flags=torch.zeros(n, dtype=torch.int32, device="cuda"),
            evidx=torch.zeros(n, dtype=torch.int32, device="cuda"), true_nphotons=n)))
    return events


FIELDS = ("pos", "dir", "pol", "wavelengths", "t", "last_hit_triangles", "flags", "weights", "evidx", "channel")


def _digest(ev, fields=FIELDS):
    h = hashlib.sha256()
    for photons in (ev.photons_end, ev.flat_hits):
        for name in fields:
            a = np.ascontiguousarray(getattr(photons, name))
            h.update(name.encode() + a.dtype.str.encode() + a.tobytes())
    for channel in sorted(ev.hits):
        h.update(str(channel).encode() + np.ascontiguousarray(ev.hits[channel].t).tobytes())
    for a in (ev.channels.hit, ev.channels.t, ev.channels.q, ev.channels.flags):
        h.update(a.dtype.str.encode() + np.ascontiguousarray(a).tobytes())
    return ev.nphotons, len(ev.flat_hits), h.hexdigest()


def _run(detector, events, seed=4242, photons_per_batch=5000, use_weights=False, fields=FIELDS):
    from trichroma.simulation import Simulation

    sim = Simulation(detector, seed=seed)
    return [_digest(ev, fields) for ev in sim.simulate(events, keep_photons_end=True, keep_hits=True,
                                                       keep_flat_hits=True, run_daq=True, max_steps=1000,
                                                       use_weights=use_weights, photons_per_batch=photons_per_batch)]


@pytest.mark.parametrize("use_weights", [False, True])
def test_pipelined_equals_sequential(detector, sources, monkeypatch, use_weights):
    monkeypatch.setenv("CHROMA_TRITON_PIPELINE", "0")
    sequential = _run(detector, _host_events(sources), use_weights=use_weights)
    monkeypatch.setenv("CHROMA_TRITON_PIPELINE", "1")
    pipelined = _run(detector, _host_events(sources), use_weights=use_weights)
    assert pipelined == sequential
    assert sum(hits for _, hits, _ in pipelined) > 0


def test_results_do_not_depend_on_batching(detector, sources):
    # evidx is the event's index within its batch (as in Chroma), so it is
    # the one field that depends on the batching.
    fields = tuple(f for f in FIELDS if f != "evidx")
    one_per_batch = _run(detector, _host_events(sources), photons_per_batch=1, fields=fields)
    all_in_one = _run(detector, _host_events(sources), photons_per_batch=10**9, fields=fields)
    assert one_per_batch == all_in_one


def test_device_input_matches_host_input(detector, sources):
    assert _run(detector, _device_events(sources)) == _run(detector, _host_events(sources))


def test_seed_above_int32_is_deterministic(detector, sources):
    big = _run(detector, _host_events(sources), seed=3_000_000_000)
    assert big == _run(detector, _host_events(sources), seed=3_000_000_000)
    assert big != _run(detector, _host_events(sources), seed=5)


def test_trichroma_sources_simulate(detector):
    from trichroma import sources
    from trichroma.simulation import Simulation

    photons = sources.photon_bombs(2000, np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]]), voxel_size=20,
                                   wavelength=(300.0, 500.0))
    sim = Simulation(detector, seed=9)
    events = list(sim.simulate([Event(photons_beg=photons[:2000]), Event(photons_beg=photons[2000:])],
                               keep_flat_hits=True, max_steps=1000))
    assert [ev.nphotons for ev in events] == [2000, 2000]
    assert all(len(ev.flat_hits) > 0 for ev in events)
