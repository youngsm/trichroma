"""Analytic wires against an independent float64 Monte Carlo.

A plane of wires at x = 2160 mm (metre-scale coordinates, as in the LAr
detectors), 75 um radius and 3 mm pitch, with a surface that absorbs 20% and
reflects 80% specularly: nothing is transmitted, so no photon can ever be
inside a wire. A mirror 5 mm behind the plane makes photons bounce between
it and the wires; every other wall detects.

At these coordinates an FP32 reflection point can land just inside the
cylinder. CUDA Chroma then takes the exit for a hit from inside, puts the
photon in the steel and absorbs it; the production engine skips the wire a
photon has just left outward (a ray leaving a convex cylinder cannot meet it
again), so its outcomes must match the float64 simulation.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

X_WIRES, PITCH, RADIUS, KMAX = 2160.0, 3.0, 0.075, 100
X_MIN, X_MIRROR, HALF = 1900.0, 2165.0, 400.0
P_ABSORB = 0.2
MAX_STEPS = 1000
N = 300_000


def _source(n, seed):
    rng = np.random.default_rng(seed)
    pos = np.array([2100.0, 0.0, 0.0]) + rng.uniform(-10.0, 10.0, (n, 3))
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    pol = np.cross(d, rng.normal(size=(n, 3)))
    pol /= np.linalg.norm(pol, axis=1)[:, None]
    return pos, d, pol


def _reference(n, seed):
    """Float64 tracking of the same photons: counts of (absorbed at a wire,
    detected on the x = X_MIN wall, detected on a side wall)."""
    pos, d, _ = _source(n, seed)
    rng = np.random.default_rng(seed + 1)
    axes = np.arange(-KMAX, KMAX + 1, dtype=np.float64) * PITCH
    alive = np.ones(n, bool)
    last_wire = np.full(n, -1)
    outcome = np.full(n, 3)
    for _ in range(MAX_STEPS):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        for chunk in np.array_split(idx, max(1, idx.size // 20000)):
            p, v = pos[chunk], d[chunk]
            with np.errstate(divide="ignore", invalid="ignore"):
                walls = [np.where(v[:, 0] > 0, (X_MIRROR - p[:, 0]) / v[:, 0], (X_MIN - p[:, 0]) / v[:, 0]),
                         np.where(v[:, 1] > 0, (HALF - p[:, 1]) / v[:, 1], (-HALF - p[:, 1]) / v[:, 1]),
                         np.where(v[:, 2] > 0, (HALF - p[:, 2]) / v[:, 2], (-HALF - p[:, 2]) / v[:, 2])]
            t_wall = np.min([np.where(np.isfinite(w), w, np.inf) for w in walls], axis=0)
            ox, oy = p[:, 0:1] - X_WIRES, p[:, 1:2] - axes[None, :]
            a = v[:, 0:1] ** 2 + v[:, 1:2] ** 2
            b = ox * v[:, 0:1] + oy * v[:, 1:2]
            c = ox ** 2 + oy ** 2 - RADIUS ** 2
            disc = b * b - a * c
            with np.errstate(invalid="ignore", divide="ignore"):
                t = (-b - np.sqrt(disc)) / a
            ok = (disc > 0) & (c > 0) & (t > 0) & (np.arange(axes.size)[None, :] != last_wire[chunk][:, None])
            t = np.where(ok, t, np.inf)
            k = np.argmin(t, axis=1)
            t_wire = t[np.arange(len(chunk)), k]
            wire_first = t_wire < t_wall
            q = p + v * np.where(wire_first, t_wire, t_wall)[:, None]
            w = chunk[wire_first]
            absorbed = rng.random(w.size) < P_ABSORB
            outcome[w[absorbed]] = 0
            alive[w[absorbed]] = False
            r, qr, kr = w[~absorbed], q[wire_first][~absorbed], k[wire_first][~absorbed]
            normal = np.stack([qr[:, 0] - X_WIRES, qr[:, 1] - axes[kr], np.zeros(r.size)], 1)
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            d[r] -= 2 * (d[r] * normal).sum(1)[:, None] * normal
            pos[r], last_wire[r] = qr, kr
            wl, qw = chunk[~wire_first], q[~wire_first]
            mirror = np.isclose(qw[:, 0], X_MIRROR, atol=1e-9, rtol=0) & (d[wl][:, 0] > 0)
            m = wl[mirror]
            d[m, 0] = -d[m, 0]
            pos[m], last_wire[m] = qw[mirror], -1
            other = wl[~mirror]
            x_wall = np.isclose(qw[~mirror][:, 0], X_MIN, atol=1e-9, rtol=0)
            outcome[other[x_wall]] = 1
            outcome[other[~x_wall]] = 2
            alive[other] = False
    return np.bincount(outcome, minlength=4)[:3]


def _engine(n, seed):
    """The same photons through trichroma.simulation: (counts, bulk absorbed)."""
    from chroma.detector import Detector
    from chroma.event import Photons
    from chroma.geometry import Material, Solid, Surface
    from chroma.loader import create_geometry_from_obj
    from chroma.make import box
    from trichroma.simulation import Simulation

    def material(name, index, absorption):
        m = Material(name)
        m.set("refractive_index", index)
        m.set("absorption_length", absorption)
        m.set("scattering_length", 1e30)
        return m

    def surface(name, **properties):
        s = Surface(name)
        for key, value in properties.items():
            s.set(key, value)
        return s

    argon, steel = material("argon", 1.38, 1e30), material("steel", 1.07, 0.0)
    detect = surface("detect", detect=1.0)
    mirror = surface("mirror", reflect_specular=1.0)
    polished = surface("polished", absorb=P_ABSORB, reflect_specular=1.0 - P_ABSORB)
    detector = Detector(argon)
    sx, sy, cx = X_MIRROR - X_MIN, 2 * HALF, (X_MIRROR + X_MIN) / 2
    for size, centre, surf in [((1.0, sy + 2, sy + 2), (X_MIN - 0.5, 0, 0), detect),
                               ((1.0, sy + 2, sy + 2), (X_MIRROR + 0.5, 0, 0), mirror),
                               ((sx + 2, 1.0, sy + 2), (cx, -HALF - 0.5, 0), detect),
                               ((sx + 2, 1.0, sy + 2), (cx, HALF + 0.5, 0), detect),
                               ((sx + 2, sy + 2, 1.0), (cx, 0, -HALF - 0.5), detect),
                               ((sx + 2, sy + 2, 1.0), (cx, 0, HALF + 0.5), detect)]:
        solid = Solid(box(*size), steel, argon, surface=surf)
        if surf is detect:
            detector.add_pmt(solid, displacement=centre)
        else:
            detector.add_solid(solid, displacement=centre)
    # outside the argon volume: registers the wire materials and surface
    detector.add_solid(Solid(box(1.0, 1.0, 1.0), steel, argon, surface=polished), displacement=[X_MIN - 50.0, 0, 0])
    detector.wireplanes = [dict(origin=[X_WIRES, 0.0, 0.0], u=[0.0, 0.0, 1.0], v=[0.0, 1.0, 0.0], pitch=PITCH,
                                radius=RADIUS, umin=-HALF, umax=HALF, vmin=-KMAX * PITCH, vmax=KMAX * PITCH, v0=0.0,
                                surface=polished, material_inner=steel, material_outer=argon, color=0xFFFFFFFF)]
    pos, d, pol = _source(n, seed)
    sim = Simulation(create_geometry_from_obj(detector), seed=seed)
    ev = next(sim.simulate([Photons(pos, d, pol, np.full(n, 450.0))], keep_photons_end=True, keep_flat_hits=False,
                           keep_hits=False, max_steps=MAX_STEPS, photons_per_batch=n))
    flags, last, final = ev.photons_end.flags, ev.photons_end.last_hit_triangles, ev.photons_end.pos
    detected = (flags & 4) != 0
    x_wall = detected & (np.abs(final[:, 0] - X_MIN) < 0.01)
    counts = np.array([(((flags & 8) != 0) & (last == -2)).sum(), x_wall.sum(), (detected & ~x_wall).sum()])
    return counts, int(((flags & 2) != 0).sum())


def test_no_photon_enters_a_wire_and_outcomes_match_float64():
    engine, bulk = _engine(N, 7)
    reference = _reference(N, 7)
    assert bulk == 0
    # Same source photons: only the wire interactions are random. Five
    # standard errors of the difference of two binomial fractions.
    p = (engine + reference) / (2 * N)
    sigma = np.sqrt(2 * p * (1 - p) / N)
    assert np.all(np.abs(engine - reference) / N < 5 * sigma), (engine / N, reference / N)


def test_bug_compatible_wires_keep_chroma_losses(monkeypatch):
    monkeypatch.setenv("CHROMA_TRITON_LEGACY_WIRES", "1")
    _, bulk = _engine(N // 3, 7)
    assert bulk > 0  # CUDA Chroma's defect, reproduced on request
