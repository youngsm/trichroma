"""trichroma.utils: bitwise and statistical comparison of simulation results."""

import numpy as np

from chroma.event import Event, Photons
from trichroma.utils import compare_events, compare_photons, compare_statistics


def _photons(n, seed=1):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    p = Photons(pos=rng.normal(size=(n, 3)).astype(np.float32), dir=d.astype(np.float32),
                pol=np.tile(np.float32([1, 0, 0]), (n, 1)), t=rng.random(n).astype(np.float32),
                wavelengths=np.full(n, 128.0, np.float32))
    p.flags[:] = rng.choice([4, 8, 2, 20], n).astype(np.uint32)
    p.last_hit_triangles[:] = rng.integers(0, 100, n).astype(np.int32)
    return p


def _copy(p):
    return Photons(pos=p.pos.copy(), dir=p.dir.copy(), pol=p.pol.copy(), t=p.t.copy(),
                   wavelengths=p.wavelengths.copy(), last_hit_triangles=p.last_hit_triangles.copy(),
                   flags=p.flags.copy(), weights=p.weights.copy())


def test_identical_photons_compare_equal():
    a = _photons(1000)
    result = compare_photons(a, _copy(a))
    assert result
    assert "bitwise identical" in str(result)


def test_one_ulp_and_signed_zero_are_differences():
    a = _photons(1000)
    b = _copy(a)
    b.t[7] = np.nextafter(b.t[7], np.float32(np.inf))
    b.pos[3, 1] = np.float32(0.0)
    a.pos[3, 1] = np.float32(-0.0)
    result = compare_photons(a, b)
    assert not result
    assert result.details["t"] == {"differ": 1, "first": 7, "max_ulp": 1}
    assert result.details["pos"]["differ"] == 1 and result.details["pos"]["first"] == 3
    assert result.details["dir"]["differ"] == 0


def test_equal_nans_are_equal():
    a = _photons(10)
    a.t[2] = np.nan
    assert compare_photons(a, _copy(a))


def test_events_compare_hits_as_multisets():
    a, b = Event(), Event()
    a.photons_end, b.photons_end = _photons(500), _photons(500)
    b.photons_end = _copy(a.photons_end)
    hits = _photons(40, seed=2)
    hits.channel = np.arange(40, dtype=np.int32) % 7
    a.flat_hits = hits
    order = np.random.default_rng(3).permutation(40)
    shuffled = Photons(pos=hits.pos[order], dir=hits.dir[order], pol=hits.pol[order], t=hits.t[order],
                       wavelengths=hits.wavelengths[order], last_hit_triangles=hits.last_hit_triangles[order],
                       flags=hits.flags[order], weights=hits.weights[order])
    shuffled.channel = hits.channel[order]
    b.flat_hits = shuffled
    assert compare_events(a, b)
    b.flat_hits.t[0] += np.float32(1.0)
    assert not compare_events(a, b)


def test_statistics_accept_the_same_distribution_and_reject_a_shift():
    rng = np.random.default_rng(5)

    def run(n, p_detect, t_shift=0.0):
        flags = np.where(rng.random(n) < p_detect, 4, 8).astype(np.uint32)
        hits = int((flags == 4).sum())
        return {"flags": flags, "t": rng.exponential(10.0, n) + t_shift, "last": np.zeros(n, np.int32),
                "hit_channel": rng.integers(0, 20, hits), "hit_t": rng.exponential(10.0, hits) + t_shift,
                "hit_wl": np.full(hits, 128.0), "hit_w": np.ones(hits)}

    assert compare_statistics(run(200_000, 0.30), run(200_000, 0.30))
    result = compare_statistics(run(200_000, 0.30), run(200_000, 0.33))
    assert not result and "surface_detect" in result.details["failures"]
    assert not compare_statistics(run(200_000, 0.30), run(200_000, 0.30, t_shift=1.0))
