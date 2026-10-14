"""Compare two simulation results.

* :func:`compare_photons` and :func:`compare_events`: bitwise, word for word.
  Use them where the two runs must be identical: TriChroma's exact mode
  (``CHROMA_TRITON_TAPE=replay:<dir>``) against the CUDA run it replays, or
  TriChroma against itself on the same GPU type (``CHROMA_TRITON=strict``
  against the default, one version against another).
* :func:`compare_statistics`: for runs with different random numbers, such as
  TriChroma's production engine against CUDA Chroma. It tests that the two
  agree within statistical errors: history-bit fractions, hits per channel,
  weighted efficiency per channel, and hit-time, hit-wavelength and final-time
  distributions.

Each returns a result that is true when the runs agree and prints as a short
report::

    from trichroma.utils import compare_events
    result = compare_events(events_a, events_b)
    print(result)
    assert result
"""

import numpy as np

PHOTON_FIELDS = ("pos", "dir", "pol", "wavelengths", "t", "last_hit_triangles", "flags", "weights")

# History bits of photon flags (chroma/cuda/photon.h).
FLAG_BITS = {"no_hit": 1, "bulk_absorb": 2, "surface_detect": 4, "surface_absorb": 8, "rayleigh_scatter": 16,
             "reflect_diffuse": 32, "reflect_specular": 64, "surface_reemit": 128, "surface_transmit": 256,
             "bulk_reemit": 512}
TERMINAL = 1 | 2 | 4 | 8


def _array(x):
    """A numpy view of a numpy array, torch tensor or GPU array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    elif hasattr(x, "get") and not isinstance(x, np.ndarray):
        x = x.get()
    return np.asarray(x)


def _bytes(x):
    """The bytes of an array, one row per photon."""
    x = np.ascontiguousarray(x)
    return x.reshape(len(x), -1).view(np.uint8)


def _ulps(a, b):
    """ulp distance of two float32 arrays (NaN against a number counts as the largest distance)."""
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-(2 ** 31)) - ia, ia)
    ib = np.where(ib < 0, np.int64(-(2 ** 31)) - ib, ib)
    d = np.abs(ia - ib)
    return np.where(np.isnan(a) != np.isnan(b), np.iinfo(np.int64).max, d)


class Comparison:
    """Result of a comparison: true when the two runs agree; ``str()`` is a report."""

    def __init__(self, equal, lines, details):
        self.equal = bool(equal)
        self.lines = lines
        self.details = details

    def __bool__(self):
        return self.equal

    def __str__(self):
        return "\n".join(self.lines)

    __repr__ = __str__


def compare_photons(a, b, fields=PHOTON_FIELDS):
    """Bitwise comparison of two photon sets (``chroma.event.Photons`` or
    TriChroma's device photons), field by field, in photon order.

    For each field it reports how many photons differ, the first one, and for
    float32 fields the largest difference in ulp. Comparison is by bytes:
    NaNs with the same bits are equal, and 0.0 and -0.0 differ."""
    lines, details, equal = [], {}, True
    na, nb = len(_array(a.pos)), len(_array(b.pos))
    if na != nb:
        return Comparison(False, ["photon counts differ: %d and %d" % (na, nb)], {"count": (na, nb)})
    for name in fields:
        xa, xb = getattr(a, name, None), getattr(b, name, None)
        if xa is None or xb is None:
            continue
        xa, xb = _array(xa), _array(xb)
        if xa.dtype != xb.dtype or xa.shape != xb.shape:
            details[name] = {"differ": na, "first": 0, "types": (str(xa.dtype), str(xb.dtype))}
            lines.append("%-19s types or shapes differ: %s %s and %s %s" % (name, xa.dtype, xa.shape, xb.dtype,
                                                                           xb.shape))
            equal = False
            continue
        rows = np.flatnonzero(np.any(_bytes(xa) != _bytes(xb), axis=1))
        entry = {"differ": int(len(rows)), "first": int(rows[0]) if len(rows) else None}
        if len(rows) and xa.dtype == np.float32:
            fa = np.ascontiguousarray(xa).reshape(len(xa), -1)[rows]
            fb = np.ascontiguousarray(xb).reshape(len(xb), -1)[rows]
            entry["max_ulp"] = int(_ulps(fa, fb).max())
        details[name] = entry
        equal &= not len(rows)
        if len(rows):
            ulp = " (max %d ulp)" % entry["max_ulp"] if "max_ulp" in entry else ""
            lines.append("%-19s %d of %d photons differ, first %d%s" % (name, len(rows), na, rows[0], ulp))
    lines.insert(0, "photons: %d, %s" % (na, "bitwise identical" if equal else "DIFFERENT"))
    return Comparison(equal, lines, details)


def compare_events(a, b):
    """Bitwise comparison of two runs' events (one ``Event`` or a list of them):
    final photons, flat hits (as multisets: CUDA Chroma returns them in
    arbitrary order) and DAQ channels, whichever both runs kept."""
    from trichroma.tape import verify

    events_a = list(a) if isinstance(a, (list, tuple)) else [a]
    events_b = list(b) if isinstance(b, (list, tuple)) else [b]
    if len(events_a) != len(events_b):
        return Comparison(False, ["event counts differ: %d and %d" % (len(events_a), len(events_b))], {})
    arrays_a = verify._event_arrays(events_a, has_channels=True)
    arrays_b = verify._event_arrays(events_b, has_channels=True)
    shared = {k for k in arrays_a if k in arrays_b}
    ok, report = verify.compare_outputs({k: arrays_a[k] for k in shared}, {k: arrays_b[k] for k in shared})
    lines = ["events: %d, %s" % (len(events_a), "bitwise identical" if ok else "DIFFERENT")]
    for key, value in report.items():
        state = "identical" if value.get("equal") else "differ, first difference %s" % (value.get("first_difference"),)
        lines.append("%-15s %s" % (key, state))
    return Comparison(ok, lines, report)


def two_proportion_z(k1, n1, k2, n2):
    """z statistic of the difference between two proportions k1/n1 and k2/n2."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-300))
    return (p1 - p2) / se if (k1 + k2) > 0 else 0.0


def ks_test(a, b, alpha=1e-6):
    """Two-sample Kolmogorov-Smirnov distance and its DKW bound at ``alpha``
    (None for fewer than 20 values on either side)."""
    if len(a) < 20 or len(b) < 20:
        return None
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    distance = float(np.max(np.abs(fa - fb)))
    bound = float(np.sqrt(np.log(2 / alpha) / 2) * (np.sqrt(1 / len(a)) + np.sqrt(1 / len(b))))
    return {"distance": distance, "bound": bound, "pass": distance <= bound}


def event_statistics_arrays(events):
    """The arrays :func:`compare_statistics` needs, from one ``Event`` or a list:
    final flags, times and last-hit triangles, flat hits (channel, time,
    wavelength, weight) and DAQ hits and charges when kept."""
    events = list(events) if isinstance(events, (list, tuple)) else [events]
    out = {}
    end = [ev.photons_end for ev in events if getattr(ev, "photons_end", None) is not None]
    if end:
        out["flags"] = np.concatenate([_array(p.flags) for p in end])
        out["t"] = np.concatenate([_array(p.t) for p in end])
        out["last"] = np.concatenate([_array(p.last_hit_triangles) for p in end])
    hits = [ev.flat_hits for ev in events if getattr(ev, "flat_hits", None) is not None]
    if hits:
        out["hit_channel"] = np.concatenate([_array(h.channel) for h in hits]).astype(np.int64)
        out["hit_t"] = np.concatenate([_array(h.t) for h in hits])
        out["hit_wl"] = np.concatenate([_array(h.wavelengths) for h in hits])
        out["hit_w"] = np.concatenate([_array(h.weights) for h in hits])
    chans = [ev.channels for ev in events if getattr(ev, "channels", None) is not None]
    if chans:
        out["daq_hit"] = np.concatenate([_array(c.hit) for c in chans])
        out["daq_q"] = np.concatenate([_array(c.q) for c in chans])
    return out


def compare_statistics(a, b, nphotons=None, max_sigma=6.0, alpha=1e-6):
    """Statistical agreement of two runs of the same photons with different
    random numbers. ``a``/``b`` are events (one or a list) or the arrays of
    :func:`event_statistics_arrays`; ``nphotons`` (a pair, or one number for
    both) is the number of photons simulated, needed when the runs kept no
    final photons.

    Tests: every history-bit fraction and the unfinished fraction
    (two-proportion z), hits per channel and weighted efficiency per channel
    for channels with at least 25 hits (z), and the hit-time, hit-wavelength
    and final-time distributions (KS distance against the DKW bound at
    ``alpha``). Fails where |z| > ``max_sigma`` or a KS distance exceeds its
    bound."""
    a = a if isinstance(a, dict) else event_statistics_arrays(a)
    b = b if isinstance(b, dict) else event_statistics_arrays(b)
    if nphotons is None:
        if "flags" not in a or "flags" not in b:
            raise ValueError("the runs kept no final photons: pass nphotons")
        n1, n2 = len(a["flags"]), len(b["flags"])
    else:
        n1, n2 = (nphotons, nphotons) if np.isscalar(nphotons) else nphotons
    lines, failures, details = [], [], {"photons": (n1, n2)}

    def fraction(name, k1, k2):
        z = two_proportion_z(k1, n1, k2, n2)
        details.setdefault("fractions", {})[name] = {"a": k1 / n1, "b": k2 / n2, "z": z}
        lines.append("%-19s %10.6f %10.6f  z=%7.2f%s" % (name, k1 / n1, k2 / n2, z,
                                                          "  FAIL" if abs(z) > max_sigma else ""))
        if abs(z) > max_sigma:
            failures.append(name)

    if "flags" in a and "flags" in b:
        for name, bit in FLAG_BITS.items():
            fraction(name, int(np.count_nonzero(a["flags"] & bit)), int(np.count_nonzero(b["flags"] & bit)))
        fraction("unfinished", int(np.count_nonzero((a["flags"] & TERMINAL) == 0)),
                 int(np.count_nonzero((b["flags"] & TERMINAL) == 0)))
    if "hit_channel" in a and "hit_channel" in b:
        nch = int(max(a["hit_channel"].max(initial=-1), b["hit_channel"].max(initial=-1)) + 1)
        ca = np.bincount(a["hit_channel"], minlength=nch)
        cb = np.bincount(b["hit_channel"], minlength=nch)
        tested = (ca + cb) / 2 >= 25
        zc = [two_proportion_z(int(ca[c]), n1, int(cb[c]), n2) for c in np.flatnonzero(tested)]
        max_z = float(np.max(np.abs(zc))) if zc else 0.0
        details["hits_per_channel"] = {"channels": len(zc), "max_z": max_z}
        lines.append("hits per channel: %d channels, max |z| %.2f%s" % (len(zc), max_z,
                                                                       "  FAIL" if max_z > max_sigma else ""))
        if max_z > max_sigma:
            failures.append("hits_per_channel")
        wa = np.bincount(a["hit_channel"], weights=a["hit_w"], minlength=nch)
        wb = np.bincount(b["hit_channel"], weights=b["hit_w"], minlength=nch)
        va = np.bincount(a["hit_channel"], weights=a["hit_w"].astype(float) ** 2, minlength=nch)
        vb = np.bincount(b["hit_channel"], weights=b["hit_w"].astype(float) ** 2, minlength=nch)
        both = (ca >= 25) & (cb >= 25)
        wz = (wa / n1 - wb / n2)[both] / np.sqrt(va[both] / n1 ** 2 + vb[both] / n2 ** 2)
        max_wz = float(np.max(np.abs(wz))) if wz.size else 0.0
        details["weighted_efficiency"] = {"channels": int(both.sum()), "max_z": max_wz,
                                          "total": (float(wa.sum() / n1), float(wb.sum() / n2))}
        lines.append("weighted efficiency per channel: %d channels, max |z| %.2f; total %.6g and %.6g%s" % (
            int(both.sum()), max_wz, wa.sum() / n1, wb.sum() / n2, "  FAIL" if max_wz > max_sigma else ""))
        if max_wz > max_sigma:
            failures.append("weighted_efficiency")
    for key, label in (("hit_t", "hit times"), ("hit_wl", "hit wavelengths"), ("t", "final times")):
        if key in a and key in b:
            ks = ks_test(a[key], b[key], alpha)
            details[key] = ks
            if ks is not None:
                lines.append("%-19s KS %.4g (bound %.4g)%s" % (label, ks["distance"], ks["bound"],
                                                               "" if ks["pass"] else "  FAIL"))
                if not ks["pass"]:
                    failures.append(key)
    details["failures"] = failures
    lines.insert(0, "photons: %d and %d, %s" % (n1, n2, "statistically consistent" if not failures
                                                 else "INCONSISTENT: " + ", ".join(failures)))
    return Comparison(not failures, lines, details)
