"""Float64 wire intersections for the rays of geometry_rays.py where production and CUDA Chroma (W) disagree
about a wire (and a random sample of rays where they agree). Rays starting on a wire (last == -2) are left out:
in transport the production engine skips the wire just left (tested by tests/integration/test_wires.py).

usage: wires_float64.py GEOMETRY.npz [SAMPLE]  (GEOMETRY.npz from geometry_rays.py)
"""
import sys
import numpy as np
from chroma_lar.geometry.config_loader import build_detector_from_config
from trichroma.engine.scene import _wire_records

z = np.load(sys.argv[1])
sample = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
g = build_detector_from_config("detector_config_reflect_reflect3wires")
rec = _wire_records(g, [], []).astype(np.float64)
bits = _wire_records(g, [], [])[:, 17:19].view(np.int32)
planes = [dict(o=r[0:3], u=r[3:6], v=r[6:9], n=r[9:12], pitch=r[12], r=r[13], umin=r[14], umax=r[15], v0=r[16],
               kmin=int(b[0]), kmax=int(b[1])) for r, b in zip(rec, bits)]


def nearest_wire(o, d, T):
    """(t, plane, k) of the nearest float64 wire hit with 0 < t <= T (inside origin: exit root, as both engines)."""
    best = (np.inf, -1, 0)
    for ip, p in enumerate(planes):
        w = o - p["o"]
        wu, wv, wn = w @ p["u"], w @ p["v"] - p["v0"], w @ p["n"]
        du, dv, dn = d @ p["u"], d @ p["v"], d @ p["n"]
        r = p["r"]
        if abs(dn) < 1e-12:
            if abs(wn) > r:
                continue
            t0, t1 = 0.0, T
        else:
            ta, tb = (-r - wn) / dn, (r - wn) / dn
            t0, t1 = max(min(ta, tb), 0.0), min(max(ta, tb), T)
        if t0 > t1:
            continue
        va, vb = wv + dv * t0, wv + dv * t1
        klo = max(int(np.ceil((min(va, vb) - r) / p["pitch"])), p["kmin"])
        khi = min(int(np.floor((max(va, vb) + r) / p["pitch"])), p["kmax"])
        if klo > khi:
            continue
        k = np.arange(klo, khi + 1)
        pv = wv - k * p["pitch"]
        A = dv * dv + dn * dn
        B = pv * dv + wn * dn
        C = pv * pv + wn * wn - r * r
        disc = B * B - A * C
        with np.errstate(invalid="ignore", divide="ignore"):
            sq = np.sqrt(np.maximum(disc, 0))
            t = np.where(C > 0, (-B - sq) / A, (-B + sq) / A)
        ok = (disc >= 0) & (t > 1e-4) & (t <= T)
        uc = wu + du * t
        ok &= (uc >= p["umin"]) & (uc <= p["umax"])
        if ok.any():
            j = np.argmin(np.where(ok, t, np.inf))
            if t[j] < best[0]:
                best = (t[j], ip, int(k[j]))
    return best


tp, tw = z["tri_prod"], z["tri_w"]
dp, dw = z["t_prod"].astype(np.float64), z["t_w"].astype(np.float64)
pos, dirs, last = z["pos"].astype(np.float64), z["dir"].astype(np.float64), z["last"]
wire_any = (tp == -2) | (tw == -2)
leaving = last == -2
disagree = np.flatnonzero(wire_any & ~leaving & ((tp != tw) | (np.abs(dp - dw) > 1e-3 + 1e-6 * dw)))
agree = np.flatnonzero(wire_any & ~leaving & (tp == tw) & (np.abs(dp - dw) <= 1e-3 + 1e-6 * dw))
rng = np.random.default_rng(0)
agree = rng.choice(agree, min(sample, len(agree)), replace=False) if len(agree) else agree
print("rays involving a wire in either engine (not leaving a wire): %d; disagreeing %d; agreeing (sampled) %d; "
      "rays leaving a wire (excluded) %d" % (int((wire_any & ~leaving).sum()), len(disagree), len(agree),
                                              int((wire_any & leaving).sum())))


def wire_at(q):
    """(plane, k, radial distance / r) of the wire axis nearest to point q."""
    best = (-1, 0, np.inf)
    for ip, p in enumerate(planes):
        w = q - p["o"]
        wv, wn = w @ p["v"] - p["v0"], w @ p["n"]
        k = int(np.clip(np.round(wv / p["pitch"]), p["kmin"], p["kmax"]))
        rr = np.hypot(wv - k * p["pitch"], wn) / p["r"]
        if abs(rr - 1) < abs(best[2] - 1):
            best = (ip, k, rr)
    return best


def impact(o, d, ip, k):
    """Closest approach of the ray line to wire (ip, k), in units of r."""
    p = planes[ip]
    w = o - p["o"]
    pv, pn = w @ p["v"] - p["v0"] - k * p["pitch"], w @ p["n"]
    dv, dn = d @ p["v"], d @ p["n"]
    return abs(pv * dn - pn * dv) / np.sqrt(dv * dv + dn * dn) / p["r"]


CATS = ("correct", "same wire, distance off", "wire hit, float64 has another wire first", "spurious wire hit",
        "missed wire hit")


def judge(idx):
    ok = {"production": np.zeros(len(idx), bool), "W": np.zeros(len(idx), bool)}
    cats = {e: {c: 0 for c in CATS} for e in ok}
    off = []
    grazing = {e: [] for e in ok}
    for j, i in enumerate(idx):
        T = max(dp[i] if tp[i] != -1 else 0, dw[i] if tw[i] != -1 else 0, 1.0) + 1.0
        t64, ip64, k64 = nearest_wire(pos[i], dirs[i], T)
        tol = 2e-3 + 2e-6 * T
        for eng, eng_tri, eng_t in (("production", tp[i], dp[i]), ("W", tw[i], dw[i])):
            mesh_t = eng_t if eng_tri >= 0 else np.inf
            if eng_tri == -2:
                good = abs(eng_t - t64) <= tol
                if good:
                    c = "correct"
                else:
                    ipw, kw, _ = wire_at(pos[i] + dirs[i] * eng_t)
                    if (ipw, kw) == (ip64, k64):
                        c = "same wire, distance off"
                        if eng == "W":
                            off.append(eng_t - t64)
                    elif np.isfinite(t64) and t64 < eng_t:
                        c = "wire hit, float64 has another wire first"
                    else:
                        c = "spurious wire hit"
                        grazing[eng].append(impact(pos[i], dirs[i], ipw, kw))
            else:
                good = not (t64 < mesh_t - tol)
                c = "correct" if good else "missed wire hit"
                if not good:
                    grazing[eng].append(impact(pos[i], dirs[i], ip64, k64))
            ok[eng][j] = good
            cats[eng][c] += 1
    return ok["production"], ok["W"], cats, np.array(off), grazing


for name, idx in (("disagreeing", disagree), ("agreeing sample", agree)):
    if not len(idx):
        continue
    ok_p, ok_w, cats, off, grazing = judge(idx)
    print("%s rays: production matches float64 %d/%d (%.4f%%); W matches float64 %d/%d (%.4f%%); both %d, neither %d" % (
        name, ok_p.sum(), len(idx), 100 * ok_p.mean(), ok_w.sum(), len(idx), 100 * ok_w.mean(), (ok_p & ok_w).sum(),
        (~ok_p & ~ok_w).sum()))
    for eng in ("production", "W"):
        print("   %-10s %s" % (eng, "; ".join("%s %d" % (c, n) for c, n in cats[eng].items() if n)))
        if grazing[eng]:
            g_ = np.array(grazing[eng])
            print("      closest approach / r of its missed or spurious wires: min %.6f median %.6f max %.6f" % (
                g_.min(), np.median(g_), g_.max()))
    if len(off):
        print("   W same-wire distance error (mm): median |dt| %.3g, 90%% %.3g, max %.3g" % (
            np.median(np.abs(off)), np.percentile(np.abs(off), 90), np.abs(off).max()))
    bad = idx[~ok_p][:8]
    for i in bad:
        T = max(dp[i], dw[i], 1.0) + 1.0
        print("   production wrong: ray %d pos %s dir %s last %d: prod tri %d t %.5f, W tri %d t %.5f, float64 %s" % (
            i, np.round(pos[i], 4), np.round(dirs[i], 5), last[i], tp[i], dp[i], tw[i], dw[i], nearest_wire(pos[i], dirs[i], T)))
