"""Statistics of the weighted validation runs (weighted_run.py outputs named TAG_WORKLOAD_sSEED.npz).

usage: weighted_stats.py DATA_DIR
Tags: C_cuda (CUDA Chroma), P0_fixes0 (CHROMA_TRITON_FIXES=0), P1_fixes_legacywires (CHROMA_TRITON_LEGACY_WIRES=1),
P2_default, P3_roulette (CHROMA_TRITON_ROULETTE=0.05), P4_wavefront (CHROMA_TRITON_FUSED=0 CHROMA_TRITON_GRID=0),
P5_wavefront_grid (CHROMA_TRITON_FUSED=0), P6_wavefront_fixes0; workloads W1, W2 (and W1nw, W2nw without wires).

Observable: every photon's detected weight w_i (0 if not detected; a non-finite weight counts as 0, as
chroma-lar's generate_lut.py drops them), its channel and its detection time. Errors are per-photon
(photons are independent), so no assumption about the weight distribution is made.
"""
import glob, os, sys
import numpy as np
from scipy import stats

D = sys.argv[1] if len(sys.argv) > 1 else "."
TBINS = np.concatenate([np.arange(0, 400, 5.0), np.arange(400, 1000, 25.0), np.arange(1000, 3001, 250.0), [1e9]])
NCH = 162
BITS = {"rayleigh": 16, "diffuse": 32, "specular": 64}


def load(tag, w, seed):
    f = "%s/%s_%s_s%d.npz" % (D, tag, w, seed)
    if not os.path.exists(f):
        return None
    z = np.load(f)
    flags = z["flags"].astype(np.uint32)
    weights = z["weights"].astype(np.float64)
    det = (flags & 4) != 0
    fin = np.isfinite(weights)
    wd = np.where(det & fin, weights, 0.0)
    ch = np.where(det & fin, z["channel"].astype(np.int64), -1)
    tb = np.where(det & fin, np.searchsorted(TBINS, z["t"], side="right") - 1, -1)
    cats = {
        "detected": det & fin, "detected, weight not finite": det & ~fin,
        "not detected, weight not finite": ~det & ~fin,
        "bulk absorbed": ((flags & 2) != 0) & fin & ~det, "surface absorbed": ((flags & 8) != 0) & ~det & fin,
        "no hit": (flags & 1) != 0, "roulette": (flags & (1 << 29)) != 0,
        "alive at max_steps": (flags & (1 | 2 | 4 | 8 | (1 << 29) | (1 << 31) | (1 << 15))) == 0,
    }
    return dict(N=len(wd), w=wd, ch=ch, tb=tb, flags=flags, cats={k: int(v.sum()) for k, v in cats.items()},
                elapsed=float(z["elapsed"]), env=str(z["env"]))


def binned(r, key, nb, g=None):
    """Per-bin mean contribution per photon and its variance of the mean."""
    idx = r[key] if g is None else remap(r, key, g)
    ok = idx >= 0
    s1 = np.bincount(idx[ok], r["w"][ok], nb)
    s2 = np.bincount(idx[ok], r["w"][ok] ** 2, nb)
    n = r["N"]
    m = s1 / n
    return m, (s2 / n - m * m) / n


def bits(r):
    out = {}
    for name, b in BITS.items():
        x = np.where((r["flags"] & b) != 0, r["w"], 0.0)
        out[name] = (x.mean(), x.var() / r["N"])
    return out


def scalar(r):
    return r["w"].mean(), r["w"].var() / r["N"]


def chi2(diff, var):
    ok = var > 0
    c = float(np.sum(diff[ok] ** 2 / var[ok]))
    k = int(ok.sum())
    return c, k, float(stats.chi2.sf(c, k)), float(np.max(np.abs(diff[ok]) / np.sqrt(var[ok]))) if k else 0.0


MIN_HITS = 100  # bins are merged (in order) until both samples have this many hits: the per-photon variance
                # estimate of a bin is only reliable with enough hits (a Russian-roulette sample has few, heavy hits)


def groups(runs_a, runs_b, key, nb):
    """Map bin -> merged bin such that every merged bin holds >= MIN_HITS hits of both samples."""
    ca = sum(np.bincount(r[key][r[key] >= 0], minlength=nb) for r in runs_a)
    cb = sum(np.bincount(r[key][r[key] >= 0], minlength=nb) for r in runs_b)
    g = np.zeros(nb, np.int64)
    k, sa, sb = 0, 0, 0
    for i in range(nb):
        g[i] = k
        sa += ca[i]; sb += cb[i]
        if sa >= MIN_HITS and sb >= MIN_HITS:
            k, sa, sb = k + 1, 0, 0
    if k > 0 and (sa < MIN_HITS or sb < MIN_HITS):
        g[g == k] = k - 1  # the incomplete last group joins the previous one
        k -= 1
    return g, k + 1


def remap(r, key, g):
    return np.where(r[key] >= 0, g[np.maximum(r[key], 0)], -1)


def paired(x, y):
    """X - Y for the same photons (same seed and ids)."""
    d = x["w"] - y["w"]
    n = x["N"]
    out = {"total": (d.mean(), d.var() / n, y["w"].mean()), "changed": int((d != 0).sum())}
    for key, nb0 in (("ch", NCH), ("tb", len(TBINS) - 1)):
        g, nb = groups([x], [y], key, nb0)
        xk, yk = remap(x, key, g), remap(y, key, g)
        sx = np.bincount(xk[xk >= 0], x["w"][xk >= 0], nb)
        sy = np.bincount(yk[yk >= 0], y["w"][yk >= 0], nb)
        sxx = np.bincount(xk[xk >= 0], x["w"][xk >= 0] ** 2, nb)
        syy = np.bincount(yk[yk >= 0], y["w"][yk >= 0] ** 2, nb)
        same = (xk == yk) & (xk >= 0)
        sxy = np.bincount(xk[same], x["w"][same] * y["w"][same], nb)
        dm = (sx - sy) / n
        var = ((sxx + syy - 2 * sxy) / n - dm * dm) / n
        out[key] = chi2(dm, var)
    for name, b in BITS.items():
        dx = np.where((x["flags"] & b) != 0, x["w"], 0.0) - np.where((y["flags"] & b) != 0, y["w"], 0.0)
        out[name] = (dx.mean(), dx.var() / n)
    return out


def fmt(m, v, ref=None):
    s = "%+.3e +- %.1e (%+.1f sigma)" % (m, np.sqrt(v), m / np.sqrt(v) if v > 0 else 0)
    if ref:
        s += ", relative %+.4f%% +- %.4f%%" % (100 * m / ref, 100 * np.sqrt(v) / ref)
    return s


def report_paired(w):
    print("\n## Paired production variants, %s, seed 1 (same photons, same Philox keys)" % w)
    runs = {t: load(t, w, 1) for t in ("P0_fixes0", "P1_fixes_legacywires", "P2_default", "P3_roulette",
                                       "P4_wavefront", "P5_wavefront_grid", "P6_wavefront_fixes0")}
    for t, r in runs.items():
        if r is not None:
            m, v = scalar(r)
            print("%-22s detected weight/photon %.6f +- %.6f  %s  (%.1f s)" % (t, m, np.sqrt(v), r["cats"], r["elapsed"]))
    pairs = [("P1_fixes_legacywires", "P0_fixes0", "polarization + Fresnel + 32-bit history fixes"),
             ("P2_default", "P1_fixes_legacywires", "production wires (FP32 form, exit rule, cull)"),
             ("P2_default", "P0_fixes0", "all fixes"),
             ("P3_roulette", "P2_default", "Russian roulette 0.05 (must be 0)"),
             ("P4_wavefront", "P2_default", "wavefront vs fused scheduler (must be 0)"),
             ("P5_wavefront_grid", "P4_wavefront", "grid bulk shortcut (must be 0)"),
             ("P6_wavefront_fixes0", "P0_fixes0", "wavefront vs fused, bug-compatible (must be 0)")]
    for a, b, what in pairs:
        if runs[a] is None or runs[b] is None:
            continue
        p = paired(runs[a], runs[b])
        m, v, ref = p["total"]
        print("- %s: %s - %s = %s; photons changed %d" % (what, a, b, fmt(m, v, ref), p["changed"]))
        for key, label in (("ch", "channels"), ("tb", "time bins")):
            c, k, pv, zmax = p[key]
            print("    %s: chi2 %.1f / %d (p = %.3g), max |z| %.1f" % (label, c, k, pv, zmax))
        print("    history-bit weighted detections: " + "; ".join("%s %s" % (n, fmt(*p[n])) for n in BITS))


def pooled(tag, w, seeds):
    rs = [r for r in (load(tag, w, s) for s in seeds) if r is not None]
    return rs


def report_unpaired(w, a_tag, b_tag, seeds, what):
    A, B = pooled(a_tag, w, seeds), pooled(b_tag, w, seeds)
    if not A or not B:
        return
    print("\n## %s, %s: %s (%d runs) vs %s (%d runs), 10M photons each" % (what, w, a_tag, len(A), b_tag, len(B)))
    for tag, rs in ((a_tag, A), (b_tag, B)):
        ms = np.array([scalar(r) for r in rs])
        c, k, pv, zmax = chi2(ms[:, 0] - ms[:, 0].mean(), ms[:, 1] * (1 - 1 / len(ms)))
        print("  %-12s per-seed detected weight/photon: %s; seed-to-seed chi2 %.1f / %d (p = %.2g)" % (
            tag, " ".join("%.6f" % m for m in ms[:, 0]), c, k - 1, stats.chi2.sf(c, k - 1)))
    def pool(rs, fn):
        vals = [fn(r) for r in rs]
        m = np.mean([v[0] for v in vals], axis=0)
        var = np.sum([v[1] for v in vals], axis=0) / len(vals) ** 2
        return m, var
    ma, va = pool(A, scalar); mb, vb = pool(B, scalar)
    print("  detected weight/photon: %s %.6f, %s %.6f; difference %s" % (a_tag, ma, b_tag, mb, fmt(ma - mb, va + vb, mb)))
    for key, nb0, label in (("ch", NCH, "channels"), ("tb", len(TBINS) - 1, "time bins")):
        g, nb = groups(A, B, key, nb0)
        ma_, va_ = pool(A, lambda r: binned(r, key, nb, g)); mb_, vb_ = pool(B, lambda r: binned(r, key, nb, g))
        c, k, pv, zmax = chi2(ma_ - mb_, va_ + vb_)
        print("  %s: chi2 %.1f / %d (p = %.3g), max |z| %.1f" % (label, c, k, pv, zmax))
    ba = [bits(r) for r in A]; bb = [bits(r) for r in B]
    for name in BITS:
        m1 = np.mean([x[name][0] for x in ba]); v1 = np.sum([x[name][1] for x in ba]) / len(ba) ** 2
        m2 = np.mean([x[name][0] for x in bb]); v2 = np.sum([x[name][1] for x in bb]) / len(bb) ** 2
        print("  weighted detections with %-8s: %s %.6f, %s %.6f; difference %s" % (name, a_tag, m1, b_tag, m2, fmt(m1 - m2, v1 + v2, m2)))
    for cat in A[0]["cats"]:
        na = sum(r["cats"][cat] for r in A); Na = sum(r["N"] for r in A)
        nb_ = sum(r["cats"][cat] for r in B); Nb = sum(r["N"] for r in B)
        pa, pb = na / Na, nb_ / Nb
        p = (na + nb_) / (Na + Nb)
        se = np.sqrt(p * (1 - p) * (1 / Na + 1 / Nb)) if 0 < p < 1 else 0
        print("  %-32s %s %.6f  %s %.6f  (%+.1f sigma)" % (cat, a_tag, pa, b_tag, pb, (pa - pb) / se if se else 0))


if __name__ == "__main__":
    for w in ("W1", "W2"):
        report_paired(w)
        report_unpaired(w, "C_cuda", "P0_fixes0", range(1, 7), "CUDA Chroma vs bug-compatible production (FIXES=0)")
        report_unpaired(w, "P2_default", "C_cuda", range(1, 7), "Production default vs CUDA Chroma (the fixes' net effect)")
    for w in ("W1nw", "W2nw"):
        report_unpaired(w, "C_cuda", "P0_fixes0", range(1, 7), "No wires: CUDA Chroma vs bug-compatible production")
