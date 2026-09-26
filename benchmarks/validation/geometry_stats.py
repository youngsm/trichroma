"""Summarize geometry_rays.py output: production vs CUDA Chroma (W) nearest boundaries on the same rays."""
import sys
import numpy as np
z = np.load(sys.argv[1])
tp, tw = z["tri_prod"], z["tri_w"]
dp, dw = z["t_prod"].astype(np.float64), z["t_w"].astype(np.float64)
n = len(tp)
kind = lambda t: np.where(t >= 0, "mesh", np.where(t == -2, "wire", "none"))
kp, kw = kind(tp), kind(tw)
same = tp == tw
print("rays %d; W flags (NaN/overflow): %d" % (n, int((z["flag_w"] != 0).sum())))
print("same boundary: %d (%.5f%%)" % (same.sum(), 100 * same.mean()))
for a in ("mesh", "wire", "none"):
    for b in ("mesh", "wire", "none"):
        c = int(((kp == a) & (kw == b) & ~same).sum())
        if c:
            print("  differ: production %-4s vs W %-4s: %d (%.2e of rays)" % (a, b, c, c / n))
mesh = same & (tp >= 0)
rel = np.abs(dp[mesh] - dw[mesh]) / np.maximum(dw[mesh], 1e-3)
absd = np.abs(dp[mesh] - dw[mesh])
print("same mesh triangle: %d; |t_prod - t_W|: median %.2e mm, 99.99%% %.2e mm, max %.2e mm; relative max %.2e" % (
    mesh.sum(), np.median(absd), np.percentile(absd, 99.99), absd.max(), rel.max()))
dot = np.abs((z["n_prod"][mesh] * z["n_w"][mesh]).sum(1))
print("  normals: 1 - |n_prod . n_W| max %.2e" % (1 - dot).max())
for name in ("m1", "m2", "s"):
    pairs = np.unique(np.stack([z[name + "_prod"][mesh], z[name + "_w"][mesh]], 1), axis=0)
    ok = len(np.unique(pairs[:, 0])) == len(pairs) and len(np.unique(pairs[:, 1])) == len(pairs)
    print("  %s mapping production->W: %s (%d pairs)" % (name, "one-to-one" if ok else "INCONSISTENT", len(pairs)))
wire = same & (tp == -2)
if wire.any():
    d = np.abs(dp[wire] - dw[wire])
    print("same wire hit: %d; |t diff| median %.2e max %.2e mm" % (wire.sum(), np.median(d), d.max()))
diff = ~same
# mesh/mesh differences: ties (same distance) vs genuine
mm = diff & (kp == "mesh") & (kw == "mesh")
if mm.any():
    dd = np.abs(dp[mm] - dw[mm])
    print("mesh/mesh different triangles: %d; |t diff| quantiles 50%% %.2e, 90%% %.2e, max %.2e mm" % (
        mm.sum(), np.median(dd), np.percentile(dd, 90), dd.max()))
    same_surface = (z["s_prod"][mm] == z["s_prod"][mm])  # placeholder for mapping-based check below
    tie = dd < 1e-3
    print("  of which within 1e-3 mm (edge/vertex ties): %d; farther: %d" % (tie.sum(), (~tie).sum()))
    idx = np.flatnonzero(mm)[~tie][:10]
    for i in idx:
        print("   ray %d kind %d pos %s dir %s last %d: prod tri %d t %.5f, W tri %d t %.5f" % (
            i, z["kind"][i], np.round(z["pos"][i], 4), np.round(z["dir"][i], 4), z["last"][i], tp[i], dp[i], tw[i], dw[i]))
for a, b in (("mesh", "none"), ("none", "mesh"), ("wire", "mesh"), ("mesh", "wire"), ("wire", "none"), ("none", "wire")):
    sel = np.flatnonzero(diff & (kp == a) & (kw == b))
    if len(sel):
        print("examples production %s / W %s:" % (a, b))
        for i in sel[:6]:
            print("   ray %d kind %d pos %s dir %s last %d: prod tri %d t %.5f, W tri %d t %.5f" % (
                i, z["kind"][i], np.round(z["pos"][i], 4), np.round(z["dir"][i], 4), z["last"][i], tp[i], dp[i], tw[i], dw[i]))
