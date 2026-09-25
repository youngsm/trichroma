"""Bitwise tests of chroma.triton.engine.exact against CUDA probe kernels.

The ground truth (``test/data/legacy_exact_probes.npz``) was produced by
``python -m chroma.triton.legacy.probes generate`` in the CUDA environment:
small kernels compiled with the original ``cuda_options`` from the ORIGINAL
Chroma headers, run on random and adversarial inputs (special values,
normal/grazing incidence, the critical angle, parallel rays, rays through
vertices/edges, axis-aligned and signed-zero directions, wavelengths on and
beyond the table grid, repeated CDF abscissae). Draw-consuming probes ran on
random XORWOW states; the draws are regenerated here.

Set ``CHROMA_LEGACY_PROBES`` to test a freshly generated file.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA device required", allow_module_level=True)

from chroma.triton.engine import exact as X  # noqa: E402
from chroma.triton.legacy.probes import xorwow_draws  # noqa: E402

DATA = os.environ.get("CHROMA_LEGACY_PROBES",
                      os.path.join(os.path.dirname(__file__), "data", "legacy_exact_probes.npz"))
BLOCK = 128


@pytest.fixture(scope="module")
def probes():
    if not os.path.exists(DATA):
        pytest.skip("probe data %s not found" % DATA)
    with np.load(DATA) as data:
        return {k: data[k] for k in data.files}


def dev(x):
    x = np.ascontiguousarray(x)
    if x.dtype == np.uint32:
        x = x.view(np.int32)
    return torch.from_numpy(x.copy()).cuda()


def same_bits(expected, actual, what):
    e = np.ascontiguousarray(expected, np.float32).view(np.uint32)
    a = np.ascontiguousarray(actual, np.float32).view(np.uint32)
    assert e.shape == a.shape, what
    bad = np.flatnonzero((e != a).reshape(len(e), -1).any(axis=1)) if e.ndim > 1 else np.flatnonzero(e != a)
    assert len(bad) == 0, "%s: %d/%d rows differ, first %d: expected %s actual %s" % (
        what, len(bad), len(e), bad[0], e[bad[0]], a[bad[0]])


# ------------------------------------------------------------------ kernels


@triton.jit
def _math_kernel(x, y, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(x + i, mask=m, other=0.5)
    b = tl.load(y + i, mask=m, other=0.5)
    tl.store(out + i * 8 + 0, X.acosf_(a), mask=m)
    tl.store(out + i * 8 + 1, X.asinf_(a), mask=m)
    tl.store(out + i * 8 + 2, X.atan2f_(b, a), mask=m)
    tl.store(out + i * 8 + 3, X.expf_(a), mask=m)
    tl.store(out + i * 8 + 4, X.logf_(b), mask=m)
    tl.store(out + i * 8 + 5, X.fsin(a), mask=m)
    tl.store(out + i * 8 + 6, X.fcos(a), mask=m)
    tl.store(out + i * 8 + 7, X.fdiv(a, b), mask=m)


@triton.jit
def _normalize_kernel(v, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x, y, z = X.normalize3(tl.load(v + i * 3, mask=m), tl.load(v + i * 3 + 1, mask=m), tl.load(v + i * 3 + 2, mask=m))
    tl.store(out + i * 3, x, mask=m)
    tl.store(out + i * 3 + 1, y, mask=m)
    tl.store(out + i * 3 + 2, z, mask=m)


@triton.jit
def _interp_kernel(x, table, nw, start, step, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    xx = tl.load(x + i, mask=m, other=100.0)
    tl.store(out + i, X.interp_property(table, xx, start, step, nw, m), mask=m)


@triton.jit
def _interpn_kernel(x, xp, fp, k, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, X.interp_nonuniform(tl.load(x + i, mask=m, other=0.0), xp, fp, k, m), mask=m)


@triton.jit
def _idx_kernel(x, xp, k, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    idx, iidx, top = X.interp_idx(tl.load(x + i, mask=m, other=0.0), xp, k, m)
    tl.store(out + i, idx, mask=m)


@triton.jit
def _cdf_kernel(u, cdf, k, x0, delta, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    uu = tl.load(u + i, mask=m, other=0.5)
    tl.store(out + i, X.sample_cdf_uniform(cdf, k, x0, delta, uu, m), mask=m)


@triton.jit
def _triangle_kernel(v, o, d, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    hit, t = X.intersect_triangle(
        tl.load(v + i * 9 + 0, mask=m), tl.load(v + i * 9 + 1, mask=m), tl.load(v + i * 9 + 2, mask=m),
        tl.load(v + i * 9 + 3, mask=m), tl.load(v + i * 9 + 4, mask=m), tl.load(v + i * 9 + 5, mask=m),
        tl.load(v + i * 9 + 6, mask=m), tl.load(v + i * 9 + 7, mask=m), tl.load(v + i * 9 + 8, mask=m),
        tl.load(o + i * 3, mask=m), tl.load(o + i * 3 + 1, mask=m), tl.load(o + i * 3 + 2, mask=m),
        tl.load(d + i * 3, mask=m), tl.load(d + i * 3 + 1, mask=m), tl.load(d + i * 3 + 2, mask=m))
    tl.store(out + i * 2, tl.where(hit, 1.0, 0.0), mask=m)
    tl.store(out + i * 2 + 1, tl.where(hit, t, 0.0), mask=m)


@triton.jit
def _sphere_kernel(u, width, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    x, y, z = X.uniform_sphere(tl.load(u + i * width, mask=m), tl.load(u + i * width + 1, mask=m))
    tl.store(out + i * 3, x, mask=m)
    tl.store(out + i * 3 + 1, y, mask=m)
    tl.store(out + i * 3 + 2, z, mask=m)


@triton.jit
def _rayleigh_kernel(u, width, d, p, od, op, n, BLOCK: tl.constexpr, ISOLATED: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    dx = tl.load(d + i * 3, mask=m)
    dy = tl.load(d + i * 3 + 1, mask=m)
    dz = tl.load(d + i * 3 + 2, mask=m)
    px = tl.load(p + i * 3, mask=m)
    py = tl.load(p + i * 3 + 1, mask=m)
    pz = tl.load(p + i * 3 + 2, mask=m)
    u0 = tl.load(u + i * width, mask=m)
    u1 = tl.load(u + i * width + 1, mask=m)
    if ISOLATED:
        a, b, c, e, f, g = X.rayleigh_scatter_raw(dx, dy, dz, px, py, pz, u0, u1)
        a, b, c = X.normalize3(a, b, c)
        e, f, g = X.normalize3_xfirst(e, f, g)
    else:
        a, b, c, e, f, g = X.rayleigh_scatter(dx, dy, dz, px, py, pz, u0, u1)
    tl.store(od + i * 3, a, mask=m)
    tl.store(od + i * 3 + 1, b, mask=m)
    tl.store(od + i * 3 + 2, c, mask=m)
    tl.store(op + i * 3, e, mask=m)
    tl.store(op + i * 3 + 1, f, mask=m)
    tl.store(op + i * 3 + 2, g, mask=m)


@triton.jit
def _fresnel_kernel(u, width, d, p, nn, n1, n2, od, op, hist, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a, b, c, e, f, g, refl = X.fresnel(
        tl.load(d + i * 3, mask=m), tl.load(d + i * 3 + 1, mask=m), tl.load(d + i * 3 + 2, mask=m),
        tl.load(p + i * 3, mask=m), tl.load(p + i * 3 + 1, mask=m), tl.load(p + i * 3 + 2, mask=m),
        tl.load(nn + i * 3, mask=m), tl.load(nn + i * 3 + 1, mask=m), tl.load(nn + i * 3 + 2, mask=m),
        tl.load(n1 + i, mask=m), tl.load(n2 + i, mask=m),
        tl.load(u + i * width, mask=m), tl.load(u + i * width + 1, mask=m))
    tl.store(od + i * 3, a, mask=m)
    tl.store(od + i * 3 + 1, b, mask=m)
    tl.store(od + i * 3 + 2, c, mask=m)
    tl.store(op + i * 3, e, mask=m)
    tl.store(op + i * 3 + 1, f, mask=m)
    tl.store(op + i * 3 + 2, g, mask=m)
    tl.store(hist + i, tl.where(refl, 64, 0), mask=m)


@triton.jit
def _specular_kernel(d, nn, od, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a, b, c = X.specular_reflect(
        tl.load(d + i * 3, mask=m), tl.load(d + i * 3 + 1, mask=m), tl.load(d + i * 3 + 2, mask=m),
        tl.load(nn + i * 3, mask=m), tl.load(nn + i * 3 + 1, mask=m), tl.load(nn + i * 3 + 2, mask=m))
    tl.store(od + i * 3, a, mask=m)
    tl.store(od + i * 3 + 1, b, mask=m)
    tl.store(od + i * 3 + 2, c, mask=m)


@triton.jit
def _diffuse_kernel(u, width, count, nn, od, op, used, n, BLOCK: tl.constexpr, ISOLATED: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    nx = tl.load(nn + i * 3, mask=m, other=0.0)
    ny = tl.load(nn + i * 3 + 1, mask=m, other=0.0)
    nz = tl.load(nn + i * 3 + 2, mask=m, other=1.0)
    limit = tl.load(count + i, mask=m, other=0)
    cur = tl.zeros((BLOCK,), tl.int32)
    dx = tl.zeros((BLOCK,), tl.float32)
    dy = tl.zeros((BLOCK,), tl.float32)
    dz = tl.zeros((BLOCK,), tl.float32)
    pending = m
    while tl.max(pending.to(tl.int32), 0) > 0:
        a = tl.load(u + i * width + cur, mask=pending, other=0.5)
        b = tl.load(u + i * width + cur + 1, mask=pending, other=0.5)
        x, y, z, ndotv = X.diffuse_candidate(nx, ny, nz, a, b)
        acc = tl.load(u + i * width + cur + 2, mask=pending, other=0.0)
        dx = tl.where(pending, x, dx)
        dy = tl.where(pending, y, dy)
        dz = tl.where(pending, z, dz)
        cur = tl.where(pending, cur + 3, cur)
        pending = pending & ~X.diffuse_accept(acc, ndotv) & (cur + 3 <= limit)
    a = tl.load(u + i * width + cur, mask=m, other=0.5)
    b = tl.load(u + i * width + cur + 1, mask=m, other=0.5)
    if ISOLATED:
        sx, sy, sz = X.uniform_sphere(a, b)
        px, py, pz = X.cross3(sx, sy, sz, dx, dy, dz)
        px, py, pz = X.normalize3_xfirst(px, py, pz)
    else:
        px, py, pz = X.random_polarization(a, b, dx, dy, dz)
    tl.store(od + i * 3, dx, mask=m)
    tl.store(od + i * 3 + 1, dy, mask=m)
    tl.store(od + i * 3 + 2, dz, mask=m)
    tl.store(op + i * 3, px, mask=m)
    tl.store(op + i * 3 + 1, py, mask=m)
    tl.store(op + i * 3 + 2, pz, mask=m)
    tl.store(used + i, cur + 2, mask=m)


@triton.jit
def _box_kernel(nodes, o, d, wx, wy, wz, scale, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tmin, tmax = X.node_box(tl.load(nodes + i * 4, mask=m, other=0), tl.load(nodes + i * 4 + 1, mask=m, other=0),
                            tl.load(nodes + i * 4 + 2, mask=m, other=0),
                            tl.load(o + i * 3, mask=m), tl.load(o + i * 3 + 1, mask=m), tl.load(o + i * 3 + 2, mask=m),
                            tl.load(d + i * 3, mask=m), tl.load(d + i * 3 + 1, mask=m), tl.load(d + i * 3 + 2, mask=m),
                            wx, wy, wz, scale)
    hit = X.fle(tmin, tmax)  # intersect_box: tmin <= tmax
    tl.store(out + i * 2, tl.where(hit, 1.0, 0.0), mask=m)
    tl.store(out + i * 2 + 1, tl.where(hit, tmin, 0.0), mask=m)


def grid(n):
    return ((n + BLOCK - 1) // BLOCK,)


# -------------------------------------------------------------------- tests


def test_fast_math_functions(probes):
    x, y = probes["math_x"], probes["math_y"]
    out = torch.empty(len(x) * 8, dtype=torch.float32, device="cuda")
    _math_kernel[grid(len(x))](dev(x), dev(y), out, len(x), BLOCK=BLOCK)
    got = out.cpu().numpy().reshape(-1, 8)
    for k, name in enumerate(("acosf", "asinf", "atan2f", "expf", "logf", "sinf", "cosf", "fdiv")):
        same_bits(probes["math_out"][:, k], got[:, k], name)


def test_normalize(probes):
    v = probes["norm_in"]
    out = torch.empty(v.size, dtype=torch.float32, device="cuda")
    _normalize_kernel[grid(len(v))](dev(v), out, len(v), BLOCK=BLOCK)
    same_bits(probes["norm_out"], out.cpu().numpy().reshape(-1, 3), "normalize")


def test_interp_property_including_past_the_end_read(probes):
    x = probes["interp_x"]
    nw, start, step = probes["interp_grid"]
    out = torch.empty(len(x), dtype=torch.float32, device="cuda")
    _interp_kernel[grid(len(x))](dev(x), dev(probes["interp_table"]), int(nw), float(start), float(step), out,
                                 len(x), BLOCK=BLOCK)
    same_bits(probes["interp_out"], out.cpu().numpy(), "interp_property")


def test_interp_nonuniform(probes):
    x = probes["interpn_x"]
    out = torch.empty(len(x), dtype=torch.float32, device="cuda")
    _interpn_kernel[grid(len(x))](dev(x), dev(probes["interpn_xp"]), dev(probes["interpn_fp"]),
                                  len(probes["interpn_xp"]), out, len(x), BLOCK=BLOCK)
    same_bits(probes["interpn_out"], out.cpu().numpy(), "interp")


def test_interp_idx(probes):
    x = probes["idx_x"]
    out = torch.empty(len(x), dtype=torch.float32, device="cuda")
    _idx_kernel[grid(len(x))](dev(x), dev(probes["idx_xp"]), len(probes["idx_xp"]), out, len(x), BLOCK=BLOCK)
    same_bits(probes["idx_out"], out.cpu().numpy(), "interp_idx")


def test_sample_cdf(probes):
    draws = xorwow_draws(probes["cdf_states"], probes["cdf_counts"])
    assert np.all(probes["cdf_counts"] == 1)
    out = torch.empty(len(draws), dtype=torch.float32, device="cuda")
    _cdf_kernel[grid(len(draws))](dev(draws[:, 0].copy()), dev(probes["cdf_table"]), len(probes["cdf_table"]),
                                  60.0, 5.0, out, len(draws), BLOCK=BLOCK)
    same_bits(probes["cdf_out"], out.cpu().numpy(), "sample_cdf")


def test_intersect_triangle(probes):
    n = len(probes["tri_v"])
    out = torch.empty(n * 2, dtype=torch.float32, device="cuda")
    _triangle_kernel[grid(n)](dev(probes["tri_v"]), dev(probes["tri_o"]), dev(probes["tri_d"]), out, n, BLOCK=BLOCK)
    same_bits(probes["tri_out"], out.cpu().numpy().reshape(-1, 2), "intersect_triangle")


def test_uniform_sphere(probes):
    counts = probes["sphere_counts"]
    assert np.all(counts == 2)
    draws = xorwow_draws(probes["sphere_states"], counts)
    out = torch.empty(len(draws) * 3, dtype=torch.float32, device="cuda")
    _sphere_kernel[grid(len(draws))](dev(draws), draws.shape[1], out, len(draws), BLOCK=BLOCK)
    same_bits(probes["sphere_out"], out.cpu().numpy().reshape(-1, 3), "uniform_sphere")


# Context-dependent lowering. In a probe compiled in isolation NVVM keeps
# ``x*x`` as the plain multiply of the polarization's dot(v, v) (Rayleigh and
# diffuse); every normalization in the original propagate kernel keeps
# ``y*y`` (exact.normalize3, verified by the tape fixtures). The probes are
# therefore compared with the same arithmetic normalized the isolated way
# (exact.normalize3_xfirst), which pins down everything but that one choice.


def _rayleigh(probes, isolated):
    counts = probes["ray_counts"]
    assert np.all(counts == 2)
    draws = xorwow_draws(probes["ray_states"], counts)
    n = len(draws)
    od = torch.empty(n * 3, dtype=torch.float32, device="cuda")
    op = torch.empty(n * 3, dtype=torch.float32, device="cuda")
    _rayleigh_kernel[grid(n)](dev(draws), draws.shape[1], dev(probes["ray_dir"]), dev(probes["ray_pol"]), od, op, n,
                              BLOCK=BLOCK, ISOLATED=isolated)
    return od.cpu().numpy().reshape(-1, 3), op.cpu().numpy().reshape(-1, 3)


def test_rayleigh(probes):
    od, op = _rayleigh(probes, isolated=True)
    same_bits(probes["ray_od"], od, "rayleigh direction")
    same_bits(probes["ray_op"], op, "rayleigh polarization (isolated normalization)")
    od2, op2 = _rayleigh(probes, isolated=False)
    same_bits(od, od2, "rayleigh_scatter direction")
    # the two normalizations differ in the last bit for a few percent of rows
    assert 0 < np.count_nonzero((op.view(np.uint32) != op2.view(np.uint32)).any(axis=1)) < len(op) // 5


def test_fresnel(probes):
    counts = probes["fr_counts"]
    assert np.all(counts == 2)
    draws = xorwow_draws(probes["fr_states"], counts)
    n = len(draws)
    od = torch.empty(n * 3, dtype=torch.float32, device="cuda")
    op = torch.empty(n * 3, dtype=torch.float32, device="cuda")
    hist = torch.empty(n, dtype=torch.int32, device="cuda")
    _fresnel_kernel[grid(n)](dev(draws), draws.shape[1], dev(probes["fr_dir"]), dev(probes["fr_pol"]),
                             dev(probes["fr_normal"]), dev(probes["fr_n1"]), dev(probes["fr_n2"]), od, op, hist, n,
                             BLOCK=BLOCK)
    same_bits(probes["fr_od"], od.cpu().numpy().reshape(-1, 3), "fresnel direction")
    same_bits(probes["fr_op"], op.cpu().numpy().reshape(-1, 3), "fresnel polarization")
    assert np.array_equal(probes["fr_hist"], hist.cpu().numpy().view(np.uint32))


def test_specular(probes):
    n = len(probes["spec_dir"])
    od = torch.empty(n * 3, dtype=torch.float32, device="cuda")
    _specular_kernel[grid(n)](dev(probes["spec_dir"]), dev(probes["spec_normal"]), od, n, BLOCK=BLOCK)
    same_bits(probes["spec_out"], od.cpu().numpy().reshape(-1, 3), "specular")


def _diffuse(probes, isolated):
    counts = probes["dif_counts"]
    draws = xorwow_draws(probes["dif_states"], counts)
    n = len(draws)
    od = torch.empty(n * 3, dtype=torch.float32, device="cuda")
    op = torch.empty(n * 3, dtype=torch.float32, device="cuda")
    used = torch.empty(n, dtype=torch.int32, device="cuda")
    _diffuse_kernel[grid(n)](dev(draws), draws.shape[1], dev(counts.astype(np.int32)), dev(probes["dif_normal"]),
                             od, op, used, n, BLOCK=BLOCK, ISOLATED=isolated)
    assert np.array_equal(used.cpu().numpy(), counts), "diffuse draw counts"
    return od.cpu().numpy().reshape(-1, 3), op.cpu().numpy().reshape(-1, 3)


def test_diffuse(probes):
    counts = probes["dif_counts"]
    assert np.all(counts % 3 == 2) and counts.max() > 5, "rejection loop exercised"
    od, op = _diffuse(probes, isolated=True)
    same_bits(probes["dif_od"], od, "diffuse direction")
    same_bits(probes["dif_op"], op, "diffuse polarization (isolated normalization)")
    od2, op2 = _diffuse(probes, isolated=False)
    same_bits(od, od2, "diffuse direction (random_polarization path)")


def test_node_box(probes):
    n = len(probes["box_nodes"])
    out = torch.empty(n * 2, dtype=torch.float32, device="cuda")
    w = probes["box_world"]
    _box_kernel[grid(n)](dev(probes["box_nodes"]), dev(probes["box_o"]), dev(probes["box_d"]), float(w[0]),
                         float(w[1]), float(w[2]), float(probes["box_scale"][0]), out, n, BLOCK=BLOCK)
    same_bits(probes["box_out"], out.cpu().numpy().reshape(-1, 2), "intersect_box")
