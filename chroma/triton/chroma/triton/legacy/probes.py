"""CUDA probe kernels compiled from the ORIGINAL headers (unit ground truth).

``python -m chroma.triton.legacy.probes generate --out probes.npz
[--headers /path/to/original/chroma/cuda] [--count N]`` (CUDA environment,
PyCUDA) compiles small kernels that call the original device functions with
the original ``cuda_options`` and runs them on random and adversarial inputs.
``src/trichroma/test/test_legacy_exact.py`` evaluates the same inputs with
:mod:`chroma.triton.engine.exact` and requires identical bits.

Probes consume draws from ordinary ``curandStateXORWOW`` states (random
words); the draws each probe consumed are recorded (Weyl-counter delta) and
regenerated on the host for the Triton side, exactly as the tape does.

These kernels are compiled in isolation; nvcc/ptxas may contract an
expression differently in a small kernel than in the integrated
``propagate`` kernel that :mod:`exact` follows (the analytic-wire quadratic
is the known case). A probe disagreement therefore points to either a
transcription error or a context-dependent lowering, which the integrated
tape fixtures decide.
"""

import argparse
import os
import sys

import numpy as np

PROBE_SOURCE = r"""
#include "linalg.h"
#include "geometry.h"
#include "photon.h"
#include "interpolate.h"
#include "rotate.h"
#include "random.h"

extern "C" {

__global__ void probe_math(int n, const float *x, const float *y, float *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = x[i], b = y[i];
    out[i*8+0] = acosf(a);
    out[i*8+1] = asinf(a);
    out[i*8+2] = atan2f(b, a);
    out[i*8+3] = expf(a);
    out[i*8+4] = logf(b);
    out[i*8+5] = sinf(a);
    out[i*8+6] = cosf(a);
    out[i*8+7] = a / b;
}

__global__ void probe_normalize(int n, const float3 *v, float3 *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    float3 a = v[i];
    a /= norm(a);
    out[i] = a;
}

__global__ void probe_interp(int n, const float *x, float *table, int nw, float start, float step, float *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    Material m;
    m.wavelength_n = nw;
    m.wavelength_start = start;
    m.wavelength_step = step;
    out[i] = interp_property(&m, x[i], table);
}

__global__ void probe_interp_nonuniform(int n, const float *x, float *xp, float *fp, int m, float *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = interp(x[i], m, xp, fp);
}

__global__ void probe_interp_idx(int n, const float *x, float *xp, int m, float *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = interp_idx(x[i], m, xp);
}

__global__ void probe_sample_cdf(int n, curandState *s, float *cdf, int ncdf, float x0, float delta, float *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState r = s[i];
    out[i] = sample_cdf(&r, ncdf, x0, delta, cdf);
    s[i] = r;
}

__global__ void probe_triangle(int n, const float *tri, const float3 *o, const float3 *d, float *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    Triangle t;
    t.v0 = make_float3(tri[i*9+0], tri[i*9+1], tri[i*9+2]);
    t.v1 = make_float3(tri[i*9+3], tri[i*9+4], tri[i*9+5]);
    t.v2 = make_float3(tri[i*9+6], tri[i*9+7], tri[i*9+8]);
    float dist = -12345.0f;
    bool hit = intersect_triangle(o[i], d[i], t, dist);
    out[i*2+0] = hit ? 1.0f : 0.0f;
    out[i*2+1] = hit ? dist : 0.0f;
}

__global__ void probe_sphere(int n, curandState *s, float3 *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState r = s[i];
    out[i] = uniform_sphere(&r);
    s[i] = r;
}

__global__ void probe_rayleigh(int n, curandState *s, const float3 *dir, const float3 *pol, float3 *od, float3 *op)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState r = s[i];
    Photon p;
    p.direction = dir[i];
    p.polarization = pol[i];
    rayleigh_scatter(p, r);
    od[i] = p.direction;
    op[i] = p.polarization;
    s[i] = r;
}

__global__ void probe_fresnel(int n, curandState *s, const float3 *dir, const float3 *pol, const float3 *normal,
                              const float *n1, const float *n2, float3 *od, float3 *op, unsigned int *hist)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState r = s[i];
    Photon p;
    p.direction = dir[i];
    p.polarization = pol[i];
    p.history = 0;
    State st;
    st.surface_normal = normal[i];
    st.refractive_index1 = n1[i];
    st.refractive_index2 = n2[i];
    propagate_at_boundary(p, st, r);
    od[i] = p.direction;
    op[i] = p.polarization;
    hist[i] = p.history;
    s[i] = r;
}

__global__ void probe_specular(int n, const float3 *dir, const float3 *normal, float3 *od)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    Photon p;
    p.direction = dir[i];
    p.history = 0;
    State st;
    st.surface_normal = normal[i];
    propagate_at_specular_reflector(p, st);
    od[i] = p.direction;
}

__global__ void probe_diffuse(int n, curandState *s, const float3 *normal, float3 *od, float3 *op)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandState r = s[i];
    Photon p;
    p.history = 0;
    State st;
    st.surface_normal = normal[i];
    propagate_at_diffuse_reflector(p, st, r);
    od[i] = p.direction;
    op[i] = p.polarization;
    s[i] = r;
}

__global__ void probe_box(int n, const uint4 *nodes, const float3 *o, const float3 *d,
                          float wx, float wy, float wz, float scale, float *out)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint4 node = nodes[i];
    float3 world = make_float3(wx, wy, wz);
    float3 lo = world + make_float3(node.x & 0xFFFF, node.y & 0xFFFF, node.z & 0xFFFF) * scale;
    float3 hi = world + make_float3(node.x >> 16, node.y >> 16, node.z >> 16) * scale;
    float3 neg_origin_inv_dir = -o[i] / d[i];
    float3 inv_dir = 1.0f / d[i];
    float dist = -12345.0f;
    bool hit = intersect_box(neg_origin_inv_dir, inv_dir, lo, hi, dist);
    out[i*2+0] = hit ? 1.0f : 0.0f;
    out[i*2+1] = hit ? dist : 0.0f;
}

}
"""


def _states(rng, count):
    """Random curandStateXORWOW words (d, v0..v4, zeros) as uint32 [N,12]."""
    words = np.zeros((count, 12), np.uint32)
    words[:, :6] = rng.integers(1, 2 ** 32, size=(count, 6), dtype=np.uint64).astype(np.uint32)
    return words


def xorwow_draws(states, counts):
    """Host regeneration of curand_uniform draws from state words [N,12]."""
    st = np.array(states[:, :6], dtype=np.uint64)
    out = []
    mask32 = np.uint64(0xFFFFFFFF)
    width = int(np.max(counts)) if len(counts) else 0
    draws = np.zeros((len(states), width), np.float32)
    for k in range(width):
        v0 = st[:, 1]
        t = (v0 ^ (v0 >> np.uint64(2))) & mask32
        v4 = st[:, 5]
        new = ((v4 ^ (v4 << np.uint64(4))) ^ (t ^ (t << np.uint64(1)))) & mask32
        st[:, 1:5] = st[:, 2:6]
        st[:, 5] = new
        st[:, 0] = (st[:, 0] + np.uint64(362437)) & mask32
        x = ((new + st[:, 0]) & mask32).astype(np.uint32)
        xf = x.astype(np.float32).astype(np.float64)  # cvt.rn.f32.u32
        u = (xf * 2.0 ** -32 + 2.0 ** -33).astype(np.float32)  # one rounding (exact in float64)
        draws[:, k] = u
    return draws


def _vec(rng, count):
    v = rng.normal(size=(count, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v.astype(np.float32)


def generate(out, headers, count=2048, seed=20260925):
    import pycuda.autoinit  # noqa: F401
    import pycuda.compiler
    import pycuda.driver as cuda
    from pycuda import gpuarray as ga
    from chroma.gpu.tools import cuda_options

    WEYL_INV = pow(362437, -1, 2 ** 32)
    module = pycuda.compiler.SourceModule(PROBE_SOURCE, no_extern_c=True,
                                          options=list(cuda_options) + ["-I" + headers])
    rng = np.random.default_rng(seed)
    data = {"headers": np.frombuffer(os.path.abspath(headers).encode(), np.uint8)}
    block = 128
    grid = lambda n: ((n + block - 1) // block, 1)

    def run_states(name, n, *args):
        states = _states(rng, n)
        before = states.copy()
        s_gpu = ga.to_gpu(states)
        module.get_function(name)(np.int32(n), s_gpu, *args, block=(block, 1, 1), grid=grid(n))
        after = s_gpu.get()
        counts = ((after[:, 0].astype(np.uint64) - before[:, 0].astype(np.uint64)) * WEYL_INV) % (2 ** 32)
        return before, counts.astype(np.int64)

    special = np.array([0.0, -0.0, 1.0, -1.0, 0.56, -0.56, np.float32(0.56).view(np.uint32) and 0.5600001,
                        0.5599999, 1e-20, -1e-20, 1e-40, np.inf, -np.inf, np.nan, 0.9999999, -0.9999999,
                        1.0000001, 3.1415927, 1.5707964, 100.0, -100.0], np.float32)
    x = np.concatenate([special, rng.uniform(-1.2, 1.2, count).astype(np.float32),
                        rng.uniform(-10, 10, count).astype(np.float32)])
    y = np.concatenate([special[::-1], rng.uniform(-1.2, 1.2, count).astype(np.float32),
                        rng.uniform(0.001, 10, count).astype(np.float32)])
    out_g = ga.empty(len(x) * 8, np.float32)
    module.get_function("probe_math")(np.int32(len(x)), ga.to_gpu(x), ga.to_gpu(y), out_g,
                                      block=(block, 1, 1), grid=grid(len(x)))
    data.update(math_x=x, math_y=y, math_out=out_g.get().reshape(-1, 8))

    v = np.concatenate([_vec(rng, count) * rng.uniform(0.1, 10, (count, 1)).astype(np.float32),
                        np.array([[0, 0, 0], [1, 0, 0], [0, -0.0, 1], [1e-20, 0, 0], [1e20, 1e20, 0],
                                  [np.nan, 0, 1], [3e-39, 0, 0]], np.float32)]).astype(np.float32)
    o = ga.empty(len(v) * 3, np.float32)
    module.get_function("probe_normalize")(np.int32(len(v)), ga.to_gpu(v), o, block=(block, 1, 1),
                                           grid=grid(len(v)))
    data.update(norm_in=v, norm_out=o.get().reshape(-1, 3))

    nw, start, step = 188, np.float32(60.0), np.float32(5.0)
    table = np.concatenate([rng.uniform(0.1, 10.0, nw), [123.456]]).astype(np.float32)  # padding word
    wl = np.concatenate([np.array([60.0, 995.0, 994.9999, 995.0001, 59.9999, 400.0, 402.5, 61.0, 0.0,
                                   np.nan, np.inf], np.float32),
                         rng.uniform(50.0, 1010.0, count).astype(np.float32),
                         (60.0 + 5.0 * rng.integers(0, 188, count)).astype(np.float32)])
    o = ga.empty(len(wl), np.float32)
    module.get_function("probe_interp")(np.int32(len(wl)), ga.to_gpu(wl), ga.to_gpu(table), np.int32(nw),
                                        start, step, o, block=(block, 1, 1), grid=grid(len(wl)))
    data.update(interp_x=wl, interp_table=table, interp_out=o.get(), interp_grid=np.array([nw, start, step],
                                                                                           np.float32))

    xp = np.sort(rng.uniform(-5, 5, 49)).astype(np.float32)
    xp[10] = xp[11]  # repeated abscissa
    fp = np.cumsum(rng.uniform(0, 1, 49)).astype(np.float32)
    xs = np.concatenate([xp, rng.uniform(-6, 6, count).astype(np.float32), [np.nan]]).astype(np.float32)
    o = ga.empty(len(xs), np.float32)
    module.get_function("probe_interp_nonuniform")(np.int32(len(xs)), ga.to_gpu(xs), ga.to_gpu(xp), ga.to_gpu(fp),
                                                   np.int32(len(xp)), o, block=(block, 1, 1), grid=grid(len(xs)))
    data.update(interpn_x=xs, interpn_xp=xp, interpn_fp=fp, interpn_out=o.get())

    angles = np.radians(np.linspace(0, 90, 7)).astype(np.float32)
    xa = np.concatenate([angles, rng.uniform(-0.1, 1.7, count).astype(np.float32)]).astype(np.float32)
    o = ga.empty(len(xa), np.float32)
    module.get_function("probe_interp_idx")(np.int32(len(xa)), ga.to_gpu(xa), ga.to_gpu(angles),
                                            np.int32(len(angles)), o, block=(block, 1, 1), grid=grid(len(xa)))
    data.update(idx_x=xa, idx_xp=angles, idx_out=o.get())

    cdf = np.clip(np.linspace(-0.2, 1.3, 188), 0, 1).astype(np.float32)
    o = ga.empty(count, np.float32)
    states, counts = run_states("probe_sample_cdf", count, ga.to_gpu(cdf), np.int32(len(cdf)), np.float32(60.0),
                                np.float32(5.0), o)
    data.update(cdf_states=states, cdf_counts=counts, cdf_table=cdf, cdf_out=o.get())

    # triangles: random, plus rays through vertices/edges and parallel rays
    tri = rng.uniform(-1, 1, (count, 9)).astype(np.float32)
    orig = rng.uniform(-3, 3, (count, 3)).astype(np.float32)
    target = (tri[:, 0:3] * 0.3 + tri[:, 3:6] * 0.3 + tri[:, 6:9] * 0.4).astype(np.float32)
    k = np.arange(count) % 8
    target[k == 1] = tri[k == 1, 0:3]
    target[k == 2] = 0.5 * (tri[k == 2, 0:3] + tri[k == 2, 3:6])
    dirs = target - orig
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    dirs = dirs.astype(np.float32)
    par = k == 3
    e1 = tri[par, 3:6] - tri[par, 0:3]
    dirs[par] = (e1 / np.linalg.norm(e1, axis=1)[:, None]).astype(np.float32)
    o = ga.empty(count * 2, np.float32)
    module.get_function("probe_triangle")(np.int32(count), ga.to_gpu(tri), ga.to_gpu(orig), ga.to_gpu(dirs), o,
                                          block=(block, 1, 1), grid=grid(count))
    data.update(tri_v=tri, tri_o=orig, tri_d=dirs, tri_out=o.get().reshape(-1, 2))

    o = ga.empty(count * 3, np.float32)
    states, counts = run_states("probe_sphere", count, o)
    data.update(sphere_states=states, sphere_counts=counts, sphere_out=o.get().reshape(-1, 3))

    d = _vec(rng, count)
    p = np.cross(d, _vec(rng, count)).astype(np.float32)
    p /= np.linalg.norm(p, axis=1)[:, None]
    p = p.astype(np.float32)
    p[:16] = [[0, 0, 1]] * 8 + [[0, 0, -1]] * 8  # sin_axis_theta == 0
    od = ga.empty(count * 3, np.float32)
    op = ga.empty(count * 3, np.float32)
    states, counts = run_states("probe_rayleigh", count, ga.to_gpu(d), ga.to_gpu(p), od, op)
    data.update(ray_states=states, ray_counts=counts, ray_dir=d, ray_pol=p, ray_od=od.get().reshape(-1, 3),
                ray_op=op.get().reshape(-1, 3))

    nrm = _vec(rng, count)
    d = _vec(rng, count)
    flip = np.einsum("ij,ij->i", d, nrm) > 0
    d[flip] = -d[flip]
    kk = np.arange(count) % 8
    d[kk == 1] = -nrm[kk == 1]  # normal incidence
    grazing = kk == 2
    t = np.cross(nrm[grazing], _vec(rng, int(grazing.sum())))
    t /= np.linalg.norm(t, axis=1)[:, None]
    d[grazing] = (t - 1e-4 * nrm[grazing]).astype(np.float32)
    d = (d / np.linalg.norm(d, axis=1)[:, None]).astype(np.float32)
    p = np.cross(d, _vec(rng, count))
    p = (p / np.linalg.norm(p, axis=1)[:, None]).astype(np.float32)
    p[kk == 3] = d[kk == 3]
    n1 = rng.uniform(1.0, 1.8, count).astype(np.float32)
    n2 = rng.uniform(1.0, 1.8, count).astype(np.float32)
    n2[kk == 4] = n1[kk == 4]
    # critical angle: sin(theta_i) = n2/n1 with n1 > n2
    crit = kk == 5
    n1[crit], n2[crit] = 1.5, 1.2
    ang = np.arcsin(1.2 / 1.5)
    tt = np.cross(nrm[crit], _vec(rng, int(crit.sum())))
    tt /= np.linalg.norm(tt, axis=1)[:, None]
    d[crit] = (np.sin(ang) * tt - np.cos(ang) * nrm[crit]).astype(np.float32)
    od = ga.empty(count * 3, np.float32)
    op = ga.empty(count * 3, np.float32)
    hist = ga.empty(count, np.uint32)
    states, counts = run_states("probe_fresnel", count, ga.to_gpu(d), ga.to_gpu(p), ga.to_gpu(nrm),
                                ga.to_gpu(n1), ga.to_gpu(n2), od, op, hist)
    data.update(fr_states=states, fr_counts=counts, fr_dir=d, fr_pol=p, fr_normal=nrm, fr_n1=n1, fr_n2=n2,
                fr_od=od.get().reshape(-1, 3), fr_op=op.get().reshape(-1, 3), fr_hist=hist.get())

    od = ga.empty(count * 3, np.float32)
    module.get_function("probe_specular")(np.int32(count), ga.to_gpu(d), ga.to_gpu(nrm), od, block=(block, 1, 1),
                                          grid=grid(count))
    data.update(spec_dir=d, spec_normal=nrm, spec_out=od.get().reshape(-1, 3))

    axisn = nrm.copy()
    axisn[:6] = np.concatenate([np.eye(3), -np.eye(3)]).astype(np.float32)
    od = ga.empty(count * 3, np.float32)
    op = ga.empty(count * 3, np.float32)
    states, counts = run_states("probe_diffuse", count, ga.to_gpu(axisn), od, op)
    data.update(dif_states=states, dif_counts=counts, dif_normal=axisn, dif_od=od.get().reshape(-1, 3),
                dif_op=op.get().reshape(-1, 3))

    lo = rng.integers(0, 60000, (count, 3))
    hi = lo + rng.integers(0, 5000, (count, 3))
    hi = np.minimum(hi, 65535)
    nodes = np.zeros((count, 4), np.uint32)
    nodes[:, :3] = (lo | (hi << 16)).astype(np.uint32)
    bo = rng.uniform(-50, 50, (count, 3)).astype(np.float32)
    bd = _vec(rng, count)
    bd[:32] = np.concatenate([np.eye(3), -np.eye(3)] * 6)[:32].astype(np.float32)
    bd[32:40] = [0.0, -0.0, 1.0]
    o = ga.empty(count * 2, np.float32)
    world = np.float32([-40.0, -30.0, -20.0])
    scale = np.float32(0.0015)
    module.get_function("probe_box")(np.int32(count), ga.to_gpu(nodes), ga.to_gpu(bo), ga.to_gpu(bd),
                                     world[0], world[1], world[2], scale, o, block=(block, 1, 1), grid=grid(count))
    data.update(box_nodes=nodes, box_o=bo, box_d=bd, box_world=world, box_scale=np.float32([scale]),
                box_out=o.get().reshape(-1, 2))
    np.savez_compressed(out, **data)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m chroma.triton.legacy.probes")
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--count", type=int, default=2048)
    g.add_argument("--headers", default=None, help="original chroma/cuda directory (default: chroma.cuda.srcdir)")
    args = parser.parse_args(argv)
    if args.command == "generate":
        headers = args.headers
        if headers is None:
            from chroma.cuda import srcdir
            headers = srcdir
        generate(args.out, headers, count=args.count)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
