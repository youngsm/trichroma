"""Statistical comparison of the CUDA and Triton ``chroma.sim.Simulation`` backends.

Each ``run`` executes one fixture through the public Simulation API in the
current process (select the backend with CHROMA_BACKEND); ``compare`` tests two
result files:

* every history bit, terminal classes and wire terminations: two-proportion
  z statistics (fail above 6 standard errors);
* hits per channel: z statistic per channel with at least 25 expected hits;
* hit times, hit wavelengths and final photon times: two-sample
  Kolmogorov-Smirnov distance against the DKW bound at alpha = 1e-6;
* DAQ channel hit fractions and charges.

Both backends see the same photons. They use different random streams and
the Triton production engine applies documented numerical fixes, so only
statistical agreement is expected. For bitwise checks see trichroma.tape.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BITS = {"no_hit": 1, "bulk_absorb": 2, "surface_detect": 4, "surface_absorb": 8, "rayleigh_scatter": 16,
        "reflect_diffuse": 32, "reflect_specular": 64, "surface_reemit": 128, "surface_transmit": 256,
        "bulk_reemit": 512}
TERMINAL = 1 | 2 | 4 | 8


def _surface(name, model=0, **values):
    from chroma.geometry import Surface

    s = Surface(name, model=model)
    for key, value in values.items():
        s.set(key, value)
    return s


def _isotropic(rng, count, center, spread, wavelength):
    positions = np.asarray(center, float) + rng.uniform(-spread, spread, (count, 3))
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    pol = np.cross(direction, rng.normal(size=(count, 3)))
    pol /= np.linalg.norm(pol, axis=1)[:, None]
    wl = rng.uniform(*wavelength, count) if isinstance(wavelength, tuple) else np.full(count, wavelength)
    return positions, direction, pol, wl


def build_fixture(name, count, seed):
    """Return (geometry, Photons) for a named fixture; identical in both backends."""
    from chroma.detector import Detector
    from chroma.event import Photons
    from chroma.geometry import Material, Solid
    from chroma.loader import create_geometry_from_obj
    from chroma.make import box
    from chroma.demo.optics import water, glass, vacuum, black_surface, r7081hqe_photocathode

    rng = np.random.default_rng(seed)
    if name in ("wirebox", "wls_box"):
        det = Detector(water)
        if name == "wirebox":
            det.add_solid(Solid(box(4000, 4000, 4000), water, vacuum, surface=black_surface))
            coat = r7081hqe_photocathode
        else:
            wall = _surface("mixed-wall", absorb=0.2, reflect_specular=0.5, reflect_diffuse=0.3)
            det.add_solid(Solid(box(4000, 4000, 4000), water, vacuum, surface=wall))
            coat = _surface("wls-coat", model=2, absorb=0.6, reemit=0.8, reflect_specular=0.1, reflect_diffuse=0.1,
                            reemission_cdf=np.clip((np.arange(60, 1000, 5) - 420.0) / 60.0, 0.0, 1.0))
        cathode = r7081hqe_photocathode
        for x in (-1000.0, 0.0, 1000.0):
            for y in (-1000.0, 1000.0):
                det.add_pmt(Solid(box(300, 300, 20), glass, water, surface=cathode), displacement=(x, y, 1500))
                det.add_solid(Solid(box(320, 320, 5), glass, water, surface=coat), displacement=(x, y, 1480))
        det.wireplanes = [dict(origin=np.array([0., 0., 500.]), u=np.array([1., 0, 0]), v=np.array([0, 1., 0]),
                               pitch=5.0, radius=0.5, umin=-1500., umax=1500., vmin=-1500., vmax=1500., v0=0.0,
                               surface=black_surface, material_inner=vacuum, material_outer=water),
                          dict(origin=np.array([0., 0., 700.]), u=np.array([1., 1., 0]), v=np.array([-1., 1., 0.3]),
                               pitch=4.0, radius=0.2, umin=-1500., umax=1500., vmin=-1500., vmax=1500., v0=1.0,
                               surface=_surface("wire-mirror", reflect_specular=0.9, absorb=0.1),
                               material_inner=vacuum, material_outer=water)]
        geometry = create_geometry_from_obj(det, read_bvh_cache=False, update_bvh_cache=False)
        pos, d, pol, wl = _isotropic(rng, count, (0, 0, 0), 50.0, (350.0, 500.0))
        return geometry, Photons(pos, d, pol, wl, t=np.zeros(count))
    if name == "reemit_box":
        grid = np.arange(60, 1000, 5).astype(float)
        scint = Material("scintillator")
        scint.set("refractive_index", 1.5)
        scint.set("absorption_length", np.where(grid < 420, 300.0, 5000.0))
        scint.set("scattering_length", 800.0)
        scint.comp_reemission_prob = [np.stack([grid, np.where(grid < 420, 0.7, 0.0)], 1).astype(np.float32),
                                      np.stack([grid, np.full_like(grid, 0.4)], 1).astype(np.float32)]
        scint.comp_absorption_length = [np.stack([grid, np.where(grid < 420, 500.0, 1e9)], 1).astype(np.float32),
                                        np.stack([grid, np.where(grid < 420, 750.0, 1e9)], 1).astype(np.float32)]
        wcdf = np.clip((grid - 420.0) / 80.0, 0.0, 1.0)
        scint.comp_reemission_wvl_cdf = [np.stack([grid, wcdf], 1).astype(np.float32)] * 2
        times = np.arange(0, 1000, 0.05)
        tcdf = 1.0 - np.exp(-times / 5.0)
        tcdf[-1] = 1.0
        scint.comp_reemission_times = [times, times]
        scint.comp_reemission_time_cdf = [np.stack([times, tcdf], 1).astype(np.float32)] * 2
        det = Detector(scint)
        det.add_solid(Solid(box(3000, 3000, 3000), scint, vacuum, surface=_surface("wall", absorb=0.5, reflect_diffuse=0.5)))
        for x in (-800.0, 800.0):
            det.add_pmt(Solid(box(400, 400, 20), glass, scint, surface=r7081hqe_photocathode), displacement=(x, 0, 1200))
        geometry = create_geometry_from_obj(det, read_bvh_cache=False, update_bvh_cache=False)
        pos, d, pol, wl = _isotropic(rng, count, (0, 0, 0), 100.0, (360.0, 400.0))
        return geometry, Photons(pos, d, pol, wl, t=np.zeros(count))
    if name.startswith("lar_lut"):
        # The installed chroma-lar scripts/generate_lut.py setup. Its literal
        # build_detector(**get_config()) now fails on the config's
        # detector_type key; build_detector_from_config removes that key.
        from chroma_lar.geometry.config_loader import build_detector_from_config

        geometry = build_detector_from_config("detector_config_reflect_reflect3wires")
        # x = 0 is inside the steel cathode; LUT voxels in the drift volumes.
        voxel = {"lar_lut": (-450.0, 60.0, -120.0), "lar_lut_corner": (450.0, -420.0, 390.0),
                 "lar_lut_450": (-450.0, 60.0, -120.0)}[name]
        wavelength = 450.0 if name.endswith("_450") else 128.0
        np.random.seed(seed)
        positions = np.tile(voxel, (count, 1)).astype(np.float32)
        theta = np.arccos(2 * np.random.random(count) - 1)
        phi = np.random.uniform(0, 2 * np.pi, count)
        direction = np.zeros((count, 3), dtype=np.float32)
        direction[:, 0] = np.sin(theta) * np.cos(phi)
        direction[:, 1] = np.sin(theta) * np.sin(phi)
        direction[:, 2] = np.cos(theta)
        pol = np.zeros_like(direction)
        pol[:, 0] = 1
        return geometry, Photons(pos=positions, dir=direction, pol=pol, t=np.zeros(count, np.float32),
                                 wavelengths=np.full(count, wavelength, np.float32))
    if name.startswith("reflect3wires") or name.startswith("pixel"):
        from chroma_lar.geometry.config_loader import build_detector_from_config

        pixel = name.startswith("pixel")
        vuv = name.endswith("_vuv")
        config = "detector_config_pixel" if pixel else "detector_config_reflect_reflect3wires"
        options = {"flatten": False}
        if not pixel:
            options["analytic_wires"] = True
        grid = np.arange(60, 1000, 5).astype(float)
        if vuv:
            coat = _surface("fixture-tpb", model=2, reemit=0.85, absorb=np.where(grid < 200.0, 1.0, 0.0),
                            reemission_cdf=np.clip((grid - 410.0) / 80.0, 0.0, 1.0))
            options["pmt_coating_surface"] = coat
        geometry = build_detector_from_config(config, **options)
        geometry = create_geometry_from_obj(geometry, read_bvh_cache=True, update_bvh_cache=True)
        pos, d, pol, wl = _isotropic(rng, count, (0, 0, 0), 15.0, (120.0, 140.0) if vuv else (390.0, 495.0))
        pos[:, 0] += np.where(np.arange(count) % 2, 1000.0, -1000.0)
        return geometry, Photons(pos, d, pol, wl, t=rng.uniform(0.0, 1500.0, count))
    raise ValueError("unknown fixture %r" % name)


def _sync():
    if os.environ.get("CHROMA_BACKEND", "cuda") == "triton":
        import torch

        torch.cuda.synchronize()
    else:
        import pycuda.driver as cuda

        cuda.Context.synchronize()


def run(args):
    from chroma.sim import Simulation

    geometry, photons = build_fixture(args.fixture, args.photons, args.seed)
    begin = time.time()
    sim = Simulation(geometry, seed=args.sim_seed, use_packed=args.use_packed)
    built = time.time()
    options = dict(keep_photons_end=True, run_daq=True, max_steps=args.max_steps,
                   use_weights=args.use_weights, photons_per_batch=args.photons_per_batch)
    if args.warmup:
        list(sim.simulate([photons], **options))
        _sync()
    start = time.time()
    events = list(sim.simulate([photons], **options))
    _sync()
    done = time.time()
    built = start if args.warmup else built
    ev = events[0]
    end, hits, ch = ev.photons_end, ev.flat_hits, ev.channels
    out = dict(
        backend=np.array(os.environ.get("CHROMA_BACKEND", "cuda")),
        flags=end.flags.astype(np.uint32), last=end.last_hit_triangles.astype(np.int32),
        t=end.t.astype(np.float32), wl=end.wavelengths.astype(np.float32), weights=end.weights.astype(np.float32),
        hit_channel=hits.channel.astype(np.int32), hit_t=hits.t.astype(np.float32),
        hit_wl=hits.wavelengths.astype(np.float32), hit_w=hits.weights.astype(np.float32),
        daq_hit=ch.hit.astype(bool), daq_t=ch.t.astype(np.float32), daq_q=ch.q.astype(np.float32),
        construct_seconds=np.array(built - begin), simulate_seconds=np.array(done - built),
    )
    np.savez(args.output, **out)
    print(json.dumps({"backend": str(out["backend"]), "fixture": args.fixture, "photons": args.photons,
                      "construct_s": built - begin, "simulate_s": done - built,
                      "detected": int(np.count_nonzero(out["flags"] & 4)), "hits": int(len(out["hit_t"]))}))


def _z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-300))
    return (p1 - p2) / se if (k1 + k2) > 0 else 0.0


def _ks(a, b, alpha=1e-6):
    if len(a) < 20 or len(b) < 20:
        return None
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    distance = float(np.max(np.abs(fa - fb)))
    bound = float(np.sqrt(np.log(2 / alpha) / 2) * (np.sqrt(1 / len(a)) + np.sqrt(1 / len(b))))
    return {"distance": distance, "bound": bound, "pass": distance <= bound}


def compare(args):
    a, b = np.load(args.a), np.load(args.b)
    n1, n2 = len(a["flags"]), len(b["flags"])
    rows, failures = [], []

    def frac(name, k1, k2):
        z = _z(k1, n1, k2, n2)
        rows.append({"quantity": name, "a": k1 / n1, "b": k2 / n2, "z": z})
        if abs(z) > 6:
            failures.append(name)

    for name, bit in BITS.items():
        frac(name, int(np.count_nonzero(a["flags"] & bit)), int(np.count_nonzero(b["flags"] & bit)))
    frac("unfinished", int(np.count_nonzero((a["flags"] & TERMINAL) == 0)), int(np.count_nonzero((b["flags"] & TERMINAL) == 0)))
    frac("ends_on_wire", int(np.count_nonzero(a["last"] == -2)), int(np.count_nonzero(b["last"] == -2)))
    nch = int(max(a["hit_channel"].max(initial=-1), b["hit_channel"].max(initial=-1)) + 1)
    ca = np.bincount(a["hit_channel"], minlength=nch)
    cb = np.bincount(b["hit_channel"], minlength=nch)
    channel_z = []
    for c in range(nch):
        if (ca[c] + cb[c]) / 2 >= 25:
            channel_z.append(_z(int(ca[c]), n1, int(cb[c]), n2))
    max_channel_z = float(np.max(np.abs(channel_z))) if channel_z else 0.0
    if max_channel_z > 6:
        failures.append("channel_hits")
    # Weighted efficiency per channel (implicit capture): sum(w)/N with
    # variance sum(w^2)/N^2 per backend.
    wa = np.bincount(a["hit_channel"], weights=a["hit_w"], minlength=nch)
    wb = np.bincount(b["hit_channel"], weights=b["hit_w"], minlength=nch)
    va = np.bincount(a["hit_channel"], weights=a["hit_w"].astype(float) ** 2, minlength=nch)
    vb = np.bincount(b["hit_channel"], weights=b["hit_w"].astype(float) ** 2, minlength=nch)
    tested = (ca >= 25) & (cb >= 25)
    wz = (wa / n1 - wb / n2)[tested] / np.sqrt(va[tested] / n1**2 + vb[tested] / n2**2)
    max_weighted_z = float(np.max(np.abs(wz))) if wz.size else 0.0
    if max_weighted_z > 6:
        failures.append("channel_weighted_efficiency")
    total_eff = (float(wa.sum() / n1), float(wb.sum() / n2))
    dists = {"hit_t": _ks(a["hit_t"], b["hit_t"]), "hit_wl": _ks(a["hit_wl"], b["hit_wl"]),
             "final_t": _ks(a["t"], b["t"])}
    for key, value in dists.items():
        if value is not None and not value["pass"]:
            failures.append(key)
    daq = {"a_hit_fraction": float(a["daq_hit"].mean()), "b_hit_fraction": float(b["daq_hit"].mean()),
           "a_total_q": float(a["daq_q"].sum()), "b_total_q": float(b["daq_q"].sum())}
    report = {"a": str(args.a), "b": str(args.b), "photons": [n1, n2], "fractions": rows,
              "channels_tested": len(channel_z), "max_channel_z": max_channel_z, "distributions": dists,
              "weighted_channels_tested": int(tested.sum()), "max_weighted_z": max_weighted_z,
              "total_weighted_efficiency": total_eff,
              "daq": daq, "failures": failures,
              "timing_s": {"a": float(a["simulate_seconds"]), "b": float(b["simulate_seconds"])}}
    for r in rows:
        mark = "  FAIL" if abs(r["z"]) > 6 else ""
        print("%-18s %10.6f %10.6f  z=%7.2f%s" % (r["quantity"], r["a"], r["b"], r["z"], mark))
    print("channels tested %d, max |z| %.2f" % (len(channel_z), max_channel_z))
    print("weighted channels tested %d, max |z| %.2f, total efficiency %.6g vs %.6g" % (int(tested.sum()), max_weighted_z, *total_eff))
    for key, value in dists.items():
        print("%-10s %s" % (key, value))
    print("daq", daq)
    print("FAILURES:" if failures else "PASS", failures)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fixture", required=True)
    r.add_argument("--photons", type=int, default=200000)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--sim-seed", type=int, default=1234)
    r.add_argument("--max-steps", type=int, default=1000)
    r.add_argument("--use-weights", action="store_true")
    r.add_argument("--use-packed", action="store_true")
    r.add_argument("--warmup", action="store_true", help="run once untimed first (JIT/allocation)")
    r.add_argument("--photons-per-batch", type=int, default=1000000)
    r.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        sys.exit(compare(args))


if __name__ == "__main__":
    main()
