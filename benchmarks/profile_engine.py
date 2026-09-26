"""Profile the Triton production engine on the installed reflect3wires detector.

Scenarios: a photon bomb (isotropic, 30 mm voxel) at a chosen position and
wavelength, with or without use_weights. Reports warm wall time, photons/s,
and per-kernel CUDA time from torch.profiler.
"""

import argparse
import hashlib
import json
import os
import time

import numpy as np

os.environ.setdefault("CHROMA_BACKEND", "triton")


def bomb(count, position, wavelength, seed):
    rng = np.random.default_rng(seed)
    pos = np.asarray(position, np.float32) + rng.uniform(-15, 15, (count, 3)).astype(np.float32)
    d = rng.normal(size=(count, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    p = np.cross(d, rng.normal(size=(count, 3)))
    p /= np.linalg.norm(p, axis=1)[:, None]
    return pos, d.astype(np.float32), p.astype(np.float32), np.full(count, wavelength, np.float32)


def main():
    import torch
    from chroma_lar.geometry.config_loader import build_detector_from_config
    from trichroma.engine.api import DevicePhotons
    from trichroma.engine.core import ProductionEngine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photons", type=int, default=1_000_000)
    parser.add_argument("--wavelength", type=float, default=450.0)
    parser.add_argument("--position", type=float, nargs=3, default=[-450.0, 60.0, -120.0])
    parser.add_argument("--use-weights", action="store_true")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-grid", action="store_true")
    parser.add_argument("--config", default="detector_config_reflect_reflect3wires")
    args = parser.parse_args()
    if args.no_grid:
        os.environ["CHROMA_TRITON_GRID"] = "0"
    geometry = build_detector_from_config(args.config)
    t0 = time.time()
    engine = ProductionEngine(geometry, seed=1234, device="cuda")
    print("engine %.1fs grid %s" % (time.time() - t0, None if engine.grid is None else
                                     "certified %.3f" % engine.grid.certified_fraction), flush=True)
    dev = engine.device
    pos, d, p, wl = bomb(args.photons, args.position, args.wavelength, 7)

    def make(seed_offset):
        ph = DevicePhotons.empty(args.photons, dev)
        ph.pos.copy_(torch.from_numpy(pos))
        ph.dir.copy_(torch.from_numpy(d))
        ph.pol.copy_(torch.from_numpy(p))
        ph.wavelengths.copy_(torch.from_numpy(wl))
        ph.t.zero_()
        ph.ids.copy_(torch.arange(args.photons, device=dev) + seed_offset * args.photons)
        return ph

    rows = []
    for r in range(args.repeats + 1):
        ph = make(r)
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.propagate(ph, max_steps=args.max_steps, use_weights=args.use_weights, history=args.history)
        torch.cuda.synchronize()
        dt = time.perf_counter() - start
        flags = ph.flags.cpu().numpy().view(np.uint32)
        digest = hashlib.sha256()
        for field in (ph.pos, ph.dir, ph.pol, ph.t, ph.wavelengths, ph.flags, ph.last_hit_triangles, ph.weights):
            digest.update(field.cpu().numpy().tobytes())
        row = {"repeat": r, "seconds": dt, "photons_per_second": args.photons / dt,
               "detected": int(np.count_nonzero(flags & 4)), "unfinished": int(np.count_nonzero((flags & 15) == 0)),
               "sha256": digest.hexdigest()[:16]}
        print(json.dumps(row), flush=True)
        if r:
            rows.append(row)
    print(json.dumps({"median_seconds": float(np.median([r["seconds"] for r in rows])),
                      "median_photons_per_second": float(np.median([r["photons_per_second"] for r in rows]))}))
    if args.profile:
        from torch.profiler import ProfilerActivity, profile

        ph = make(99)
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            engine.propagate(ph, max_steps=args.max_steps, use_weights=args.use_weights, history=args.history)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


if __name__ == "__main__":
    main()
