"""Transport throughput versus the number of photons per launch (the README's scaling plot).

For each count, photons from an isotropic point source are drawn on the GPU and the production engine
runs them to completion (``engine.propagate``: unweighted, ``max_steps=1000``, default settings). The
time is the best of several launches after a warm-up launch, so compilation is excluded::

    python benchmarks/throughput_scaling.py out.json 1e4 1e5 1e6 1e7 1e8
    python benchmarks/throughput_scaling.py out.json 1e6 1e7 --detector my_pkg.geometry:build \\
        --position 0 0 0 --wavelength 420

``--detector`` is a chroma-lar configuration name (default: reflect3wires, which needs chroma-lar on
``PYTHONPATH``) or ``module:function``, a function without arguments that returns a Chroma detector.
``benchmarks/plot_scaling.py`` draws the JSON files of several runs.
"""
import argparse
import importlib
import json
import time

import numpy as np


def load_detector(spec):
    if ":" in spec:
        module, name = spec.split(":", 1)
        return getattr(importlib.import_module(module), name)()
    from chroma_lar.geometry.config_loader import build_detector_from_config

    return build_detector_from_config(spec)


def point_source(n, position, wavelength, device):
    """Isotropic directions; polarization uniform in the plane perpendicular to the direction."""
    import torch
    from trichroma.engine.api import DevicePhotons

    gen = torch.Generator(device=device)
    gen.manual_seed(3)
    u = torch.rand((3, n), device=device, generator=gen)
    c = 2 * u[0] - 1
    s = torch.sqrt(torch.clamp(1 - c * c, min=0.0))
    p = 2 * np.pi * u[1]
    q = 2 * np.pi * u[2]
    ph = DevicePhotons.empty(n, device)
    ph.pos[:] = torch.tensor([float(x) for x in position], device=device)
    ph.dir[:, 0] = s * torch.cos(p)
    ph.dir[:, 1] = s * torch.sin(p)
    ph.dir[:, 2] = c
    ph.pol[:, 0] = torch.cos(q) * c * torch.cos(p) - torch.sin(q) * torch.sin(p)
    ph.pol[:, 1] = torch.cos(q) * c * torch.sin(p) + torch.sin(q) * torch.cos(p)
    ph.pol[:, 2] = -torch.cos(q) * s
    ph.wavelengths.fill_(float(wavelength))
    ph.t.zero_()
    ph.ids.copy_(torch.arange(n, dtype=torch.int64, device=device))
    return ph


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="JSON file for the results")
    parser.add_argument("counts", nargs="+", type=float, help="photons per launch")
    parser.add_argument("--detector", default="detector_config_reflect_reflect3wires")
    parser.add_argument("--position", type=float, nargs=3, default=(-450.0, 60.0, -120.0))
    parser.add_argument("--wavelength", type=float, default=128.0)
    args = parser.parse_args()

    import torch
    from chroma.sim import Simulation

    counts = [int(n) for n in args.counts]
    engine = Simulation(load_detector(args.detector), seed=5).engine
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name()
    engine.propagate(point_source(min(counts), args.position, args.wavelength, device), max_steps=1000)
    torch.cuda.synchronize()
    rows = []
    for n in counts:
        times = []
        for _ in range(7 if n <= 1_000_000 else 3):
            photons = point_source(n, args.position, args.wavelength, device)
            torch.cuda.synchronize()
            started = time.time()
            engine.propagate(photons, max_steps=1000)
            torch.cuda.synchronize()
            times.append(time.time() - started)
        best = min(times)
        steps = float(engine.last_steps.float().mean())
        rows.append(dict(n=n, seconds=best, photons_per_s=n / best, steps_per_photon=steps))
        print("%s N=%.0e: %.2fM photons/s  %.3fG steps/s  %.2f steps/photon" % (
            gpu, n, n / best / 1e6, n * steps / best / 1e9, steps), flush=True)
        del photons
        torch.cuda.empty_cache()
    with open(args.output, "w") as f:
        json.dump(dict(detector=args.detector, gpu=gpu, wavelength=args.wavelength, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
