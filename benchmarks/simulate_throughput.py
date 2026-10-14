"""Throughput of ``chroma.sim.Simulation`` on a lookup-table-style workload.

``count`` photons start at one point (by default a LUT voxel of chroma-lar's
reflect3wires detector) with isotropic directions; one ``simulate()`` call
keeps flat hits. The script uses whichever backend ``chroma.sim`` selects, so
running it in a PyCUDA environment and in a Triton one compares the two::

    python benchmarks/simulate_throughput.py 30000000 --wavelength 128
    python benchmarks/simulate_throughput.py 10000000 --weighted

Needs chroma-lar (for the detector) on ``PYTHONPATH``.
"""

import argparse
import time

import numpy as np


def synchronize():
    from chroma.backend import backend_name

    if backend_name() == "triton":
        import torch

        torch.cuda.synchronize()
    else:
        import pycuda.driver as cuda

        cuda.Context.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("count", type=int)
    parser.add_argument("--wavelength", type=float, default=128.0)
    parser.add_argument("--weighted", action="store_true", help="use_weights=True (generate_lut.py style)")
    parser.add_argument("--position", type=float, nargs=3, default=(-450.0, 60.0, -120.0))
    parser.add_argument("--detector", default="detector_config_reflect_reflect3wires")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    from chroma.backend import backend_name
    from chroma.event import Photons
    from chroma.sim import Simulation
    from chroma_lar.geometry.config_loader import build_detector_from_config

    detector = build_detector_from_config(args.detector)
    n = args.count
    rng = np.random.default_rng(3)
    pos = np.tile(np.asarray(args.position, np.float32), (n, 1))
    theta = np.arccos(2 * rng.random(n) - 1)
    phi = rng.uniform(0, 2 * np.pi, n)
    direction = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], 1).astype(np.float32)
    pol = np.zeros_like(direction)
    pol[:, 0] = 1
    photons = Photons(pos=pos, dir=direction, pol=pol, t=np.zeros(n, np.float32),
                      wavelengths=np.full(n, args.wavelength, np.float32))
    started = time.time()
    sim = Simulation(detector, seed=5)
    print("backend %s: Simulation() %.1f s" % (backend_name(), time.time() - started), flush=True)
    for repeat in range(args.repeats):
        started = time.time()
        event = next(sim.simulate([photons], keep_flat_hits=True, max_steps=1000, use_weights=args.weighted))
        synchronize()
        elapsed = time.time() - started
        weights = event.flat_hits.weights
        ok = np.isfinite(weights) & (weights > 0)
        print("run %d: %.2f s  %.2fM photons/s  flat hits %d  detected weight per photon %.5f"
              % (repeat, elapsed, n / elapsed / 1e6, len(weights), weights[ok].sum() / n), flush=True)


if __name__ == "__main__":
    main()
