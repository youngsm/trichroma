"""High-occupancy transport: original CUDA Chroma versus the general Triton mesh engine.

Both backends propagate identical persisted photons through the same flattened
Chroma geometry and the same Chroma-built BVH. Run each stage in its own
environment:

  prepare        (Triton env)  build the fixture detector and photon files
  cuda-geometry  (PyCUDA env)  flatten + build the BVH with the original loader
  triton         (Triton env)  time SpectralSimulation on the shared geometry/BVH
  cuda           (PyCUDA env)  time GPUPhotons.propagate on the shared geometry/BVH

Transport time starts with photon states resident on the device and ends at
device synchronization after the last step; queue allocation and host
scheduling are included. "Prepared event" adds the host-to-device upload and
the download of every final photon state. Geometry construction, BVH build,
compilation and file I/O are excluded. RNG algorithms, numerical edge
handling and physics details (phase vs group velocity) differ between the
backends, so identical inputs do not imply identical trajectories; terminal
outcome fractions are recorded so the work performed can be compared.
"""

import argparse
import gc
import hashlib
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

FIELDS = ("pos", "dir", "pol", "wavelengths", "t")
# Original Chroma history bits (chroma/cuda/photon.h).
BITS = {
    "no_hit": 1 << 0,
    "bulk_absorb": 1 << 1,
    "surface_detect": 1 << 2,
    "surface_absorb": 1 << 3,
    "rayleigh_scatter": 1 << 4,
    "reflect_diffuse": 1 << 5,
    "reflect_specular": 1 << 6,
    "surface_reemit": 1 << 7,
    "surface_transmit": 1 << 8,
    "bulk_reemit": 1 << 9,
    "nan_abort": 1 << 31,
}
TERMINAL = BITS["no_hit"] | BITS["bulk_absorb"] | BITS["surface_detect"] | BITS["surface_absorb"] | BITS["nan_abort"]


def digest(arrays):
    h = hashlib.sha256()
    for name in FIELDS:
        value = np.ascontiguousarray(arrays[name])
        h.update(name.encode())
        h.update(str(value.shape).encode())
        h.update(value.dtype.str.encode())
        h.update(value.tobytes())
    return h.hexdigest()


def write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def outcome_summary(flags):
    flags = np.asarray(flags).astype(np.uint32)
    n = len(flags)
    summary = {name: int(np.count_nonzero(flags & np.uint32(bit))) for name, bit in BITS.items()}
    summary["unfinished"] = int(np.count_nonzero((flags & np.uint32(TERMINAL)) == 0))
    summary["fractions"] = {k: (v / n if n else 0.0) for k, v in summary.items() if k != "fractions"}
    return summary


def photon_file(directory, count, seed):
    return Path(directory) / "photons" / f"{count}_{seed}.npz"


def load_photons(directory, manifest, count, seed):
    path = photon_file(directory, count, seed)
    with np.load(path) as data:
        arrays = {name: np.ascontiguousarray(data[name]) for name in FIELDS}
    if digest(arrays) != manifest["photons"][path.name]:
        raise RuntimeError(f"photon file {path} does not match its manifest hash")
    return arrays


# ----------------------------------------------------------------------------
# prepare (Triton environment; uses the fixture builders in this repository)


def prepare(args):
    directory = args.directory
    (directory / "photons").mkdir(parents=True, exist_ok=True)
    manifest = {"fixture": args.fixture, "counts": args.counts, "seeds": args.seeds, "photons": {}}
    if args.fixture == "theia":
        from chroma.triton.examples.theia import build_theia, cherenkov_photons

        fixture = build_theia(25500.0, coverage=0.81, diameter=508.0)
        detector = fixture.detector()
        manifest["geometry"] = {"channels": fixture.channel_count, "size_mm": 25500.0,
                                "coverage": 0.81, "diameter_mm": 508.0}
        manifest["source"] = "1 m beta=1 Cherenkov track along +z at the centre, 300-650 nm (chroma.triton.examples.theia)"

        def make(count, seed):
            p = cherenkov_photons(count, seed=seed)
            return {"pos": p.pos, "dir": p.dir, "pol": p.pol, "wavelengths": p.wavelengths, "t": p.t}
    else:
        raise ValueError(args.fixture)
    with open(directory / "detector.pkl", "wb") as f:
        pickle.dump(detector, f, protocol=4)
    for count in args.counts:
        for seed in args.seeds:
            arrays = make(count, seed)
            arrays = {
                "pos": np.ascontiguousarray(arrays["pos"], np.float32),
                "dir": np.ascontiguousarray(arrays["dir"], np.float32),
                "pol": np.ascontiguousarray(arrays["pol"], np.float32),
                "wavelengths": np.ascontiguousarray(arrays["wavelengths"], np.float32),
                "t": np.ascontiguousarray(arrays["t"], np.float32),
            }
            path = photon_file(directory, count, seed)
            np.savez(path, **arrays)
            manifest["photons"][path.name] = digest(arrays)
            print(f"prepared {path.name}", flush=True)
    write_json(directory / "manifest.json", manifest)


# ----------------------------------------------------------------------------
# cuda-geometry (PyCUDA environment with the original Chroma)


def cuda_geometry(args):
    import chroma
    from chroma.loader import create_geometry_from_obj

    with open(args.directory / "detector.pkl", "rb") as f:
        detector = pickle.load(f)
    begin = time.perf_counter()
    geometry = create_geometry_from_obj(detector, read_bvh_cache=False, update_bvh_cache=False)
    seconds = time.perf_counter() - begin
    with open(args.directory / "flat.pkl", "wb") as f:
        pickle.dump(geometry, f, protocol=4)
    export_triton_geometry(args.directory, geometry)
    info = {
        "chroma": chroma.__file__,
        "triangles": int(len(geometry.mesh.triangles)),
        "vertices": int(len(geometry.mesh.vertices)),
        "bvh_nodes": int(len(geometry.bvh.nodes)),
        "flatten_and_bvh_seconds": seconds,
        "mesh_sha256": hashlib.sha256(np.ascontiguousarray(geometry.mesh.triangles).tobytes()
                                      + np.ascontiguousarray(geometry.mesh.vertices).tobytes()).hexdigest(),
        "bvh_sha256": hashlib.sha256(np.ascontiguousarray(geometry.bvh.nodes).tobytes()).hexdigest(),
    }
    write_json(args.directory / "flat.json", info)
    print(json.dumps(info), flush=True)


def export_triton_geometry(directory, geometry):
    """Store the Chroma BVH as plain arrays; its class imports PyCUDA on unpickling."""
    bvh = geometry.bvh
    np.savez(directory / "bvh.npz", nodes=np.asarray(bvh.nodes),
             world_origin=np.asarray(bvh.world_coords.world_origin, np.float32),
             world_scale=np.float32(bvh.world_coords.world_scale),
             layer_offsets=np.asarray(bvh.layer_offsets, np.int64))
    geometry.bvh = None
    try:
        with open(directory / "flat_triton.pkl", "wb") as f:
            pickle.dump(geometry, f, protocol=4)
    finally:
        geometry.bvh = bvh


def load_flat(directory, *, triton=False):
    if triton:
        from types import SimpleNamespace

        with open(directory / "flat_triton.pkl", "rb") as f:
            geometry = pickle.load(f)
        with np.load(directory / "bvh.npz") as data:
            geometry.bvh = SimpleNamespace(nodes=data["nodes"], world_origin=data["world_origin"],
                                           world_scale=np.float32(data["world_scale"]),
                                           layer_offsets=tuple(int(x) for x in data["layer_offsets"]))
    else:
        with open(directory / "flat.pkl", "rb") as f:
            geometry = pickle.load(f)
    info = json.loads((directory / "flat.json").read_text())
    mesh = hashlib.sha256(np.ascontiguousarray(geometry.mesh.triangles).tobytes()
                          + np.ascontiguousarray(geometry.mesh.vertices).tobytes()).hexdigest()
    if mesh != info["mesh_sha256"]:
        raise RuntimeError("flattened mesh differs from flat.json")
    if hashlib.sha256(np.ascontiguousarray(geometry.bvh.nodes).tobytes()).hexdigest() != info["bvh_sha256"]:
        raise RuntimeError("BVH differs from flat.json")
    return geometry, info


def base_report(args, backend, info):
    return {
        "backend": backend,
        "host": platform.node(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "argv": sys.argv,
        "executable": sys.executable,
        "geometry": info,
        "max_steps": args.max_steps,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": [],
    }


# ----------------------------------------------------------------------------
# triton (Triton environment)


def run_triton(args):
    import torch
    import triton
    from chroma.triton.spectral import SpectralScene, SpectralSimulation

    manifest = json.loads((args.directory / "manifest.json").read_text())
    geometry, info = load_flat(args.directory, triton=True)
    begin = time.perf_counter()
    scene = SpectralScene.compile(geometry, wavelengths=np.arange(300.0, 651.0, 5.0))
    tile = args.tile_size or max(args.counts)
    simulation = SpectralSimulation(scene, tile_size=tile)
    torch.cuda.synchronize()
    report = base_report(args, "triton general flattened-mesh SpectralSimulation", info)
    report.update(device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
                  tile_size=tile, compile_upload_seconds=time.perf_counter() - begin,
                  bvh_source=scene.host.bvh.source)
    output = args.output

    def run(count, seed):
        arrays = load_photons(args.directory, manifest, count, seed)
        from chroma.event import Photons

        photons = Photons(arrays["pos"], arrays["dir"], arrays["pol"], arrays["wavelengths"], arrays["t"])
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        start = time.perf_counter()
        result = simulation.simulate(photons, seed=seed, max_steps=args.max_steps, timings=timings)
        total = time.perf_counter() - start
        transport = sum(t["transport_seconds"] for t in timings)
        prepared = sum(t["transport_seconds"] + t["upload_allocation_seconds"] + t["download_result_seconds"]
                       for t in timings)
        row = {
            "photons": count, "seed": seed, "steps": int(result.steps),
            "transport_seconds": transport,
            "prepared_event_seconds": prepared,
            "simulate_call_seconds": total,
            "transport_photons_per_second": count / transport,
            "prepared_photons_per_second": count / prepared,
            "tiles": timings,
            "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "outcomes": outcome_summary(result.final_state["flags"]),
            "step_limited": int(result.step_limit_count),
        }
        print(json.dumps({k: v for k, v in row.items() if k not in ("tiles", "outcomes")}), flush=True)
        return row

    measure(args, report, run, output)


# ----------------------------------------------------------------------------
# cuda (PyCUDA environment with the original Chroma)


def run_cuda(args):
    import chroma
    from chroma import gpu
    from chroma.event import Photons
    from chroma.sim import Simulation
    import pycuda.driver as cuda

    manifest = json.loads((args.directory / "manifest.json").read_text())
    geometry, info = load_flat(args.directory)
    begin = time.perf_counter()
    simulation = Simulation(geometry, seed=args.cuda_seed, nthreads_per_block=args.threads,
                            max_blocks=args.blocks, use_packed=args.packed)
    cuda.Context.synchronize()
    report = base_report(args, "original CUDA Chroma GPUPhotons.propagate" + (" (packed)" if args.packed else ""), info)
    report.update(device=cuda.Device(0).name(), chroma=chroma.__file__, nthreads_per_block=args.threads,
                  max_blocks=args.blocks, use_packed=args.packed, rng_states=args.threads * args.blocks,
                  setup_seconds=time.perf_counter() - begin)

    def run(count, seed):
        arrays = load_photons(args.directory, manifest, count, seed)
        photons = Photons(arrays["pos"], arrays["dir"], arrays["pol"], arrays["wavelengths"], arrays["t"])
        gc.collect()
        cuda.Context.synchronize()
        start = time.perf_counter()
        gpu_photons = gpu.GPUPhotons(photons, copy_flags=True, copy_triangles=False, copy_weights=False,
                                     use_packed=args.packed)
        cuda.Context.synchronize()
        uploaded = time.perf_counter()
        if args.packed:
            gpu_photons.propagate_packed(simulation.gpu_geometry, simulation.rng_states,
                                         nthreads_per_block=args.threads, max_blocks=args.blocks,
                                         max_steps=args.max_steps)
        else:
            gpu_photons.propagate(simulation.gpu_geometry, simulation.rng_states,
                                  nthreads_per_block=args.threads, max_blocks=args.blocks,
                                  max_steps=args.max_steps)
        cuda.Context.synchronize()
        propagated = time.perf_counter()
        final = gpu_photons.get()
        downloaded = time.perf_counter()
        transport = propagated - uploaded
        prepared = downloaded - start
        row = {
            "photons": count, "seed": seed,
            "upload_seconds": uploaded - start,
            "transport_seconds": transport,
            "download_seconds": downloaded - propagated,
            "prepared_event_seconds": prepared,
            "transport_photons_per_second": count / transport,
            "prepared_photons_per_second": count / prepared,
            "outcomes": outcome_summary(final.flags),
        }
        del gpu_photons, final
        print(json.dumps({k: v for k, v in row.items() if k != "outcomes"}), flush=True)
        return row

    measure(args, report, run, args.output)


def measure(args, report, run, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    for count in args.counts:
        entry = {"photons": count, "warmups": [], "runs": []}
        report["counts"].append(entry)
        # Warm each measured seed once so first-use compilation/allocation is excluded.
        for seed in args.seeds:
            entry["warmups"].append(run(count, seed))
            write_json(output, report)
        for seed in args.seeds:
            entry["runs"].append(run(count, seed))
            write_json(output, report)
        runs = entry["runs"]
        entry["sustained_transport_photons_per_second"] = sum(r["photons"] for r in runs) / sum(
            r["transport_seconds"] for r in runs)
        entry["sustained_prepared_photons_per_second"] = sum(r["photons"] for r in runs) / sum(
            r["prepared_event_seconds"] for r in runs)
        entry["median_transport_seconds"] = float(np.median([r["transport_seconds"] for r in runs]))
        write_json(output, report)
        print(json.dumps({"photons": count,
                          "sustained_transport_photons_per_second": entry["sustained_transport_photons_per_second"],
                          "sustained_prepared_photons_per_second": entry["sustained_prepared_photons_per_second"]}),
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "cuda-geometry", "export-triton", "triton", "cuda"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fixture", default="theia", choices=("theia",))
    parser.add_argument("--counts", type=int, nargs="+", default=[15_000_000, 25_000_000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[901, 1910, 2919])
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--tile-size", type=int, default=0, help="Triton tile size (0: whole event)")
    parser.add_argument("--threads", type=int, default=512, help="CUDA nthreads_per_block (installed default 512)")
    parser.add_argument("--blocks", type=int, default=1024, help="CUDA max_blocks (installed default 1024)")
    parser.add_argument("--packed", action="store_true", help="CUDA float4 packed propagation")
    parser.add_argument("--cuda-seed", type=int, default=12345)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage in ("triton", "cuda") and args.output is None:
        parser.error("--output is required for timing stages")
    stages = {"prepare": prepare, "cuda-geometry": cuda_geometry, "triton": run_triton, "cuda": run_cuda,
              "export-triton": lambda a: export_triton_geometry(a.directory, load_flat(a.directory)[0])}
    stages[args.stage](args)


if __name__ == "__main__":
    main()
