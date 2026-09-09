"""Compare resident transport using identical persisted photons and output fields.

Run prepare once, then cuda and triton in separate GPU processes. Both transport
timers start with uploaded photon states and end at device synchronization;
queue allocation and host scheduling remain included. Both download position,
direction, polarization, time and history for every photon (44 bytes/photon).
This matches the I/O contract, not random trajectories or exact geometry code.
The CUDA path uses existing analytic wires; Triton retains its existing 450 nm
detector specialization. No source or physics implementation is changed here.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import time
from types import SimpleNamespace

import numpy as np


FIELDS = ("pos", "dir", "pol", "t", "flags")
TERMINAL = 1 | 2 | 4 | 8 | 32768


def digest(fields):
    result = hashlib.sha256()
    for name in FIELDS:
        value = np.ascontiguousarray(fields[name])
        result.update(name.encode())
        result.update(str(value.shape).encode())
        result.update(value.dtype.str.encode())
        result.update(value.tobytes())
    return result.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(args):
    from validate_triton_backend import _source_photons

    inputs = args.directory / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    manifest = {"counts": args.counts, "seeds": [123] + [901 + 1009*i for i in range(args.repeats)],
                "center_mm": [-1000, 0, 0], "voxel_size_mm": 30, "wavelength_nm": 450,
                "files": {}}
    for count in args.counts:
        for seed in manifest["seeds"]:
            name = f"{count}_{seed}.npz"
            photons = _source_photons(count, (-1000, 0, 0), 30, seed)
            data = {field: np.ascontiguousarray(getattr(photons, field)) for field in FIELDS}
            np.savez(inputs / name, **data)
            manifest["files"][name] = digest(data)
    write_json(args.directory / "inputs.json", manifest)
    print(json.dumps({"prepared_files": len(manifest["files"])}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("prepare", "cuda", "triton"), required=True)
    parser.add_argument("--directory", type=Path,
                        default=Path("chroma-lar/benchmarks/optical_validation/prepared_transport"))
    parser.add_argument("--counts", type=int, nargs="+", default=[100000, 1000000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2048)
    args = parser.parse_args()
    if min(args.counts) <= 0 or args.repeats < 2 or args.max_steps < 1:
        parser.error("counts and max-steps must be positive; repeats must be at least two")
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.backend == "prepare":
        prepare(args)
        return

    from chroma.event import Photons

    manifest = json.loads((args.directory / "inputs.json").read_text())
    if manifest["counts"] != args.counts or len(manifest["seeds"]) != args.repeats + 1:
        raise ValueError("prepare manifest does not match requested counts/repeats")
    context = None
    setup_started = time.perf_counter()
    if args.backend == "cuda":
        from chroma import gpu
        from chroma.triton.bvh import build_packed_bvh
        from chroma_lar.geometry.config_loader import build_detector_from_config
        from pycuda import gpuarray as ga

        Path(os.environ.get("PYCUDA_CACHE_DIR", "/tmp/chroma-prepared-pycuda")).mkdir(parents=True, exist_ok=True)
        context = gpu.create_cuda_context()
        sync = context.synchronize
        device = context.get_device().name()
        geometry = build_detector_from_config("detector_config_reflect_reflect3wires",
                                              analytic_wires=True, flatten=True)
        bvh = build_packed_bvh(geometry.mesh.vertices, geometry.mesh.triangles)
        geometry.bvh = SimpleNamespace(nodes=np.array(bvh.nodes).view(ga.vec.uint4).reshape(-1),
            world_coords=SimpleNamespace(world_origin=bvh.world_origin, world_scale=bvh.world_scale))
        # Preserve the same wavelength interpolation as the prior comparison.
        with (args.directory.parent / "original_cuda/detector.input.pkl").open("rb") as stream:
            shared_scene, _, _, _ = pickle.load(stream)
        simulation = gpu.GPUGeometry(geometry, wavelengths=shared_scene.host.optics.wavelength_grid.values)
        random = gpu.get_rng_states(512*1024, seed=12345)
        geometry_scope = "existing Chroma CUDA transport, analytic wires, full detector"
        rng_scope = "persistent XORWOW, initialized once with seed 12345"
        implementation = Path("chroma-lite/chroma/cuda/photon.h")
    else:
        import torch
        from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

        simulation = Reflect3WiresTritonSimulation()
        sync = torch.cuda.synchronize
        device = torch.cuda.get_device_name()
        geometry_scope = "existing 450 nm Triton instance/analytic-wire specialization"
        rng_scope = "existing Philox transport stream, seed from each input file"
        implementation = Path("chroma-lar/chroma_lar/triton_backend.py")
    sync()
    report = {"backend": args.backend, "device": device, "host": platform.node(),
              "python": platform.python_version(), "numpy": np.__version__,
              "setup_seconds": time.perf_counter()-setup_started, "max_steps": args.max_steps,
              "output_fields": list(FIELDS), "output_bytes_per_photon": 44,
              "transport_scope": "resident photon state to completion, including queue allocation and host scheduling; synchronized wall time",
              "event_scope": "prepared CPU arrays through matching full per-photon CPU output; excludes source generation and file I/O",
              "geometry_scope": geometry_scope, "rng_scope": rng_scope,
              "cache_scope": "retained compiler caches; one discarded full-size warm-up at each count",
              "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "counts": []}

    def run(count, seed):
        name = f"{count}_{seed}.npz"
        with np.load(args.directory / "inputs" / name) as data:
            source = {key: np.ascontiguousarray(data[key]) for key in FIELDS}
        input_hash = digest(source)
        if input_hash != manifest["files"][name]:
            raise RuntimeError("input digest mismatch")
        photons = Photons(source["pos"], source["dir"], source["pol"],
                          np.full(count, 450, np.float32), t=source["t"], flags=source["flags"])
        gc.collect()
        sync()
        start = time.perf_counter()
        if args.backend == "cuda":
            state = gpu.GPUPhotons(photons)
        else:
            state = tuple(torch.from_numpy(source[key]).to("cuda") for key in ("pos", "dir", "pol", "t")) + (
                torch.from_numpy(source["flags"].view(np.int32)).to("cuda"),
                torch.zeros(count, dtype=torch.int64, device="cuda"),
                torch.full((count,), -1, dtype=torch.int32, device="cuda"),
                torch.full((count,), -1, dtype=torch.int32, device="cuda"),
                torch.full((count,), -1, dtype=torch.int32, device="cuda"),
                torch.zeros(count, dtype=torch.int32, device="cuda"),
            )
        sync()
        uploaded = time.perf_counter()
        # Verify what actually arrived on the GPU before starting transport.
        # This audit is excluded from the event/transport measurements.
        def download():
            if args.backend == "cuda":
                return {"pos": state.pos.get().view(np.float32).reshape(count, 3),
                        "dir": state.dir.get().view(np.float32).reshape(count, 3),
                        "pol": state.pol.get().view(np.float32).reshape(count, 3),
                        "t": state.t.get(), "flags": state.flags.get()}
            return {key: state[i].cpu().numpy().view(np.uint32) if key == "flags" else state[i].cpu().numpy()
                    for i, key in enumerate(FIELDS)}
        resident_hash = digest(download())
        if resident_hash != input_hash:
            raise RuntimeError("uploaded input digest mismatch")
        sync()
        began_transport = time.perf_counter()
        if args.backend == "cuda":
            state.propagate(simulation, random, nthreads_per_block=512,
                            max_blocks=1024, max_steps=args.max_steps)
        else:
            simulation.pmt_workspace.clear_sticky_overflow()
            pending = torch.arange(count, dtype=torch.int32, device="cuda")
            pending, rounds, events, trace = simulation._propagate_state(
                state, pending, seed, 0, args.max_steps, args.max_steps)
            if pending.numel():
                raise RuntimeError(f"transport left {pending.numel()} pending photons")
            if simulation.pmt_workspace.sticky_overflowed():
                raise RuntimeError("Triton PMT BVH stack overflow")
        sync()
        propagated = time.perf_counter()
        final = download()
        sync()
        completed = time.perf_counter()
        flags = final["flags"]
        nonfinite = sum(int(np.count_nonzero(~np.isfinite(final[key]))) for key in FIELDS[:-1])
        unfinished = int(np.count_nonzero((flags & TERMINAL) == 0))
        aborted = int(np.count_nonzero(flags & (32768 | (1 << 31))))
        # In this enclosed detector no photon should escape. Checking NO_HIT
        # also catches the optimized path's legacy step-limit encoding.
        escaped = int(np.count_nonzero(flags & 1))
        valid = not (nonfinite or unfinished or aborted or escaped)
        if not valid:
            bad = ((flags & TERMINAL) == 0) | ((flags & (1 | 32768 | (1 << 31))) != 0)
            for field in FIELDS[:-1]:
                finite = np.isfinite(final[field])
                bad |= ~finite.all(axis=1) if finite.ndim == 2 else ~finite
            rows = np.flatnonzero(bad)
            np.savez(args.directory / f"{args.backend}_{count}_{seed}_invalid.npz", rows=rows,
                     **{"input_"+key: value[rows] for key, value in source.items()},
                     **{"final_"+key: value[rows] for key, value in final.items()})
        if sum(value.nbytes for value in final.values()) != 44*count:
            raise RuntimeError("output byte contract violated")
        upload_seconds = uploaded-start
        transport_seconds = propagated-began_transport
        download_seconds = completed-propagated
        event_seconds = upload_seconds+transport_seconds+download_seconds
        record = {"seed": seed, "photons": count, "input_sha256": input_hash,
                  "resident_input_sha256": resident_hash,
                  "upload_allocation_seconds": upload_seconds, "transport_seconds": transport_seconds,
                  "download_seconds": download_seconds, "event_seconds": event_seconds,
                  "transport_photons_per_second": count/transport_seconds,
                  "event_photons_per_second": count/event_seconds,
                  "detected": int(np.count_nonzero(flags & 4)), "unfinished": unfinished,
                  "nonfinite": nonfinite, "aborted": aborted, "escaped": escaped,
                  "valid": valid,
                  "history_counts": {str(int(flag)): int(n) for flag, n in zip(*np.unique(flags, return_counts=True))}}
        if args.backend == "triton":
            record.update(boundary_rounds=rounds, boundary_events=events,
                          max_steps_observed=int(state[9].max().item()))
        print(json.dumps({"backend": args.backend, **{key: record[key] for key in
            ("seed", "photons", "transport_seconds", "event_seconds", "detected", "valid", "nonfinite", "aborted", "escaped")}}), flush=True)
        return record

    try:
        for count in args.counts:
            warmup = run(count, manifest["seeds"][0])
            entry = {"photons": count, "warmup": warmup, "runs": []}
            report["counts"].append(entry)
            for seed in manifest["seeds"][1:]:
                entry["runs"].append(run(count, seed))
                for statistic, function in (("median", np.median), ("minimum", np.min), ("maximum", np.max)):
                    entry[statistic] = {key: float(function([r[key] for r in entry["runs"]]))
                        for key in ("transport_seconds", "event_seconds", "transport_photons_per_second", "event_photons_per_second")}
                write_json(args.directory / f"{args.backend}.json", report)
    finally:
        if context is not None:
            context.pop()


if __name__ == "__main__":
    main()
