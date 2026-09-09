"""Run shared comparison cases through the original Chroma CUDA transport.

Run in the existing local Chroma container with CHROMA_FORCE_SCATTER_AT_PASS=0.
Only optical transport is compared: original WLS has zero emission delay.
"""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from optical_comparison_cases import CASES, GRID, make_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11,29,83])
    parser.add_argument("--export-only", action="store_true", help="export exact input scene for cross-environment mesh parity")
    args = parser.parse_args()
    if os.environ.get("PYCUDA_CACHE_DIR"):
        Path(os.environ["PYCUDA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    from chroma import gpu
    from chroma.triton.bvh import build_packed_bvh
    from pycuda import gpuarray as ga
    import pycuda.driver as cuda
    context = gpu.create_cuda_context()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        for name in args.cases:
            detector, photons, steps, expected = make_case(name, args.count)
            detector.flatten()
            bvh = build_packed_bvh(detector.mesh.vertices, detector.mesh.triangles)
            detector.bvh = SimpleNamespace(nodes=np.array(bvh.nodes).view(ga.vec.uint4).reshape(-1),
                world_coords=SimpleNamespace(world_origin=bvh.world_origin, world_scale=bvh.world_scale),
                layer_offsets=bvh.layer_offsets)
            from chroma.triton.spectral import SpectralScene
            scene = SpectralScene.compile(detector, wavelengths=GRID)
            with (output/f"{name}.input.pkl").open("wb") as stream:
                pickle.dump((scene, photons, steps, expected), stream, protocol=4)
            if args.export_only:
                print(json.dumps({"case":name,"exported_triangles":scene.host.triangle_count}),flush=True)
                continue
            device = gpu.GPUGeometry(detector, wavelengths=GRID)
            geometry_hash = hashlib.sha256(detector.mesh.vertices.tobytes()+detector.mesh.triangles.tobytes()).hexdigest()
            source_hash = hashlib.sha256(b"".join(getattr(photons,f).tobytes() for f in ("pos","dir","pol","wavelengths","t"))).hexdigest()
            for seed in args.seeds:
                state = gpu.GPUPhotons(photons)
                random = gpu.get_rng_states(128*128, seed=seed)
                state.propagate(device, random, nthreads_per_block=128, max_blocks=128, max_steps=steps)
                p = state.get()
                channels = np.full(args.count,-1,np.int32)
                detected = (p.flags & 4) != 0
                channels[detected] = detector.solid_id_to_channel_index[detector.solid_id[p.last_hit_triangles[detected]]]
                meta = {"case":name,"seed":seed,"count":args.count,"max_steps":steps,"expected":expected,
                        "geometry_sha256":geometry_hash,"source_sha256":source_hash,"gpu":cuda.Context.get_device().name(),
                        "cuda_force_scatter_at_pass":0}
                np.savez_compressed(output/f"{name}.{seed}.cuda.npz", pos=p.pos, direction=p.dir,
                    polarization=p.pol, wavelengths=p.wavelengths, times=p.t, flags=p.flags,
                    last_hit=p.last_hit_triangles, channels=channels, metadata=json.dumps(meta))
                print(json.dumps({"case":name,"seed":seed,"detected":int(detected.sum())}), flush=True)
                del state, random
            del device
    finally:
        context.pop()


if __name__ == "__main__":
    main()
