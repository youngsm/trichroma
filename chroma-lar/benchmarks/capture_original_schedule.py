"""Observe native Chroma launches and their nondeterministic queue schedule.

The original public API and CUDA kernels execute unchanged. An instance-local
observer records launch arguments and downloads state after each launch. This
allows a Triton replay of one actual original execution without assuming that
another run with the same seed will choose the same atomic queue ordering.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from capture_original_chroma import digest, export_scene, photon_words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="spectral_multibounce")
    parser.add_argument("--count", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=99173)
    parser.add_argument("--max-steps", type=int, default=128)
    args = parser.parse_args()
    if args.count <= 0 or args.max_steps <= 0:
        parser.error("positive count and max-steps required")
    reference = args.reference_root.resolve()
    sys.path.insert(0, str(reference))
    # Detector fixtures belong to this checkout. Import the original Chroma
    # transport explicitly, without accidentally selecting the container's
    # unrelated chroma_lar geometry builder.
    fixture_root = Path(__file__).resolve().parents[1]
    sys.path.insert(1, str(fixture_root))
    os.environ.setdefault("PYCUDA_CACHE_DIR", "/tmp/trichroma-native-capture-cache")
    os.environ.setdefault("CHROMA_FORCE_SCATTER_AT_PASS", "0")
    from chroma import gpu
    from chroma.cuda import srcdir
    from chroma.gpu.tools import cuda_options
    from chroma.bvh.grid import make_recursive_grid_bvh
    from pycuda import characterize
    import pycuda.driver as cuda
    from original_chroma_cases import GRID, make_case

    if not Path(srcdir).resolve().is_relative_to(reference):
        raise ValueError("the selected original CUDA headers were not imported")
    if not hasattr(np.linalg, "linalg"):
        np.linalg.linalg = np.linalg
    args.output.mkdir(parents=True, exist_ok=True)
    context = gpu.create_cuda_context()
    try:
        geometry, source, _, _ = make_case(args.case, args.count)
        geometry.flatten()
        for plane in getattr(geometry, "wireplanes", None) or []:
            for name, keys in (
                ("unique_materials", ("material_inner", "material_outer")),
                ("unique_surfaces", ("surface",)),
            ):
                table = list(getattr(geometry, name))
                for key in keys:
                    if plane[key] not in table:
                        table.append(plane[key])
                setattr(geometry, name, np.asarray(table, dtype=object))
        geometry.bvh = make_recursive_grid_bvh(geometry.mesh, target_degree=3)
        device = gpu.GPUDetector(geometry, wavelengths=GRID)
        scene = export_scene(geometry, device, GRID)
        np.savez_compressed(args.output / "scene.npz", **scene)
        np.savez_compressed(args.output / "source.npz", source_words=photon_words(source))
        slots = 256 * 1024
        state_size = characterize.sizeof("curandStateXORWOW", "#include <curand_kernel.h>")
        rng = gpu.get_rng_states(slots, seed=args.seed)
        initial_rng = np.empty((slots, state_size), np.uint8)
        cuda.memcpy_dtoh(initial_rng, rng)
        report = {
            "reference": str(reference),
            "case": args.case,
            "photons": args.count,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "rng_slots": slots,
            "device": cuda.Context.get_device().name(),
            "cuda_options": list(cuda_options),
            "source_sha256": {
                str(path.relative_to(reference)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((reference / "chroma").rglob("*"))
                if path.is_file()
                and path.suffix in (".py", ".h", ".cu")
                and "__pycache__" not in path.parts
                and "_build_ext" not in path.parts
            },
            "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fixture_source_sha256": {
                str(path.relative_to(fixture_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [
                    Path(__file__).with_name("original_chroma_cases.py"),
                    Path(__file__).with_name("optical_comparison_cases.py"),
                    *sorted((fixture_root / "chroma_lar/geometry").rglob("*.py")),
                ]
            },
            "scene_array_sha256": {key: digest(value) for key, value in scene.items()},
            "initial_photons_sha256": digest(photon_words(source)),
            "initial_rng_words_sha256": digest(initial_rng.view(np.uint32)[:, :6]),
            "scope": "observation of actual original public API/kernel launches; no substituted random generator or modified CUDA kernel",
            "launches": [],
        }
        photons = gpu.GPUPhotons(source)
        original_functions = photons.gpu_funcs

        class Observer:
            def __getattr__(self, name):
                function = getattr(original_functions, name)
                if name != "propagate":
                    return function

                def observed(*call, **kwargs):
                    first, count = int(call[0]), int(call[1])
                    ids = call[2][first : first + count].get()
                    # Call the actual original kernel with its exact arguments.
                    function(*call, **kwargs)
                    cuda.Context.synchronize()
                    native_states = np.empty((count, state_size), np.uint8)
                    cuda.memcpy_dtoh(native_states, rng)
                    state_words = photon_words(photons.get())[ids]
                    output_queue = call[3].get()
                    queue_count = int(output_queue[0]) - 1
                    index = len(report["launches"])
                    path = args.output / f"launch_{index:04d}.npz"
                    np.savez_compressed(
                        path,
                        photon_ids=ids,
                        state_words=state_words,
                        native_rng_words=native_states.view(np.uint32)[:, :6],
                        output_queue=output_queue[1 : 1 + queue_count],
                    )
                    row = {
                        "index": index,
                        "first_photon": first,
                        "threads": count,
                        "max_steps": int(call[14]),
                        "use_weights": int(call[15]),
                        "scatter_first": int(call[16]),
                        "block": list(kwargs["block"]),
                        "grid": list(kwargs["grid"]),
                        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    report["launches"].append(row)
                    print(json.dumps(row), flush=True)

                return observed

        photons.gpu_funcs = Observer()
        photons.propagate(
            device, rng, nthreads_per_block=256, max_blocks=1024, max_steps=args.max_steps
        )
        final = photon_words(photons.get())
        np.savez_compressed(args.output / "final.npz", final_words=final)
        report["final_words_sha256"] = digest(final)
        report["unfinished_photons"] = int(np.count_nonzero((final[:, 11] & 32783) == 0))
        (args.output / "schedule.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        context.pop()


if __name__ == "__main__":
    main()
