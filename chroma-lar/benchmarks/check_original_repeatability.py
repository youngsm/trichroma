"""Test original Chroma's seed repeatability across its queue threshold.

Only the explicitly selected original installation executes photon transport.
The geometry, source words and complete initial RNG bytes are held fixed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from capture_original_chroma import FIELDS, photon_words, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[8192, 65536])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2 or min(args.counts) < 1:
        parser.error("positive populations and at least two repeats are required")
    reference = args.reference_root.resolve()
    sys.path.insert(0, str(reference))
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
        raise ValueError("selected original headers were not imported")
    if not hasattr(np.linalg, "linalg"):
        np.linalg.linalg = np.linalg
    args.output.mkdir(parents=True, exist_ok=True)
    context = gpu.create_cuda_context()
    try:
        slots = 256 * 1024
        nbytes = slots * characterize.sizeof("curandStateXORWOW", "#include <curand_kernel.h>")
        initial_rng = gpu.get_rng_states(slots, seed=99173)
        rng_bytes = np.empty(nbytes, np.uint8)
        cuda.memcpy_dtoh(rng_bytes, initial_rng)
        report = {
            "reference": str(reference),
            "device": cuda.Context.get_device().name(),
            "seed": 99173,
            "nthreads_per_block": 256,
            "max_blocks": 1024,
            "cuda_options": list(cuda_options),
            "source_sha256": {
                str(path.relative_to(reference)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((reference / "chroma").rglob("*"))
                if path.is_file()
                and path.suffix in (".py", ".h", ".cu")
                and "__pycache__" not in path.parts
                and "_build_ext" not in path.parts
            },
            "initial_rng_bytes_sha256": digest(rng_bytes),
            "scope": "untouched original versus itself; repeatability, not Triton equivalence",
            "populations": [],
        }
        for count in args.counts:
            geometry, source, _, _ = make_case("spectral_multibounce", count)
            geometry.flatten()
            geometry.bvh = make_recursive_grid_bvh(geometry.mesh, target_degree=3)
            device = gpu.GPUDetector(geometry, wavelengths=GRID)
            for steps in (1, 128):
                outputs = []
                for repeat in range(args.repeats):
                    photons = gpu.GPUPhotons(source)
                    native_rng = cuda.mem_alloc(nbytes)
                    cuda.memcpy_dtod(native_rng, initial_rng, nbytes)
                    photons.propagate(
                        device, native_rng, nthreads_per_block=256, max_blocks=1024, max_steps=steps
                    )
                    outputs.append(photon_words(photons.get()))
                    del photons, native_rng
                reference_words = outputs[0]
                comparisons = []
                for repeat, words in enumerate(outputs[1:], 1):
                    differences = words != reference_words
                    comparisons.append(
                        {
                            "repeat": repeat,
                            "different_photons": int(np.count_nonzero(np.any(differences, axis=1))),
                            "different_words": int(np.count_nonzero(differences)),
                            "different_history_flags": int(np.count_nonzero(differences[:, 11])),
                            "different_words_by_field": dict(
                                zip(FIELDS, map(int, differences.sum(axis=0)))
                            ),
                        }
                    )
                row = {
                    "photons": count,
                    "max_steps": steps,
                    "source_sha256": digest(photon_words(source)),
                    "output_sha256": [digest(words) for words in outputs],
                    "comparisons": comparisons,
                    "bitwise_repeatable": all(c["different_words"] == 0 for c in comparisons),
                }
                np.savez_compressed(
                    args.output / f"original_{count}_{steps}.npz",
                    **{f"repeat_{i}": words for i, words in enumerate(outputs)},
                )
                report["populations"].append(row)
                (args.output / "repeatability.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
            del device
    finally:
        context.pop()


if __name__ == "__main__":
    main()
