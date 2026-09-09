"""Capture untouched CUDA RNG goldens, or check independent NumPy initialization.

The capture command runs in the original CUDA container. The check command is
CPU-only and uses TriChroma's independently exponentiated GF(2) transition.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np

CUDA = r"""
#include "random.h"
extern "C" __global__ void inspect_xorwow(
    int n, unsigned long long seed, unsigned long long offset,
    unsigned long long *subsequences, unsigned int *initial,
    unsigned int *final, unsigned int *integers, float *uniforms)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    if(id >= n) return;
    curandState rng;
    curand_init(seed,subsequences[id],offset,&rng);
    initial[id*6] = rng.d;
    for(int j=0;j<5;++j) initial[id*6+j+1] = rng.v[j];
    for(int j=0;j<32;++j) {
        unsigned int word = curand(&rng);
        integers[id*32+j] = word;
        uniforms[id*32+j] = _curand_uniform(word);
    }
    final[id*6] = rng.d;
    for(int j=0;j<5;++j) final[id*6+j+1] = rng.v[j];
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["capture", "check"])
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    path = args.directory / "xorwow_goldens.npz"
    if args.mode == "capture":
        if args.reference_root is None:
            parser.error("capture needs --reference-root")
        reference = args.reference_root.resolve()
        os.environ.setdefault("PYCUDA_CACHE_DIR", "/tmp/trichroma-native-capture-cache")
        Path(os.environ["PYCUDA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(reference))
        from chroma import gpu
        from chroma.cuda import srcdir
        from chroma.gpu.tools import cuda_options
        from pycuda.compiler import SourceModule
        from pycuda import gpuarray as ga

        if not Path(srcdir).resolve().is_relative_to(reference):
            raise ValueError("the selected original headers were not imported")
        args.directory.mkdir(parents=True, exist_ok=True)
        context = gpu.create_cuda_context()
        try:
            module = SourceModule(
                CUDA, no_extern_c=True, options=[*cuda_options, "-I" + str(srcdir)]
            )
            function = module.get_function("inspect_xorwow")
            subsequences = np.asarray(
                [0, 1, 2, 17, 255, 8191, 2**32 + 3, 2**63, 2**64 - 1], np.uint64
            )
            subseq_gpu = ga.to_gpu(subsequences)
            arrays, runs = {"subsequences": subsequences}, []
            for seed in [0, 1, 99173, 2**32 + 7, 2**64 - 1]:
                for offset in [0, 1, 129, 2**32 + 7, 2**63 + 13]:
                    initial = ga.empty((len(subsequences), 6), np.uint32)
                    final = ga.empty_like(initial)
                    integers = ga.empty((len(subsequences), 32), np.uint32)
                    uniforms = ga.empty((len(subsequences), 32), np.float32)
                    function(
                        np.int32(len(subsequences)),
                        np.uint64(seed),
                        np.uint64(offset),
                        subseq_gpu,
                        initial,
                        final,
                        integers,
                        uniforms,
                        block=(32, 1, 1),
                        grid=(1, 1, 1),
                    )
                    key = f"run_{len(runs)}"
                    for name, value in (
                        ("initial", initial),
                        ("final", final),
                        ("integers", integers),
                        ("uniforms", uniforms),
                    ):
                        arrays[key + "_" + name] = value.get()
                    runs.append(dict(key=key, seed=seed, offset=offset))
            np.savez_compressed(path, **arrays)
            report = {
                "reference": str(reference),
                "cuda_headers": str(srcdir),
                "cuda_options": list(cuda_options),
                "original_random_header_sha256": hashlib.sha256(
                    (Path(srcdir) / "random.h").read_bytes()
                ).hexdigest(),
                "wrapper_sha256": hashlib.sha256(CUDA.encode()).hexdigest(),
                "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "runs": runs,
            }
            (args.directory / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"Captured {len(runs)*len(subsequences)} native RNG states and {len(runs)*len(subsequences)*32} draws"
            )
        finally:
            context.pop()
    else:
        from chroma.triton.xorwow import initialize_xorwow, xorwow_uint32

        report = json.loads((args.directory / "capture.json").read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != report["archive_sha256"]:
            raise ValueError("native golden archive hash mismatch")
        result = {
            "capture_sha256": hashlib.sha256(
                (args.directory / "capture.json").read_bytes()
            ).hexdigest(),
            "implementation_sha256": hashlib.sha256(
                Path(sys.modules["chroma.triton.xorwow"].__file__).read_bytes()
            ).hexdigest(),
            "runs": [],
        }
        with np.load(path, allow_pickle=False) as arrays:
            for row in report["runs"]:
                key = row["key"]
                state = initialize_xorwow(row["seed"], arrays["subsequences"], row["offset"])
                mismatches = {"initial": int(np.count_nonzero(state != arrays[key + "_initial"]))}
                integers = np.column_stack([xorwow_uint32(state) for _ in range(32)])
                uniforms = integers.astype(np.float32) * np.float32(2.0**-32) + np.float32(2.0**-33)
                for name, value in (
                    ("final", state),
                    ("integers", integers),
                    ("uniforms", uniforms.view(np.uint32)),
                ):
                    mismatches[name] = int(
                        np.count_nonzero(value != arrays[key + "_" + name].view(np.uint32))
                    )
                result["runs"].append(row | {"mismatches": mismatches})
        result["passed"] = all(not any(r["mismatches"].values()) for r in result["runs"])
        (args.directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        if not result["passed"]:
            raise AssertionError("independent XORWOW differs from original CUDA")


if __name__ == "__main__":
    main()
