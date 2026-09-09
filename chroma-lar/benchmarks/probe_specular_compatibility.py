#!/usr/bin/env python3
"""Compare Chroma CUDA and Triton legacy-reflection words on identical rays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
for source_root in (REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


def _inputs(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    direction = rng.standard_normal((count, 3)).astype(np.float32)
    normal = rng.standard_normal((count, 3)).astype(np.float32)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True).astype(np.float32)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True).astype(np.float32)
    # Chroma orients the state normal toward the incident ray.
    dot = np.einsum("ij,ij->i", direction, normal, dtype=np.float32)
    normal[dot > 0.0] *= np.float32(-1.0)
    return np.ascontiguousarray(direction), np.ascontiguousarray(normal)


def _cuda_worker(count: int, seed: int, output: Path) -> None:
    import pycuda.autoprimaryctx  # noqa: F401
    from pycuda import driver as cuda
    from pycuda import gpuarray as ga

    from chroma.gpu.tools import cuda_options, get_cu_module, to_float3

    direction, normal = _inputs(count, seed)
    direction_gpu = ga.to_gpu(to_float3(direction))
    normal_gpu = ga.to_gpu(to_float3(normal))
    result_gpu = ga.empty_like(direction_gpu)
    diagnostics_gpu = ga.empty((count, 12), dtype=np.float32)
    module = get_cu_module("specular_probe.cu", options=cuda_options)
    block = 256
    module.get_function("specular_probe")(
        np.int32(count),
        direction_gpu,
        normal_gpu,
        result_gpu,
        diagnostics_gpu,
        block=(block, 1, 1),
        grid=((count + block - 1) // block, 1, 1),
    )
    cuda.Context.synchronize()
    np.savez(
        output,
        result=result_gpu.get().view(np.float32).reshape(count, 3),
        diagnostics=diagnostics_gpu.get(),
    )


def _triton_worker(count: int, seed: int, output: Path) -> None:
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    from chroma.triton.physics_kernels import (
        acos_chroma_fast,
        reflect_specular_chroma,
    )

    globals().update(
        torch=torch,
        triton=triton,
        tl=tl,
        libdevice=libdevice,
        acos_chroma_fast=acos_chroma_fast,
        reflect_specular_chroma=reflect_specular_chroma,
    )

    @triton.jit
    def probe_kernel(
        direction, normal, result, diagnostics, count, BLOCK: tl.constexpr
    ):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < count
        base = lane * 3
        dx = tl.load(direction + base, mask=valid, other=1.0)
        dy = tl.load(direction + base + 1, mask=valid, other=0.0)
        dz = tl.load(direction + base + 2, mask=valid, other=0.0)
        nx = tl.load(normal + base, mask=valid, other=-1.0)
        ny = tl.load(normal + base + 1, mask=valid, other=0.0)
        nz = tl.load(normal + base + 2, mask=valid, other=0.0)
        rx, ry, rz = reflect_specular_chroma(dx, dy, dz, nx, ny, nz)
        incident_dot = tl.fma(
            nz, -dz, tl.fma(ny, -dy, nx * -dx)
        )
        incident_cosine = tl.maximum(-1.0, tl.minimum(1.0, incident_dot))
        incident_angle = acos_chroma_fast(incident_cosine)
        axis_x = dy * nz - dz * ny
        axis_y = dz * nx - dx * nz
        axis_z = dx * ny - dy * nx
        raw_axis_x, raw_axis_y, raw_axis_z = axis_x, axis_y, axis_z
        axis_length = tl.sqrt(
            axis_x * axis_x + axis_y * axis_y + axis_z * axis_z
        )
        axis_x /= axis_length
        axis_y /= axis_length
        axis_z /= axis_length
        cosine = libdevice.fast_cosf(incident_angle)
        sine = libdevice.fast_sinf(incident_angle)
        projection = nx * axis_x + ny * axis_y + nz * axis_z
        tl.store(result + base, rx, mask=valid)
        tl.store(result + base + 1, ry, mask=valid)
        tl.store(result + base + 2, rz, mask=valid)
        # Candidate dot-product association orders.  NVCC is free to
        # reassociate/fuse this expression under Chroma's --use_fast_math;
        # recording every useful three-term FMA tree lets the probe identify
        # its raw-word equivalent without guessing from aggregate outputs.
        dot_xyz = tl.fma(nx, -dx, tl.fma(ny, -dy, nz * -dz))
        dot_zyx = tl.fma(nz, -dz, tl.fma(ny, -dy, nx * -dx))
        dot_xzy = tl.fma(nx, -dx, tl.fma(nz, -dz, ny * -dy))
        dot_yxz = tl.fma(ny, -dy, tl.fma(nx, -dx, nz * -dz))
        dot_yzx = tl.fma(ny, -dy, tl.fma(nz, -dz, nx * -dx))
        dot_zxy = tl.fma(nz, -dz, tl.fma(nx, -dx, ny * -dy))
        diagnostic_base = lane * 18
        tl.store(diagnostics + diagnostic_base + 0, incident_cosine, mask=valid)
        tl.store(diagnostics + diagnostic_base + 1, incident_angle, mask=valid)
        tl.store(diagnostics + diagnostic_base + 2, raw_axis_x, mask=valid)
        tl.store(diagnostics + diagnostic_base + 3, raw_axis_y, mask=valid)
        tl.store(diagnostics + diagnostic_base + 4, raw_axis_z, mask=valid)
        tl.store(diagnostics + diagnostic_base + 5, axis_length, mask=valid)
        tl.store(diagnostics + diagnostic_base + 6, axis_x, mask=valid)
        tl.store(diagnostics + diagnostic_base + 7, axis_y, mask=valid)
        tl.store(diagnostics + diagnostic_base + 8, axis_z, mask=valid)
        tl.store(diagnostics + diagnostic_base + 9, cosine, mask=valid)
        tl.store(diagnostics + diagnostic_base + 10, sine, mask=valid)
        tl.store(diagnostics + diagnostic_base + 11, projection, mask=valid)
        tl.store(diagnostics + diagnostic_base + 12, dot_xyz, mask=valid)
        tl.store(diagnostics + diagnostic_base + 13, dot_zyx, mask=valid)
        tl.store(diagnostics + diagnostic_base + 14, dot_xzy, mask=valid)
        tl.store(diagnostics + diagnostic_base + 15, dot_yxz, mask=valid)
        tl.store(diagnostics + diagnostic_base + 16, dot_yzx, mask=valid)
        tl.store(diagnostics + diagnostic_base + 17, dot_zxy, mask=valid)

    direction, normal = _inputs(count, seed)
    direction_gpu = torch.as_tensor(direction, device="cuda")
    normal_gpu = torch.as_tensor(normal, device="cuda")
    result_gpu = torch.empty_like(direction_gpu)
    diagnostics_gpu = torch.empty(
        (count, 18), dtype=torch.float32, device="cuda"
    )
    block = 256
    probe_kernel[(triton.cdiv(count, block),)](
        direction_gpu, normal_gpu, result_gpu, diagnostics_gpu, count,
        BLOCK=block, num_warps=8
    )
    torch.cuda.synchronize()
    np.savez(
        output,
        result=result_gpu.cpu().numpy(),
        diagnostics=diagnostics_gpu.cpu().numpy(),
    )


def _worker_command(args: argparse.Namespace, backend: str, output: Path) -> list[str]:
    worker = [
        str(Path(__file__).resolve()),
        "--count", str(args.count),
        "--seed", str(args.seed),
        "--_worker", backend,
        "--_output", str(output),
    ]
    if backend == "cuda":
        image = Path(args.chroma_container).expanduser().resolve()
        if image.is_file():
            python_path = "%s:%s" % (
                REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"
            )
            return [
                "singularity", "exec", "--nv",
                "-B", "/sdf:/sdf", "-B", "/tmp:/tmp",
                "--pwd", str(REPOSITORY), str(image),
                "env", "PYTHONPATH=" + python_path,
                "PYTHONNOUSERSITE=1", "PYCUDA_CACHE_DIR=/tmp/chroma-pycuda-cache",
                "TMPDIR=/tmp", "python", *worker,
            ]
    return [sys.executable, *worker]


def _report(cuda: dict, triton_values: dict, seed: int) -> dict:
    cuda_result = cuda["result"]
    triton_result = triton_values["result"]
    cuda_words = np.ascontiguousarray(cuda_result).view(np.uint32)
    triton_words = np.ascontiguousarray(triton_result).view(np.uint32)
    different = cuda_words != triton_words
    row_different = np.any(different, axis=1)
    first = np.argwhere(different)
    first_detail = None
    if len(first):
        row, component = map(int, first[0])
        first_detail = {
            "row": row,
            "component": component,
            "cuda_value": float(cuda_result[row, component]),
            "triton_value": float(triton_result[row, component]),
            "cuda_word": "0x%08x" % int(cuda_words[row, component]),
            "triton_word": "0x%08x" % int(triton_words[row, component]),
        }
    absolute = np.abs(
        cuda_result.astype(np.float64) - triton_result.astype(np.float64)
    )
    diagnostic_names = (
        "incident_cosine", "incident_angle",
        "raw_axis_x", "raw_axis_y", "raw_axis_z", "axis_length",
        "axis_x", "axis_y", "axis_z", "fast_cosine", "fast_sine",
        "axis_projection",
    )
    cuda_diagnostic_words = np.ascontiguousarray(
        cuda["diagnostics"]
    ).view(np.uint32)
    triton_diagnostic_words = np.ascontiguousarray(
        triton_values["diagnostics"]
    ).view(np.uint32)
    diagnostic_exact = {}
    for index, name in enumerate(diagnostic_names):
        matches = (
            cuda_diagnostic_words[:, index]
            == triton_diagnostic_words[:, index]
        )
        diagnostic_exact[name] = {
            "exact": int(np.count_nonzero(matches)),
            "mismatched": int(np.count_nonzero(~matches)),
            "exact_fraction": float(np.mean(matches)),
        }
    dot_candidate_exact = {}
    cuda_incident_cosine = cuda_diagnostic_words[:, 0]
    for index, name in enumerate(
        ("xyz", "zyx", "xzy", "yxz", "yzx", "zxy"), start=12
    ):
        matches = triton_diagnostic_words[:, index] == cuda_incident_cosine
        dot_candidate_exact[name] = {
            "exact": int(np.count_nonzero(matches)),
            "mismatched": int(np.count_nonzero(~matches)),
            "exact_fraction": float(np.mean(matches)),
        }
    return {
        "schema_version": 1,
        "operation": "Chroma acos/Rodrigues specular reflection",
        "seed": int(seed),
        "rays": int(len(cuda_result)),
        "exact_rows": int(np.count_nonzero(~row_different)),
        "mismatched_rows": int(np.count_nonzero(row_different)),
        "mismatched_components": int(np.count_nonzero(different)),
        "exact_row_fraction": float(np.mean(~row_different)),
        "maximum_absolute_difference": float(np.max(absolute)),
        "first_difference": first_detail,
        "intermediate_raw_word_agreement": diagnostic_exact,
        "incident_dot_fma_candidate_agreement": dot_candidate_exact,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--chroma-container",
        default=str(
            REPOSITORY.parent / "chroma-lar" / "installation"
            / "chroma3.lar-plib" / "chroma.simg"
        ),
    )
    parser.add_argument("--_worker", choices=("cuda", "triton"), default=None)
    parser.add_argument("--_output", default=None)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args._worker:
        output = Path(args._output)
        if args._worker == "cuda":
            _cuda_worker(args.count, args.seed, output)
        else:
            _triton_worker(args.count, args.seed, output)
        return 0

    with tempfile.TemporaryDirectory(prefix="specular-probe-", dir="/tmp") as temp:
        paths = {name: Path(temp) / (name + ".npz") for name in ("cuda", "triton")}
        for backend in ("cuda", "triton"):
            completed = subprocess.run(
                _worker_command(args, backend, paths[backend]),
                cwd=str(REPOSITORY), text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            if completed.returncode != 0 or not paths[backend].exists():
                diagnostic = (completed.stdout + "\n" + completed.stderr)[-12_000:]
                raise RuntimeError(
                    "%s worker failed with status %d:\n%s"
                    % (backend, completed.returncode, diagnostic)
                )
        with np.load(paths["cuda"]) as cuda, np.load(paths["triton"]) as triton_values:
            report = _report(cuda, triton_values, args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        output = Path(args.json).expanduser().resolve()
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
