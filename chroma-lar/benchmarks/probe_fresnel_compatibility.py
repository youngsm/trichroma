#!/usr/bin/env python3
"""Compare Chroma CUDA and Triton Fresnel boundary arithmetic by raw word.

The generated sample starts with a pinned failing reflect3wires lockstep case,
then adds randomized, physically shaped direction/polarization/normal vectors.
An input NPZ may instead provide ``direction``, ``polarization``, and ``normal``;
``n1``, ``n2``, ``u_polarization``, and ``u_reflect`` are optional overrides.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
for source_root in (REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

DEFAULT_CHROMA_CONTAINER = (
    REPOSITORY.parent / "chroma-lar" / "installation"
    / "chroma3.lar-plib" / "chroma.simg"
)
DIAGNOSTIC_NAMES = (
    "incident_cosine",
    "incident_angle",
    "refracted_argument",
    "refracted_angle",
    "raw_axis_length",
    "incidence_axis_x",
    "incidence_axis_y",
    "incidence_axis_z",
    "normal_probability",
    "reflection_coefficient",
    "reflectance",
    "outgoing_angle",
)


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    length = np.linalg.norm(values, axis=1, keepdims=True).astype(np.float32)
    return values / np.maximum(length, np.float32(1.0e-20))


def _f32_words(*words: int) -> np.ndarray:
    return np.asarray(words, np.uint32).view(np.float32)


def _generated_inputs(count: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed & 0xFFFFFFFF)
    direction = _unit_rows(rng.standard_normal((count, 3)))
    helper = _unit_rows(rng.standard_normal((count, 3)))
    polarization = _unit_rows(np.cross(direction, helper))
    normal = _unit_rows(rng.standard_normal((count, 3)))
    # fill_state presents propagate_at_boundary with the normal facing inward.
    normal[np.sum(direction * normal, axis=1, dtype=np.float32) > 0.0] *= -1.0
    media = np.asarray((1.0, 1.23, 1.33, 1.3784, 1.49, 1.525), np.float32)
    n1 = media[rng.randint(0, len(media), size=count)]
    n2 = media[rng.randint(0, len(media), size=count)]
    u_polarization = rng.random_sample(count).astype(np.float32)
    u_reflect = rng.random_sample(count).astype(np.float32)

    # Photon 582 from detector_lockstep_one_step.cuda.npz.  Keeping it first
    # makes the original PMT-transmission discrepancy a permanent regression.
    direction[0] = _f32_words(0xBF4B3748, 0xBF0760FF, 0x3E99C6F0)
    polarization[0] = _f32_words(0xBEF80C73, 0x3F59308A, 0x3E5A7472)
    normal[0] = _f32_words(0x3F1DF2C4, 0x3F4748DF, 0xBDEC80DD)
    n1[0] = _f32_words(0x3FB06F69)[0]
    n2[0] = _f32_words(0x3FC33333)[0]
    u_polarization[0] = _f32_words(0x3F7F3BFC)[0]
    u_reflect[0] = _f32_words(0x3D806590)[0]
    return {
        "direction": np.ascontiguousarray(direction),
        "polarization": np.ascontiguousarray(polarization),
        "normal": np.ascontiguousarray(normal),
        "n1": np.ascontiguousarray(n1),
        "n2": np.ascontiguousarray(n2),
        "u_polarization": np.ascontiguousarray(u_polarization),
        "u_reflect": np.ascontiguousarray(u_reflect),
    }


def _pick(archive: np.lib.npyio.NpzFile, *names: str) -> np.ndarray | None:
    for name in names:
        if name in archive:
            return np.asarray(archive[name])
    return None


def _inputs(count: int, seed: int, input_path: str | None) -> dict[str, np.ndarray]:
    if input_path is None:
        return _generated_inputs(count, seed)
    with np.load(Path(input_path).expanduser().resolve()) as archive:
        direction = _pick(archive, "direction", "normalized_direction")
        polarization = _pick(
            archive, "polarization", "normalized_polarization"
        )
        normal = _pick(archive, "normal", "surface_normal")
        if direction is None or polarization is None or normal is None:
            raise ValueError(
                "input NPZ needs direction, polarization, and normal arrays"
            )
        available = min(len(direction), len(polarization), len(normal))
        count = min(count, available)
        values = _generated_inputs(count, seed)
        values["direction"] = np.ascontiguousarray(direction[:count], np.float32)
        values["polarization"] = np.ascontiguousarray(
            polarization[:count], np.float32
        )
        values["normal"] = np.ascontiguousarray(normal[:count], np.float32)
        aliases = {
            "n1": ("n1", "refractive_index1"),
            "n2": ("n2", "refractive_index2"),
            "u_polarization": ("u_polarization",),
            "u_reflect": ("u_reflect",),
        }
        for output_name, input_names in aliases.items():
            candidate = _pick(archive, *input_names)
            if candidate is not None:
                candidate = np.asarray(candidate, np.float32).reshape(-1)
                if candidate.size == 1:
                    candidate = np.full(count, candidate[0], np.float32)
                elif candidate.size < count:
                    raise ValueError(f"{input_names[0]} has too few rows")
                values[output_name] = np.ascontiguousarray(candidate[:count])
        draws = _pick(archive, "random_draws", "draws")
        if draws is not None:
            draws = np.asarray(draws, np.float32).reshape(-1, 2)
            if len(draws) < count:
                raise ValueError("random_draws has too few rows")
            values["u_polarization"] = np.ascontiguousarray(draws[:count, 0])
            values["u_reflect"] = np.ascontiguousarray(draws[:count, 1])
        if count == 0:
            raise ValueError("input NPZ contains no rows")
        return values


def _cuda_worker(args: argparse.Namespace, output: Path) -> None:
    os.environ.setdefault("PYCUDA_CACHE_DIR", "/tmp/chroma-pycuda-fresnel")
    Path(os.environ["PYCUDA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    import pycuda.autoprimaryctx  # noqa: F401
    from pycuda import driver as cuda
    from pycuda import gpuarray as ga

    from chroma.gpu.tools import cuda_options, get_cu_module, to_float3

    inputs = _inputs(args.count, args.seed, args.input)
    count = len(inputs["direction"])
    direction_gpu = ga.to_gpu(to_float3(inputs["direction"]))
    polarization_gpu = ga.to_gpu(to_float3(inputs["polarization"]))
    normal_gpu = ga.to_gpu(to_float3(inputs["normal"]))
    scalar_gpu = {
        name: ga.to_gpu(inputs[name])
        for name in ("n1", "n2", "u_polarization", "u_reflect")
    }
    output_direction = ga.empty_like(direction_gpu)
    output_polarization = ga.empty_like(polarization_gpu)
    diagnostics = ga.empty((count, len(DIAGNOSTIC_NAMES)), np.float32)
    decisions = ga.empty((count, 3), np.uint32)
    module = get_cu_module("fresnel_probe.cu", options=cuda_options)
    block = 256
    module.get_function("fresnel_probe")(
        np.int32(count), direction_gpu, polarization_gpu, normal_gpu,
        scalar_gpu["n1"], scalar_gpu["n2"], scalar_gpu["u_polarization"],
        scalar_gpu["u_reflect"], output_direction, output_polarization,
        diagnostics, decisions,
        block=(block, 1, 1), grid=((count + block - 1) // block, 1, 1),
    )
    cuda.Context.synchronize()
    np.savez(
        output,
        **inputs,
        output_direction=output_direction.get().view(np.float32).reshape(-1, 3),
        output_polarization=(
            output_polarization.get().view(np.float32).reshape(-1, 3)
        ),
        diagnostics=diagnostics.get(),
        decisions=decisions.get(),
    )


def _triton_worker(args: argparse.Namespace, output: Path) -> None:
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    from chroma.triton.physics_kernels import (
        acos_chroma_fast,
        asin_chroma_fast,
        fresnel_incident_cosine_chroma_fast,
        fresnel_step_chroma,
    )

    globals().update(
        torch=torch,
        triton=triton,
        tl=tl,
        libdevice=libdevice,
        acos_chroma_fast=acos_chroma_fast,
        asin_chroma_fast=asin_chroma_fast,
        fresnel_incident_cosine_chroma_fast=(
            fresnel_incident_cosine_chroma_fast
        ),
        fresnel_step_chroma=fresnel_step_chroma,
    )

    @triton.jit
    def probe_kernel(
        direction, polarization, normal, n1, n2, u_polarization, u_reflect,
        output_direction, output_polarization, diagnostics, decisions, count,
        BLOCK: tl.constexpr,
    ):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < count
        base = lane * 3
        dx = tl.load(direction + base, mask=valid, other=1.0)
        dy = tl.load(direction + base + 1, mask=valid, other=0.0)
        dz = tl.load(direction + base + 2, mask=valid, other=0.0)
        px = tl.load(polarization + base, mask=valid, other=0.0)
        py = tl.load(polarization + base + 1, mask=valid, other=1.0)
        pz = tl.load(polarization + base + 2, mask=valid, other=0.0)
        nx = tl.load(normal + base, mask=valid, other=-1.0)
        ny = tl.load(normal + base + 1, mask=valid, other=0.0)
        nz = tl.load(normal + base + 2, mask=valid, other=0.0)
        index1 = tl.load(n1 + lane, mask=valid, other=1.0)
        index2 = tl.load(n2 + lane, mask=valid, other=1.0)
        random_polarization = tl.load(
            u_polarization + lane, mask=valid, other=0.0
        )
        random_reflect = tl.load(u_reflect + lane, mask=valid, other=0.0)

        (
            out_dx, out_dy, out_dz, out_px, out_py, out_pz,
            reflected, reflectance, normal_probability, tir,
        ) = fresnel_step_chroma(
            dx, dy, dz, px, py, pz, nx, ny, nz, index1, index2,
            random_polarization, random_reflect,
        )

        incident_cosine = fresnel_incident_cosine_chroma_fast(
            dx, dy, dz, nx, ny, nz
        )
        incident_cosine = tl.maximum(-1.0, tl.minimum(1.0, incident_cosine))
        incident_angle = acos_chroma_fast(incident_cosine)
        refracted_argument = (
            libdevice.fast_sinf(incident_angle) * index1
        ) / index2
        refracted_angle = asin_chroma_fast(refracted_argument)
        axis_x = dy * nz - dz * ny
        axis_y = dz * nx - dx * nz
        axis_z = dx * ny - dy * nx
        axis_norm2 = tl.fma(
            axis_z, axis_z, tl.fma(axis_x, axis_x, axis_y * axis_y)
        )
        raw_axis_length = tl.sqrt(axis_norm2)
        normal_incidence = raw_axis_length < 1.0e-6
        axis_x = tl.where(normal_incidence, px, axis_x / raw_axis_length)
        axis_y = tl.where(normal_incidence, py, axis_y / raw_axis_length)
        axis_z = tl.where(normal_incidence, pz, axis_z / raw_axis_length)
        normal_coefficient = tl.fma(
            axis_z, pz, tl.fma(axis_x, px, axis_y * py)
        )
        diagnostic_normal_probability = normal_coefficient * normal_coefficient
        choose_normal = random_polarization < diagnostic_normal_probability
        difference = incident_angle - refracted_angle
        angle_sum = incident_angle + refracted_angle
        normal_reflection = -libdevice.fast_sinf(difference) / (
            libdevice.fast_sinf(angle_sum)
        )
        parallel_reflection = libdevice.fast_tanf(difference) / (
            libdevice.fast_tanf(angle_sum)
        )
        reflection_coefficient = tl.where(
            choose_normal, normal_reflection, parallel_reflection
        )
        pi = tl.full(dx.shape, 0x40490FDB, tl.uint32).to(
            tl.float32, bitcast=True
        )
        outgoing_angle = tl.where(reflected, incident_angle, pi - refracted_angle)

        tl.store(output_direction + base, out_dx, mask=valid)
        tl.store(output_direction + base + 1, out_dy, mask=valid)
        tl.store(output_direction + base + 2, out_dz, mask=valid)
        tl.store(output_polarization + base, out_px, mask=valid)
        tl.store(output_polarization + base + 1, out_py, mask=valid)
        tl.store(output_polarization + base + 2, out_pz, mask=valid)
        diagnostic_base = lane * 12
        tl.store(diagnostics + diagnostic_base, incident_cosine, mask=valid)
        tl.store(diagnostics + diagnostic_base + 1, incident_angle, mask=valid)
        tl.store(
            diagnostics + diagnostic_base + 2, refracted_argument, mask=valid
        )
        tl.store(diagnostics + diagnostic_base + 3, refracted_angle, mask=valid)
        tl.store(diagnostics + diagnostic_base + 4, raw_axis_length, mask=valid)
        tl.store(diagnostics + diagnostic_base + 5, axis_x, mask=valid)
        tl.store(diagnostics + diagnostic_base + 6, axis_y, mask=valid)
        tl.store(diagnostics + diagnostic_base + 7, axis_z, mask=valid)
        tl.store(
            diagnostics + diagnostic_base + 8,
            diagnostic_normal_probability,
            mask=valid,
        )
        tl.store(
            diagnostics + diagnostic_base + 9,
            reflection_coefficient,
            mask=valid,
        )
        tl.store(diagnostics + diagnostic_base + 10, reflectance, mask=valid)
        tl.store(diagnostics + diagnostic_base + 11, outgoing_angle, mask=valid)
        tl.store(decisions + base, choose_normal.to(tl.int32), mask=valid)
        tl.store(decisions + base + 1, reflected.to(tl.int32), mask=valid)
        tl.store(decisions + base + 2, tir.to(tl.int32), mask=valid)

    inputs = _inputs(args.count, args.seed, args.input)
    count = len(inputs["direction"])
    device = {
        name: torch.as_tensor(value, device="cuda")
        for name, value in inputs.items()
    }
    output_direction = torch.empty_like(device["direction"])
    output_polarization = torch.empty_like(device["polarization"])
    diagnostics = torch.empty(
        (count, len(DIAGNOSTIC_NAMES)), dtype=torch.float32, device="cuda"
    )
    decisions = torch.empty((count, 3), dtype=torch.int32, device="cuda")
    block = 256
    probe_kernel[(triton.cdiv(count, block),)](
        device["direction"], device["polarization"], device["normal"],
        device["n1"], device["n2"], device["u_polarization"],
        device["u_reflect"], output_direction, output_polarization,
        diagnostics, decisions, count, BLOCK=block, num_warps=8,
    )
    torch.cuda.synchronize()
    np.savez(
        output,
        **inputs,
        output_direction=output_direction.cpu().numpy(),
        output_polarization=output_polarization.cpu().numpy(),
        diagnostics=diagnostics.cpu().numpy(),
        decisions=decisions.cpu().numpy().astype(np.uint32),
    )


def _raw_agreement(cuda: np.ndarray, triton_values: np.ndarray) -> dict:
    cuda = np.ascontiguousarray(cuda, np.float32)
    triton_values = np.ascontiguousarray(triton_values, np.float32)
    cuda_words = cuda.view(np.uint32).reshape(len(cuda), -1)
    triton_words = triton_values.view(np.uint32).reshape(len(cuda), -1)
    different = cuda_words != triton_words
    row_exact = ~np.any(different, axis=1)
    first = np.argwhere(different)
    first_detail = None
    if len(first):
        row, component = map(int, first[0])
        cuda_flat = cuda.reshape(len(cuda), -1)
        triton_flat = triton_values.reshape(len(cuda), -1)
        first_detail = {
            "row": row,
            "component": component,
            "cuda_value": float(cuda_flat[row, component]),
            "triton_value": float(triton_flat[row, component]),
            "cuda_word": f"0x{int(cuda_words[row, component]):08x}",
            "triton_word": f"0x{int(triton_words[row, component]):08x}",
        }
    finite = np.isfinite(cuda) & np.isfinite(triton_values)
    maximum = (
        float(np.max(np.abs(cuda[finite].astype(np.float64)
                            - triton_values[finite].astype(np.float64))))
        if np.any(finite) else None
    )
    return {
        "exact_rows": int(np.count_nonzero(row_exact)),
        "mismatched_rows": int(np.count_nonzero(~row_exact)),
        "exact_row_fraction": float(np.mean(row_exact)),
        "mismatched_components": int(np.count_nonzero(different)),
        "maximum_absolute_difference": maximum,
        "first_difference": first_detail,
    }


def _report(cuda: np.lib.npyio.NpzFile, triton_values: np.lib.npyio.NpzFile,
            seed: int, input_path: str | None) -> dict:
    cuda_diagnostics = cuda["diagnostics"]
    triton_diagnostics = triton_values["diagnostics"]
    fields = {
        name: _raw_agreement(
            cuda_diagnostics[:, index:index + 1],
            triton_diagnostics[:, index:index + 1],
        )
        for index, name in enumerate(DIAGNOSTIC_NAMES)
    }
    group_arrays = {
        "incident_angle": (
            cuda_diagnostics[:, 1:2], triton_diagnostics[:, 1:2]
        ),
        "refracted_argument_and_angle": (
            cuda_diagnostics[:, 2:4], triton_diagnostics[:, 2:4]
        ),
        "normalized_incidence_axis": (
            cuda_diagnostics[:, 5:8], triton_diagnostics[:, 5:8]
        ),
        "output_direction": (
            cuda["output_direction"], triton_values["output_direction"]
        ),
        "output_polarization": (
            cuda["output_polarization"], triton_values["output_polarization"]
        ),
    }
    groups = {
        name: _raw_agreement(cuda_array, triton_array)
        for name, (cuda_array, triton_array) in group_arrays.items()
    }
    decision_equal = cuda["decisions"] == triton_values["decisions"]
    regression_exact = {
        name: bool(_raw_agreement(left[:1], right[:1])["mismatched_rows"] == 0)
        for name, (left, right) in group_arrays.items()
    }
    input_words = {}
    for name in (
        "direction", "polarization", "normal", "n1", "n2",
        "u_polarization", "u_reflect",
    ):
        row = np.ascontiguousarray(np.asarray(cuda[name][0], np.float32))
        input_words[name] = [f"0x{int(word):08x}" for word in row.view(np.uint32).flat]
    return {
        "schema_version": 1,
        "operation": "Chroma propagate_at_boundary Fresnel step",
        "seed": int(seed),
        "input": str(Path(input_path).resolve()) if input_path else "generated",
        "rays": int(len(cuda_diagnostics)),
        "required_group_raw_word_agreement": groups,
        "intermediate_raw_word_agreement": fields,
        "decision_agreement": {
            "exact_rows": int(np.count_nonzero(np.all(decision_equal, axis=1))),
            "mismatched_rows": int(np.count_nonzero(~np.all(decision_equal, axis=1))),
            "mismatched_components": int(np.count_nonzero(~decision_equal)),
        },
        ("pinned_detector_regression_row_0" if input_path is None else
         "input_row_0"): {
            "required_groups_exact": regression_exact,
            "input_words": input_words,
        },
    }


def _worker_command(args: argparse.Namespace, backend: str, output: Path) -> list[str]:
    worker = [
        str(Path(__file__).resolve()), "--count", str(args.count),
        "--seed", str(args.seed), "--_worker", backend,
        "--_output", str(output),
    ]
    if args.input:
        worker.extend(("--input", str(Path(args.input).expanduser().resolve())))
    if backend == "cuda":
        image = Path(args.chroma_container).expanduser().resolve()
        if image.is_file():
            python_path = f"{REPOSITORY / 'chroma-lite'}:{REPOSITORY / 'chroma-lar'}"
            return [
                "singularity", "exec", "--nv", "-B", "/sdf:/sdf",
                "-B", "/tmp:/tmp", "--pwd", str(REPOSITORY), str(image),
                "env", "PYTHONPATH=" + python_path, "PYTHONNOUSERSITE=1",
                "PYCUDA_CACHE_DIR=/tmp/chroma-pycuda-fresnel", "TMPDIR=/tmp",
                "python", *worker,
            ]
    return [sys.executable, *worker]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--input", default=None, help="optional exact input NPZ")
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--npz-prefix", default=None,
        help="retain worker arrays as PREFIX.cuda.npz and PREFIX.triton.npz",
    )
    parser.add_argument("--chroma-container", default=str(DEFAULT_CHROMA_CONTAINER))
    parser.add_argument("--_worker", choices=("cuda", "triton"), default=None)
    parser.add_argument("--_output", default=None)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.input and not Path(args.input).expanduser().is_file():
        parser.error("--input does not exist")
    if args._worker:
        output = Path(args._output)
        if args._worker == "cuda":
            _cuda_worker(args, output)
        else:
            _triton_worker(args, output)
        return 0

    with tempfile.TemporaryDirectory(prefix="fresnel-probe-", dir="/tmp") as temp:
        cuda_path = Path(temp) / "cuda.npz"
        triton_path = Path(temp) / "triton.npz"
        for backend, path in (("cuda", cuda_path), ("triton", triton_path)):
            completed = subprocess.run(
                _worker_command(args, backend, path), cwd=str(REPOSITORY),
                text=True, capture_output=True, check=False,
            )
            if completed.returncode or not path.exists():
                diagnostic = (completed.stdout + "\n" + completed.stderr)[-12_000:]
                raise RuntimeError(
                    f"{backend} worker failed with status "
                    f"{completed.returncode}:\n{diagnostic}"
                )
        with np.load(cuda_path) as cuda, np.load(triton_path) as triton_values:
            report = _report(cuda, triton_values, args.seed, args.input)
        if args.npz_prefix:
            prefix = Path(args.npz_prefix).expanduser().resolve()
            prefix.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cuda_path, Path(str(prefix) + ".cuda.npz"))
            shutil.copy2(triton_path, Path(str(prefix) + ".triton.npz"))
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        output = Path(args.json).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
