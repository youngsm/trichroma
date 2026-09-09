#!/usr/bin/env python3
"""Run the certificate-only Triton traversal against Chroma oracle rays.

The artifact must be exported by the Chroma/PyCUDA environment first.  This
process only reads the immutable NPZ and uses Torch/Triton, which avoids
mixing PyCUDA's primary context with Torch's CUDA context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from chroma_lar.triton_scene import (
    TARGET_OPTICAL_SEMANTICS_SHA256,
    TARGET_TRAVERSAL_SHA256,
    ChromaGlobalBVHDevice,
    load_chroma_global_bvh_artifact,
    nearest_chroma_global_hit,
)
from chroma_lar.triton_scene.chroma_global_bvh import TARGET_MESH_MD5


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _bits(value: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(value)
    if value.dtype == np.float32:
        return value.view(np.uint32)
    if value.dtype == np.float64:
        return value.view(np.uint64)
    return value


def _first_bad_rows(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    unequal = _bits(actual) != _bits(expected)
    if unequal.ndim > 1:
        unequal = np.any(unequal.reshape(unequal.shape[0], -1), axis=1)
    return np.flatnonzero(unequal)


def _compare(
    output: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> tuple[dict[str, Any], bool]:
    mapping = {
        "triangle_ids": "triangle",
        "distances": "distance",
        "surface_normals": "normal",
        "solid_ids": "solid",
        "channel_ids": "channel",
        "inside_to_outside": "inside_to_outside",
    }
    report: dict[str, Any] = {}
    all_exact = True
    missing = [name for name in mapping.values() if name not in expected]
    if missing:
        report["expected_schema"] = {
            "exact": False,
            "missing_required_fields": missing,
        }
        all_exact = False
    for actual_name, expected_name in mapping.items():
        if expected_name not in expected:
            continue
        actual = output[actual_name]
        reference = expected[expected_name].astype(actual.dtype, copy=False)
        if actual.shape != reference.shape:
            report[actual_name] = {
                "exact": False,
                "actual_shape": list(actual.shape),
                "expected_shape": list(reference.shape),
            }
            all_exact = False
            continue
        bad = _first_bad_rows(actual, reference)
        report[actual_name] = {
            "exact": not len(bad),
            "exact_rows": int(len(actual) - len(bad)),
            "rows": int(len(actual)),
            "first_bad_rows": bad[:10].tolist(),
        }
        all_exact &= not len(bad)
    return report, all_exact


def _to_numpy(result: Any) -> dict[str, np.ndarray]:
    return {
        name: getattr(result, name).detach().cpu().numpy()
        for name in result.__dataclass_fields__
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--rays", required=True, type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ray-tile", type=int, default=8192)
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()

    artifact = load_chroma_global_bvh_artifact(
        args.artifact,
        expected_mesh_md5=TARGET_MESH_MD5,
        expected_traversal_sha256=TARGET_TRAVERSAL_SHA256,
        expected_optical_semantics_sha256=TARGET_OPTICAL_SEMANTICS_SHA256,
    )
    rays = _read_npz(args.rays)
    if "origins" not in rays or "directions" not in rays:
        raise ValueError("ray archive must contain origins and directions")
    accelerator = ChromaGlobalBVHDevice.from_host(
        artifact,
        expected_traversal_sha256=TARGET_TRAVERSAL_SHA256,
    )
    result = nearest_chroma_global_hit(
        accelerator,
        rays["origins"],
        rays["directions"],
        last_triangle=rays.get("last_global_triangle"),
        ray_tile=args.ray_tile,
    )
    output = _to_numpy(result)
    if args.output is not None:
        np.savez(
            args.output,
            artifact_mesh_md5=np.asarray(artifact.mesh_md5),
            artifact_traversal_sha256=np.asarray(artifact.traversal_sha256),
            artifact_optical_semantics_sha256=np.asarray(
                artifact.optical_semantics_sha256
            ),
            **output,
        )

    summary: dict[str, Any] = {
        "artifact": str(args.artifact),
        "mesh_md5": artifact.mesh_md5,
        "traversal_sha256": artifact.traversal_sha256,
        "optical_semantics_sha256": artifact.optical_semantics_sha256,
        "rays": str(args.rays),
        "ray_count": int(len(rays["origins"])),
        "hit_count": int(np.count_nonzero(output["triangle_ids"] >= 0)),
        "overflow_count": int(np.count_nonzero(output["overflow"])),
    }
    exact = True
    if args.expected is not None:
        comparison, exact = _compare(output, _read_npz(args.expected))
        summary["expected"] = str(args.expected)
        summary["comparison"] = comparison
        summary["all_compared_fields_exact"] = exact
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if exact or args.allow_mismatch else 1


if __name__ == "__main__":
    raise SystemExit(main())
