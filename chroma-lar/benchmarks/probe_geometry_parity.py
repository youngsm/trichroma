#!/usr/bin/env python
"""Compare specialized first-boundary queries with Chroma's full geometry.

This is a diagnostic rather than a Monte-Carlo acceptance benchmark.  The two
backends run in separate processes because PyCUDA and Torch should not share a
context.  An input ``.npz`` contains float32 ``origins`` and ``directions``;
each backend writes a compact boundary record which can then be compared with
``--backend compare``.

The Chroma path calls ``fill_state`` itself, so its result includes both the
flattened 829k-triangle BVH and all six analytic wire planes.  It is therefore
independent of the specialized scene decomposition.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
for source_root in (REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


MISS = np.int32(0)
PMT = np.int32(1)
ACTIVE = np.int32(2)
CATHODE = np.int32(3)
CAVITY = np.int32(4)
WIRE = np.int32(5)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rays(path):
    data = np.load(path)
    origins = np.ascontiguousarray(data["origins"], dtype=np.float32)
    directions = np.ascontiguousarray(data["directions"], dtype=np.float32)
    if origins.ndim != 2 or origins.shape[1] != 3 or directions.shape != origins.shape:
        raise ValueError("origins and directions must both have shape (N,3)")
    count = len(origins)
    last_global = np.ascontiguousarray(
        data["last_global_triangle"] if "last_global_triangle" in data
        else np.full(count, -1, dtype=np.int32), dtype=np.int32
    )
    last_instance = np.ascontiguousarray(
        data["last_instance"] if "last_instance" in data
        else np.full(count, -1, dtype=np.int32), dtype=np.int32
    )
    last_local = np.ascontiguousarray(
        data["last_local_triangle"] if "last_local_triangle" in data
        else np.full(count, -1, dtype=np.int32), dtype=np.int32
    )
    return origins, directions, last_global, last_instance, last_local


def _run_chroma(input_path, output_path, artifact_path=None):
    os.environ.setdefault("PYCUDA_CACHE_DIR", "/tmp/chroma-pycuda-geometry-probe")
    os.makedirs(os.environ["PYCUDA_CACHE_DIR"], exist_ok=True)

    import pycuda.compiler
    from pycuda import gpuarray as ga
    import chroma.gpu
    from chroma.gpu.tools import cuda_options, to_float3
    from chroma.cuda import srcdir
    from chroma_lar.geometry import build_detector_from_config

    # Compatibility with NumPy 2 in the host validation environment.
    if not hasattr(np.linalg, "linalg"):
        np.linalg.linalg = np.linalg

    origins, directions, last_global, _, _ = _load_rays(input_path)
    detector = build_detector_from_config(
        "detector_config_reflect_reflect3wires",
        flatten=True,
        include_wires=True,
        include_active=True,
        include_cathode=True,
        include_cavity=True,
    )
    if artifact_path is not None:
        from chroma_lar.triton_scene.chroma_global_bvh import (
            build_chroma_global_bvh_artifact,
            save_chroma_global_bvh_artifact,
        )

        artifact = build_chroma_global_bvh_artifact(detector)
        save_chroma_global_bvh_artifact(artifact, artifact_path)
    retained_pmt = (
        (detector.solid_id >= np.uint32(1))
        & (detector.solid_id <= np.uint32(81))
    )
    pmt_triangle_words = np.ascontiguousarray(
        detector.mesh.vertices[detector.mesh.triangles[retained_pmt]],
        dtype=np.float32,
    )
    pmt_triangle_sha256 = hashlib.sha256(
        pmt_triangle_words.tobytes(order="C")
    ).hexdigest()
    context = chroma.gpu.create_cuda_context()
    try:
        geometry = chroma.gpu.GPUGeometry(detector, print_usage=False)
        source = r'''
#include "photon.h"
extern "C" __global__ void query_fill_state(
    int n, float3 *origins, float3 *directions, Geometry *geometry,
    int *input_last_triangle, float *distance, float3 *normal,
    int *triangle, int *surface,
    unsigned char *inside_to_outside, unsigned short *history,
    float *refractive_index1, float *refractive_index2,
    float *absorption_length,
    float *scattering_length)
{
    __shared__ Geometry shared_geometry;
    if (threadIdx.x == 0) shared_geometry = *geometry;
    __syncthreads();
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    Photon photon;
    photon.position = origins[id];
    photon.direction = directions[id];
    photon.polarization = make_float3(0.0f, 1.0f, 0.0f);
    photon.wavelength = 450.0f;
    photon.time = 0.0f;
    photon.weight = 1.0f;
    photon.history = 0;
    photon.last_hit_triangle = input_last_triangle[id];
    photon.evidx = id;
    State state;
    fill_state(state, photon, &shared_geometry);
    bool hit = (photon.history & NO_HIT) == 0;
    distance[id] = hit ? state.distance_to_boundary : __int_as_float(0x7f800000);
    normal[id] = hit ? state.surface_normal : make_float3(0.0f, 0.0f, 0.0f);
    triangle[id] = hit ? photon.last_hit_triangle : -1;
    surface[id] = hit ? state.surface_index : -1;
    inside_to_outside[id] = hit && state.inside_to_outside;
    history[id] = photon.history;
    refractive_index1[id] = hit ? state.refractive_index1 : 0.0f;
    refractive_index2[id] = hit ? state.refractive_index2 : 0.0f;
    absorption_length[id] = hit ? state.absorption_length : 0.0f;
    scattering_length[id] = hit ? state.scattering_length : 0.0f;
}
'''
        module = pycuda.compiler.SourceModule(
            source,
            options=list(cuda_options) + ["-I" + srcdir],
            no_extern_c=True,
        )
        kernel = module.get_function("query_fill_state")
        origins_gpu = ga.to_gpu(to_float3(origins))
        directions_gpu = ga.to_gpu(to_float3(directions))
        last_global_gpu = ga.to_gpu(last_global)
        n = len(origins)
        distance_gpu = ga.empty(n, np.float32)
        normal_gpu = ga.empty(n, ga.vec.float3)
        triangle_gpu = ga.empty(n, np.int32)
        surface_gpu = ga.empty(n, np.int32)
        inside_gpu = ga.empty(n, np.uint8)
        history_gpu = ga.empty(n, np.uint16)
        refractive_index1_gpu = ga.empty(n, np.float32)
        refractive_index2_gpu = ga.empty(n, np.float32)
        absorption_length_gpu = ga.empty(n, np.float32)
        scattering_length_gpu = ga.empty(n, np.float32)
        block = 128
        kernel(
            np.int32(n), origins_gpu, directions_gpu, geometry.gpudata,
            last_global_gpu,
            distance_gpu, normal_gpu, triangle_gpu, surface_gpu,
            inside_gpu, history_gpu, refractive_index1_gpu,
            refractive_index2_gpu,
            absorption_length_gpu, scattering_length_gpu,
            block=(block, 1, 1), grid=((n + block - 1) // block, 1, 1),
        )
        distance = distance_gpu.get()
        normal = normal_gpu.get().view(np.float32).reshape(-1, 3)
        triangle = triangle_gpu.get()
        surface = surface_gpu.get()
        inside = inside_gpu.get()
        history = history_gpu.get()
        refractive_index1 = refractive_index1_gpu.get()
        refractive_index2 = refractive_index2_gpu.get()
        absorption_length = absorption_length_gpu.get()
        scattering_length = scattering_length_gpu.get()
    finally:
        context.pop()

    solid = np.full(len(triangle), -1, dtype=np.int32)
    mesh_hit = triangle >= 0
    solid[mesh_hit] = detector.solid_id[triangle[mesh_hit]].astype(np.int32)
    kind = np.full(len(triangle), MISS, dtype=np.int32)
    kind[triangle == -2] = WIRE
    kind[(solid >= 1) & (solid <= 162)] = PMT
    kind[solid == 163] = ACTIVE
    kind[solid == 164] = CATHODE
    kind[solid == 0] = CAVITY
    channel = np.where(kind == PMT, solid - 1, -1).astype(np.int32)
    surface_name = np.full(len(surface), "<none>", dtype="U64")
    surface_hit = surface >= 0
    if np.any(surface_hit):
        surface_name[surface_hit] = np.asarray(
            [
                str(getattr(detector.unique_surfaces[int(index)], "name", ""))
                for index in surface[surface_hit]
            ],
            dtype="U64",
        )
    np.savez(
        output_path, distance=distance, normal=normal, triangle=triangle,
        surface=surface, inside_to_outside=inside, history=history,
        solid=solid, kind=kind, channel=channel,
        pmt_triangle_sha256=np.asarray(pmt_triangle_sha256),
        surface_name=surface_name,
        refractive_index1=refractive_index1,
        refractive_index2=refractive_index2,
        absorption_length=absorption_length,
        scattering_length=scattering_length,
    )


def _run_triton(input_path, output_path, artifact_path):
    import torch
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation
    from chroma_lar.triton_scene import nearest_chroma_global_hit
    from chroma_lar.triton_scene.intersect import intersect_scene_triton

    origins, directions, last_global, _, _ = _load_rays(input_path)
    simulation = Reflect3WiresTritonSimulation(
        tile_size=max(1, len(origins)),
        chroma_mesh_box_compatibility=True,
        chroma_global_bvh_artifact=artifact_path,
    )
    origins_gpu = torch.from_numpy(origins).cuda()
    directions_gpu = torch.from_numpy(directions).cuda()
    last_global_gpu = torch.from_numpy(last_global).cuda()
    simulation._ensure_boundary_workspace(len(origins))
    mesh = nearest_chroma_global_hit(
        simulation.chroma_global_accelerator,
        origins_gpu,
        directions_gpu,
        last_triangle=last_global_gpu,
        ray_tile=max(1, len(origins)),
        workspace=simulation.chroma_global_workspace,
    )
    wire_tmax = simulation.boundary_ray_workspace.pmt_tmax[:len(origins)]
    wire_tmax.copy_(mesh.distances)
    wire_tmax.masked_fill_(mesh.triangle_ids < 0, 1.0e30)
    analytic = intersect_scene_triton(
        simulation.analytic_scene, origins_gpu, directions_gpu,
        tmax=wire_tmax,
        out=simulation.analytic_workspace.outputs(len(origins)),
        chroma_wire_frame=True,
        chroma_wire_full_scan=True,
        _box_count_override=0,
    )
    host_artifact = simulation.chroma_global_host_artifact
    retained_pmt = (
        (host_artifact.solid_id >= np.int32(1))
        & (host_artifact.solid_id <= np.int32(81))
    )
    pmt_triangle_words = np.ascontiguousarray(
        host_artifact.vertices[host_artifact.triangles[retained_pmt]],
        dtype=np.float32,
    )
    pmt_triangle_sha256 = hashlib.sha256(
        pmt_triangle_words.tobytes(order="C")
    ).hexdigest()
    merged = simulation._merge_chroma_global_boundaries(analytic, mesh)
    distance, normal, material_from, material_to, surface, instance, triangle, channel = merged
    choose_wire = triangle == -2
    choose_mesh = triangle >= 0
    kind = torch.zeros_like(triangle)
    kind = torch.where(choose_wire, int(WIRE), kind)
    solid = mesh.solid_ids
    kind = torch.where(choose_mesh & (solid >= 1) & (solid <= 162), int(PMT), kind)
    kind = torch.where(choose_mesh & (solid == 163), int(ACTIVE), kind)
    kind = torch.where(choose_mesh & (solid == 164), int(CATHODE), kind)
    kind = torch.where(choose_mesh & (solid == 0), int(CAVITY), kind)
    inside_to_outside = torch.where(
        choose_mesh,
        mesh.inside_to_outside.to(torch.bool),
        torch.where(
            choose_wire,
            analytic.inside_to_outside.to(torch.bool),
            torch.zeros_like(choose_mesh),
        ),
    )
    hit = choose_wire | choose_mesh
    safe_material_from = torch.where(
        hit, material_from, torch.zeros_like(material_from)
    )
    safe_material_to = torch.where(
        hit, material_to, torch.zeros_like(material_to)
    )
    material_refractive_index = simulation.scene_device[
        "tables_material_refractive_index"
    ]
    material_absorption_length = simulation.scene_device[
        "tables_material_absorption_length"
    ]
    material_scattering_length = simulation.scene_device[
        "tables_material_scattering_length"
    ]
    refractive_index1 = torch.where(
        hit, material_refractive_index[safe_material_from], 0.0
    )
    refractive_index2 = torch.where(
        hit, material_refractive_index[safe_material_to], 0.0
    )
    absorption_length = torch.where(
        hit, material_absorption_length[safe_material_from], 0.0
    )
    scattering_length = torch.where(
        hit, material_scattering_length[safe_material_from], 0.0
    )
    torch.cuda.synchronize()
    surface_host = surface.cpu().numpy()
    surface_name = np.full(len(surface_host), "<none>", dtype="U64")
    surface_hit = surface_host >= 0
    if np.any(surface_hit):
        surface_names = simulation.scene.tables.surface_names
        surface_name[surface_hit] = np.asarray(
            [surface_names[int(index)] for index in surface_host[surface_hit]],
            dtype="U64",
        )
    selected_solid = torch.where(choose_mesh, solid, -1)
    history = torch.where(
        hit,
        torch.zeros_like(triangle, dtype=torch.int32),
        torch.full_like(triangle, 1, dtype=torch.int32),
    )
    np.savez(
        output_path,
        distance=distance.cpu().numpy(), normal=normal.cpu().numpy(),
        triangle=triangle.cpu().numpy(), surface=surface.cpu().numpy(),
        inside_to_outside=inside_to_outside.cpu().numpy(),
        history=history.cpu().numpy().astype(np.uint16, copy=False),
        solid=selected_solid.cpu().numpy(),
        kind=kind.cpu().numpy(), channel=channel.cpu().numpy(),
        surface_name=surface_name,
        refractive_index1=refractive_index1.cpu().numpy(),
        refractive_index2=refractive_index2.cpu().numpy(),
        absorption_length=absorption_length.cpu().numpy(),
        scattering_length=scattering_length.cpu().numpy(),
        pmt_triangle_sha256=np.asarray(pmt_triangle_sha256),
        analytic_kind=analytic.kind.cpu().numpy(),
        analytic_index=analytic.index.cpu().numpy(),
        analytic_primitive=analytic.primitive_index.cpu().numpy(),
    )


def _compare(
    chroma_path,
    triton_path,
    json_path=None,
    artifact_path=None,
    input_path=None,
):
    chroma = np.load(chroma_path)
    triton = np.load(triton_path)
    if chroma["kind"].shape != triton["kind"].shape:
        raise ValueError("backend output sizes differ")
    kind_equal = chroma["kind"] == triton["kind"]
    channel_equal = chroma["channel"] == triton["channel"]
    finite = np.isfinite(chroma["distance"]) & np.isfinite(triton["distance"])
    difference = np.abs(chroma["distance"] - triton["distance"])
    distance_equal = (
        np.ascontiguousarray(chroma["distance"], dtype=np.float32).view(np.uint32)
        == np.ascontiguousarray(triton["distance"], dtype=np.float32).view(np.uint32)
    )
    normal_equal = np.all(
        np.ascontiguousarray(chroma["normal"]).view(np.uint32)
        == np.ascontiguousarray(triton["normal"]).view(np.uint32),
        axis=1,
    )
    exact_fields = {
        "kind": kind_equal,
        "channel": channel_equal,
        "triangle": chroma["triangle"] == triton["triangle"],
        "solid": chroma["solid"] == triton["solid"],
        "distance": distance_equal,
        "normal": normal_equal,
        "inside_to_outside": (
            chroma["inside_to_outside"].astype(np.bool_)
            == triton["inside_to_outside"].astype(np.bool_)
        ),
        "history": chroma["history"] == triton["history"],
        "surface_name": chroma["surface_name"] == triton["surface_name"],
    }
    for name in (
        "refractive_index1", "refractive_index2",
        "absorption_length", "scattering_length",
    ):
        exact_fields[name] = (
            np.ascontiguousarray(chroma[name], dtype=np.float32).view(np.uint32)
            == np.ascontiguousarray(triton[name], dtype=np.float32).view(np.uint32)
        )
    mismatch = ~np.logical_and.reduce(tuple(exact_fields.values()))
    cuda_pmt_hash = str(chroma["pmt_triangle_sha256"].item())
    triton_pmt_hash = str(triton["pmt_triangle_sha256"].item())
    pmt_words_exact = cuda_pmt_hash == triton_pmt_hash
    print("rays", len(kind_equal))
    print("kind mismatches", int(np.count_nonzero(~kind_equal)))
    print(
        "kind/channel mismatches",
        int(np.count_nonzero(~(kind_equal & channel_equal))),
    )
    print("any required-field mismatches", int(np.count_nonzero(mismatch)))
    print("flattened PMT triangle words exact", pmt_words_exact)
    print("flattened PMT triangle sha256", triton_pmt_hash)
    print("finite-distance max abs mm", float(np.max(difference[finite], initial=0.0)))
    print("finite-distance p99 abs mm", float(np.quantile(difference[finite], 0.99)) if np.any(finite) else 0.0)
    print("bitwise-normal rows", int(np.count_nonzero(normal_equal)), len(normal_equal))
    field_exact_counts = {}
    for name, equal in exact_fields.items():
        exact_count = int(np.count_nonzero(equal))
        field_exact_counts[name] = exact_count
        print(name + " exact", exact_count, len(equal))
    kind_counts = {}
    for code, label in ((MISS, "miss"), (PMT, "pmt"), (ACTIVE, "active"),
                        (CATHODE, "cathode"), (CAVITY, "cavity"), (WIRE, "wire")):
        selected = chroma["kind"] == code
        chroma_count = int(np.count_nonzero(selected))
        triton_count = int(np.count_nonzero(triton["kind"] == code))
        normal_exact_count = int(np.count_nonzero(selected & normal_equal))
        kind_counts[label] = {
            "chroma": chroma_count,
            "triton": triton_count,
            "normal_exact": normal_exact_count,
        }
        print(
            label,
            chroma_count,
            triton_count,
            "normal-exact",
            normal_exact_count,
        )
    bad = np.flatnonzero(mismatch)
    if bad.size:
        print("first mismatch indices", bad[:20].tolist())
    summary = {
        "schema_version": 1,
        "scope": "first-boundary full-detector Chroma CUDA/Triton parity",
        "rays": int(len(kind_equal)),
        "matched_bitwise": bool(not bad.size and pmt_words_exact),
        "required_field_exact_counts": field_exact_counts,
        "required_field_total": int(len(kind_equal)),
        "kind_counts": kind_counts,
        "flattened_pmt_triangle_words_exact": bool(pmt_words_exact),
        "flattened_pmt_triangle_sha256": triton_pmt_hash,
        "finite_distance_maximum_absolute_difference_mm": float(
            np.max(difference[finite], initial=0.0)
        ),
        "first_mismatch_indices": bad[:20].tolist(),
        "input_sha256": {
            "chroma_output": _file_sha256(chroma_path),
            "triton_output": _file_sha256(triton_path),
        },
    }
    if input_path is not None:
        summary["input_sha256"]["rays"] = _file_sha256(input_path)
    if artifact_path is not None:
        from chroma_lar.triton_scene import (
            TARGET_OPTICAL_SEMANTICS_SHA256,
            TARGET_TRAVERSAL_SHA256,
            load_chroma_global_bvh_artifact,
        )
        from chroma_lar.triton_scene.chroma_global_bvh import TARGET_MESH_MD5

        artifact = load_chroma_global_bvh_artifact(
            artifact_path,
            expected_mesh_md5=TARGET_MESH_MD5,
            expected_traversal_sha256=TARGET_TRAVERSAL_SHA256,
            expected_optical_semantics_sha256=TARGET_OPTICAL_SEMANTICS_SHA256,
        )
        summary["global_bvh_certificate"] = {
            "mesh_md5": artifact.mesh_md5,
            "traversal_sha256": artifact.traversal_sha256,
            "optical_semantics_sha256": artifact.optical_semantics_sha256,
            "artifact_sha256": artifact.sha256,
        }
        summary["input_sha256"]["global_bvh_artifact"] = _file_sha256(
            artifact_path
        )
    if json_path is not None:
        destination = Path(json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 1 if bad.size or not pmt_words_exact else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("chroma", "triton", "compare"), required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--chroma")
    parser.add_argument("--triton")
    parser.add_argument("--json")
    parser.add_argument(
        "--artifact",
        help=(
            "exact Chroma global-BVH NPZ; --backend chroma writes it and "
            "--backend triton requires it"
        ),
    )
    args = parser.parse_args()
    if args.backend == "chroma":
        _run_chroma(args.input, args.output, args.artifact)
        return 0
    if args.backend == "triton":
        if not args.artifact:
            parser.error("--backend triton requires --artifact")
        _run_triton(args.input, args.output, args.artifact)
        return 0
    return _compare(
        args.chroma,
        args.triton,
        json_path=args.json,
        artifact_path=args.artifact,
        input_path=args.input,
    )


if __name__ == "__main__":
    raise SystemExit(main())
