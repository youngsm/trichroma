#!/usr/bin/env python3
"""Run bounded reflect3wires transport in CUDA and Triton lockstep.

The coordinator launches the backends in separate processes, because the
reference Chroma context and Torch's primary context must not coexist.  CUDA
runs first and records the exact direction/polarization words produced by the
normalization at the head of ``propagate_tape``.  Triton starts from those
words.  Both implementations then consume the same photon-indexed random tape
and execute the requested number of physical interactions (bulk or boundary).

This is a bitwise diagnostic, not an ensemble test.  A one-step run reports
the first different photon, field, raw word, process, and last consumed tape
slot.  A multi-step run compares the complete resident state and tape cursors
after the bounded replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
for source_root in (REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

DEFAULT_CHROMA_CONTAINER = (
    REPOSITORY.parent / "chroma-lar" / "installation"
    / "chroma3.lar-plib" / "chroma.simg"
)


PROOF_SOURCE_FILES = (
    "chroma-lar/benchmarks/validate_detector_lockstep.py",
    "chroma-lar/chroma_lar/triton_backend.py",
    "chroma-lar/chroma_lar/triton_scene/chroma_global_bvh.py",
    "chroma-lar/chroma_lar/triton_scene/chroma_global_traversal.py",
    "chroma-lar/chroma_lar/triton_scene/intersect.py",
    "chroma-lite/chroma/cuda/propagate_tape.cu",
    "chroma-lite/chroma/cuda/rng_alignment.h",
    "chroma-lite/chroma/gpu/tools.py",
    "chroma-lite/chroma/triton/physics_kernels.py",
    "chroma-lite/chroma/triton/rng_alignment.py",
    "chroma-lite/chroma/triton/transport.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _proof_provenance(
    args: argparse.Namespace,
    cuda_values: Any,
    triton_values: Any,
) -> dict[str, Any]:
    """Pin the code and runtime identities needed to interpret a certificate."""

    def scalar_text(values: Any, key: str) -> str:
        return str(np.asarray(values[key]).reshape(()).item())

    source_hashes = {}
    for relative in PROOF_SOURCE_FILES:
        source = REPOSITORY / relative
        if not source.is_file():
            raise RuntimeError(f"proof source is missing: {source}")
        source_hashes[relative] = _sha256_file(source)
    image = Path(args.chroma_container).expanduser().resolve()
    image_stat = image.stat()
    return {
        "coordinator": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "chroma_cuda_worker": {
            "python": scalar_text(cuda_values, "runtime_python"),
            "numpy": scalar_text(cuda_values, "runtime_numpy"),
            "pycuda": scalar_text(cuda_values, "runtime_pycuda"),
            "cuda_driver": scalar_text(cuda_values, "runtime_cuda_driver"),
            "cuda_runtime": scalar_text(cuda_values, "runtime_cuda_runtime"),
            "gpu": scalar_text(cuda_values, "runtime_gpu"),
            "tape_compile_backend": scalar_text(
                cuda_values, "runtime_cuda_tape_compile_backend"
            ),
            "tape_compile_options": json.loads(scalar_text(
                cuda_values, "runtime_cuda_tape_compile_options"
            )),
            "force_scatter_at_pass": int(np.asarray(
                cuda_values["runtime_cuda_force_scatter_at_pass"]
            ).reshape(())),
            "container": str(image),
            "container_size_bytes": int(image_stat.st_size),
            "container_mtime_ns": int(image_stat.st_mtime_ns),
        },
        "triton_worker": {
            "python": scalar_text(triton_values, "runtime_python"),
            "numpy": scalar_text(triton_values, "runtime_numpy"),
            "torch": scalar_text(triton_values, "runtime_torch"),
            "torch_cuda": scalar_text(triton_values, "runtime_torch_cuda"),
            "triton": scalar_text(triton_values, "runtime_triton"),
            "gpu": scalar_text(triton_values, "runtime_gpu"),
        },
        "source_sha256": source_hashes,
    }


def _source_arrays(
    count: int,
    center: tuple[float, float, float],
    voxel_size: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate exactly the NumPy source law used by the acceptance run."""

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    direction = np.empty((count, 3), dtype=np.float32)
    cosine = (2.0 * rng.random_sample(count) - 1.0).astype(np.float32)
    azimuth = (2.0 * np.pi * rng.random_sample(count)).astype(np.float32)
    sine = np.sqrt(np.maximum(np.float32(0.0), 1.0 - cosine * cosine))
    direction[:, 0] = sine * np.cos(azimuth)
    direction[:, 1] = sine * np.sin(azimuth)
    direction[:, 2] = cosine

    helper = np.empty_like(direction)
    cosine = (2.0 * rng.random_sample(count) - 1.0).astype(np.float32)
    azimuth = (2.0 * np.pi * rng.random_sample(count)).astype(np.float32)
    sine = np.sqrt(np.maximum(np.float32(0.0), 1.0 - cosine * cosine))
    helper[:, 0] = sine * np.cos(azimuth)
    helper[:, 1] = sine * np.sin(azimuth)
    helper[:, 2] = cosine
    polarization = np.cross(direction, helper)
    polarization /= np.maximum(
        np.linalg.norm(polarization, axis=1, keepdims=True),
        np.float32(1.0e-20),
    )

    position = np.empty_like(direction)
    center_array = np.asarray(center, dtype=np.float32)
    for axis in range(3):
        position[:, axis] = (
            rng.random_sample(count).astype(np.float32)
            * np.float32(voxel_size)
            - np.float32(0.5 * voxel_size)
            + center_array[axis]
        )
    return {
        "position": np.ascontiguousarray(position),
        "direction": np.ascontiguousarray(direction),
        "polarization": np.ascontiguousarray(polarization),
    }


def _global_ids(args: argparse.Namespace) -> np.ndarray:
    """Return tape IDs for either a full wavefront or a replay subset."""

    if args.photon_ids:
        return np.ascontiguousarray(args.photon_ids, dtype=np.int64)
    return np.arange(args.count, dtype=np.int64)


def _photon_ids_from_file(path: Path) -> list[int]:
    """Read a JSON integer list or one integer per non-empty text line."""

    text = path.read_text()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        values = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                values.append(int(line, 10))
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: expected one integer photon ID"
                ) from error
        return values
    if not isinstance(parsed, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in parsed
    ):
        raise ValueError(f"{path}: JSON input must be a list of integers")
    return [int(value) for value in parsed]


def _photon_ids_from_npz(path: Path) -> list[int]:
    """Load the stable global ``active_ids`` tail from a worker NPZ."""

    with np.load(path, allow_pickle=False) as values:
        if "active_ids" not in values:
            raise ValueError(f"{path}: NPZ has no active_ids array")
        active = np.asarray(values["active_ids"])
    if active.ndim != 1 or active.dtype.kind not in "iu":
        raise ValueError(f"{path}: active_ids must be a one-dimensional integer array")
    return np.ascontiguousarray(active, dtype=np.int64).tolist()


def _source_arrays_for_args(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Generate the original population before selecting replay photons.

    The legacy NumPy source law draws count-sized coordinate blocks, so merely
    reducing ``--count`` changes every later photon.  ``--photon-id`` uses this
    helper to retain the source words from the declared source population.
    """

    population = int(args.source_population or args.count)
    complete = _source_arrays(
        population, tuple(args.center), args.voxel_size, args.source_seed
    )
    if not args.photon_ids:
        return complete
    selected = _global_ids(args)
    return {
        name: np.ascontiguousarray(values[selected])
        for name, values in complete.items()
    }


def _cuda_worker(
    args: argparse.Namespace, output: Path, global_bvh_output: Path
) -> None:
    os.environ.setdefault("PYCUDA_CACHE_DIR", "/tmp/chroma-pycuda-lockstep")
    Path(os.environ["PYCUDA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    from chroma import gpu
    from chroma.event import Photons
    from chroma.gpu.tools import to_float3
    from chroma.triton.rng_alignment import (
        RandomTape,
        RandomTapeSpec,
        allocate_pycuda_audit,
        allocate_pycuda_certificate,
        allocate_pycuda_state_certificate,
        allocate_pycuda_trace,
        get_legacy_tape_module,
        legacy_tape_compile_policy,
        launch_legacy_tape_step,
        to_pycuda,
    )
    from chroma_lar.geometry import build_detector_from_config
    import pycuda
    from pycuda import driver as cuda
    from pycuda import gpuarray as ga

    if not hasattr(np.linalg, "linalg"):
        np.linalg.linalg = np.linalg  # type: ignore[attr-defined]
    detector = build_detector_from_config(
        "detector_config_reflect_reflect3wires",
        flatten=True,
        include_wires=True,
        include_active=True,
        include_cathode=True,
        include_cavity=True,
    )
    # Export from the exact detector object handed to GPUDetector.  The
    # separate Triton worker can then replay identical flattened words without
    # ever importing PyCUDA or sharing its CUDA context.
    from chroma_lar.triton_scene.chroma_global_bvh import (
        build_chroma_global_bvh_artifact,
        save_chroma_global_bvh_artifact,
    )

    save_chroma_global_bvh_artifact(
        build_chroma_global_bvh_artifact(detector), global_bvh_output
    )
    context = gpu.create_cuda_context()
    try:
        gpu_geometry = gpu.GPUDetector(detector)
        source = _source_arrays_for_args(args)
        host_photons = Photons(
            pos=source["position"],
            dir=source["direction"],
            pol=source["polarization"],
            wavelengths=np.full(args.count, 450.0, dtype=np.float32),
        )
        photons = gpu.GPUPhotons(
            host_photons,
            copy_flags=True,
            copy_triangles=False,
            copy_weights=False,
        )

        # Capture exactly the normalization performed inside propagate_tape.
        normalized_direction_gpu = ga.empty_like(photons.dir)
        normalized_polarization_gpu = ga.empty_like(photons.pol)
        module = get_legacy_tape_module()
        compile_policy = legacy_tape_compile_policy()
        block = 256
        module.get_function("propagate_tape_normalize_inputs")(
            np.int32(args.count),
            photons.dir,
            photons.pol,
            normalized_direction_gpu,
            normalized_polarization_gpu,
            block=(block, 1, 1),
            grid=((args.count + block - 1) // block, 1, 1),
        )

        global_ids = _global_ids(args)
        tape = RandomTape.generate(
            global_ids,
            RandomTapeSpec(
                args.tape_interactions,
                args.draws_per_interaction,
                seed=args.tape_seed,
            ),
        )
        device_tape = to_pycuda(tape)
        audit = allocate_pycuda_audit(args.count)
        certificate = allocate_pycuda_certificate(
            args.count, args.tape_interactions
        )
        state_certificate = allocate_pycuda_state_certificate(
            args.count, args.tape_interactions
        )
        trace = allocate_pycuda_trace(args.count)
        input_queue = ga.to_gpu(np.arange(args.count, dtype=np.uint32))
        host_output_queue = np.zeros(args.count + 1, dtype=np.uint32)
        host_output_queue[0] = 1
        output_queue = ga.to_gpu(host_output_queue)
        tape_rows = ga.to_gpu(np.arange(args.count, dtype=np.int32))
        global_ids_gpu = ga.to_gpu(global_ids)
        launch_legacy_tape_step(
            gpu_photons=photons,
            gpu_geometry=gpu_geometry,
            input_queue=input_queue,
            output_queue=output_queue,
            photon_tape_rows=tape_rows,
            photon_global_ids=global_ids_gpu,
            tape=device_tape,
            audit=audit,
            trace=trace,
            certificate=certificate,
            state_certificate=state_certificate,
            max_steps=args.max_steps,
        )
        cuda.Context.synchronize()

        final = photons.get()
        last_triangle = np.ascontiguousarray(
            final.last_hit_triangles, dtype=np.int32
        )
        detected_channel = np.full(args.count, -1, dtype=np.int32)
        detected = ((final.flags & np.uint32(1 << 2)) != 0) & (
            last_triangle >= 0
        )
        if np.any(detected):
            solid = detector.solid_id[last_triangle[detected]].astype(np.int64)
            detected_channel[detected] = np.asarray(
                detector.solid_id_to_channel_index, dtype=np.int32
            )[solid]
        output_host = output_queue.get()
        active_count = max(0, int(output_host[0]) - 1)
        boundary_kind = np.zeros(args.count, dtype=np.int32)
        boundary_kind[last_triangle == -2] = 2  # analytic wire
        mesh_hit = last_triangle >= 0
        hit_solid = np.full(args.count, -1, dtype=np.int32)
        hit_solid[mesh_hit] = detector.solid_id[
            last_triangle[mesh_hit]
        ].astype(np.int32)
        boundary_kind[(hit_solid >= 1) & (hit_solid <= 162)] = 1  # PMT
        boundary_kind[hit_solid == 163] = 3  # active enclosure
        boundary_kind[hit_solid == 164] = 4  # cathode
        boundary_kind[hit_solid == 0] = 5  # cavity
        np.savez(
            output,
            initial_position=source["position"],
            normalized_direction=normalized_direction_gpu.get().view(
                np.float32
            ).reshape(args.count, 3),
            normalized_polarization=normalized_polarization_gpu.get().view(
                np.float32
            ).reshape(args.count, 3),
            position=np.ascontiguousarray(final.pos, dtype=np.float32),
            direction=np.ascontiguousarray(final.dir, dtype=np.float32),
            polarization=np.ascontiguousarray(final.pol, dtype=np.float32),
            time=np.ascontiguousarray(final.t, dtype=np.float32),
            history=np.ascontiguousarray(final.flags, dtype=np.uint32),
            last_triangle=last_triangle,
            boundary_kind=boundary_kind,
            detected_channel=detected_channel,
            process=trace.process.get().astype(np.int32, copy=False),
            interaction=trace.interaction.get().astype(np.int32, copy=False),
            draw_count=trace.draw_count.get().astype(np.int32, copy=False),
            trace_stage=trace.stage.get().astype(np.int32, copy=False),
            interaction_cursor=audit.interaction_cursor.get().astype(
                np.int32, copy=False
            ),
            draw_cursor=audit.draw_cursor.get().astype(np.int32, copy=False),
            overflow=audit.overflow.get().astype(np.uint32, copy=False),
            interaction_certificate=certificate.words.get().astype(
                np.uint32, copy=False
            ),
            state_certificate=state_certificate.words.get().astype(
                np.uint32, copy=False
            ),
            evidx=np.ascontiguousarray(final.evidx, dtype=np.uint32),
            global_ids=global_ids,
            active_ids=global_ids[
                output_host[1 : 1 + active_count].astype(np.int64, copy=False)
            ],
            runtime_python=np.asarray(platform.python_version()),
            runtime_numpy=np.asarray(np.__version__),
            runtime_pycuda=np.asarray(
                str(getattr(pycuda, "VERSION_TEXT", pycuda.VERSION))
            ),
            runtime_cuda_driver=np.asarray(str(cuda.get_driver_version())),
            runtime_cuda_runtime=np.asarray(
                ".".join(str(value) for value in cuda.get_version())
            ),
            runtime_cuda_tape_compile_backend=np.asarray(
                compile_policy["backend"]
            ),
            runtime_cuda_tape_compile_options=np.asarray(
                json.dumps(compile_policy["options"], separators=(",", ":"))
            ),
            runtime_cuda_force_scatter_at_pass=np.asarray(
                compile_policy["force_scatter_at_pass"], dtype=np.int32
            ),
            runtime_gpu=np.asarray(cuda.Context.get_device().name()),
        )
    finally:
        context.pop()


def _process_from_one_step_history(history: np.ndarray) -> np.ndarray:
    process = np.zeros(len(history), dtype=np.int32)
    # The one-interaction run starts with zero history, so these are not
    # cumulative-history ambiguities.  Terminal outcomes take precedence.
    rules = (
        (1 << 15, 12),
        (1 << 0, 11),
        (1 << 1, 1),
        (1 << 2, 4),
        (1 << 3, 3),
        (1 << 5, 5),
        (1 << 6, 6),
        (1 << 4, 2),
    )
    unresolved = np.ones(len(history), dtype=np.bool_)
    for bit, code in rules:
        selected = unresolved & ((history & np.uint32(bit)) != 0)
        process[selected] = code
        unresolved[selected] = False
    return process


def _triton_worker(
    args: argparse.Namespace,
    cuda_input: Path,
    global_bvh_input: Path,
    output: Path,
) -> None:
    import torch
    import triton

    from chroma.triton.rng_alignment import (
        RandomTape,
        RandomTapeSpec,
        allocate_torch_audit,
        allocate_torch_certificate,
        allocate_torch_state_certificate,
        to_torch,
    )
    from chroma_lar.triton_backend import (
        BoundaryTapeTrace,
        Reflect3WiresTritonSimulation,
    )

    cuda_values = np.load(cuda_input)
    device = torch.device("cuda")
    simulation = Reflect3WiresTritonSimulation(
        tile_size=args.count,
        history_length=min(8, args.max_steps),
        legacy_specular_reflection=True,
        chroma_mesh_box_compatibility=True,
        chroma_global_bvh_artifact=global_bvh_input,
    )
    positions = torch.as_tensor(
        np.ascontiguousarray(cuda_values["initial_position"]), device=device
    ).clone()
    directions = torch.as_tensor(
        np.ascontiguousarray(cuda_values["normalized_direction"]), device=device
    ).clone()
    polarizations = torch.as_tensor(
        np.ascontiguousarray(cuda_values["normalized_polarization"]),
        device=device,
    ).clone()
    state = (
        positions,
        directions,
        polarizations,
        torch.zeros(args.count, dtype=torch.float32, device=device),
        torch.zeros(args.count, dtype=torch.int32, device=device),
        torch.zeros(args.count, dtype=torch.int64, device=device),
        torch.full((args.count,), -1, dtype=torch.int32, device=device),
        torch.full((args.count,), -1, dtype=torch.int32, device=device),
        torch.full((args.count,), -1, dtype=torch.int32, device=device),
        torch.zeros(args.count, dtype=torch.int32, device=device),
    )
    global_ids = _global_ids(args)
    global_ids_gpu = torch.as_tensor(global_ids, device=device)
    host_tape = RandomTape.generate(
        global_ids,
        RandomTapeSpec(
            args.tape_interactions,
            args.draws_per_interaction,
            seed=args.tape_seed,
        ),
    )
    tape = to_torch(host_tape, device=device)
    audit = allocate_torch_audit(args.count, device=device)
    certificate = allocate_torch_certificate(
        args.count, args.tape_interactions, device=device
    )
    state_certificate = allocate_torch_state_certificate(
        args.count, args.tape_interactions, device=device
    )
    boundary_trace = BoundaryTapeTrace.allocate(args.count, device=device)
    pending = torch.arange(args.count, dtype=torch.int32, device=device)
    tape_rows = pending.clone()
    simulation.chroma_global_workspace.clear_sticky_overflow()
    simulation._propagate_state(
        state,
        pending,
        args.tape_seed,
        0,
        args.max_steps,
        args.max_steps,
        random_tape=tape,
        tape_audit=audit,
        tape_row_indices=tape_rows,
        tape_trace=boundary_trace,
        tape_certificate=certificate,
        state_certificate=state_certificate,
        global_photon_ids=global_ids_gpu,
    )
    torch.cuda.synchronize()
    if simulation.chroma_global_workspace.sticky_overflowed():
        raise RuntimeError(
            "Chroma global BVH traversal stack overflowed; lockstep output "
            "is invalid"
        )

    history = state[4].cpu().numpy().astype(np.uint32, copy=False)
    boundary_interaction = boundary_trace.interaction.cpu().numpy()
    process = _process_from_one_step_history(history)
    boundary_process = boundary_trace.decision.cpu().numpy()
    is_boundary = boundary_interaction >= 0
    process[is_boundary] = boundary_process[is_boundary]
    draw_count = np.where(
        process == 2,
        4,
        np.where(process == 1, 2, 0),
    ).astype(np.int32)
    boundary_draw_count = boundary_trace.draw_count.cpu().numpy()
    draw_count[is_boundary] = boundary_draw_count[is_boundary]
    interaction = np.zeros(args.count, dtype=np.int32)
    interaction[is_boundary] = boundary_interaction[is_boundary]
    terminal_mask = np.uint32(
        (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 15)
    )
    active_slots = np.flatnonzero((history & terminal_mask) == 0)
    active_ids = np.ascontiguousarray(global_ids[active_slots], dtype=np.int64)
    last_instance = state[6].cpu().numpy()
    last_triangle = state[7].cpu().numpy()
    boundary_kind = np.zeros(args.count, dtype=np.int32)
    boundary_kind[last_triangle == -2] = 2  # analytic wire
    mesh_hit = last_triangle >= 0
    hit_solid = np.full(args.count, -1, dtype=np.int32)
    hit_solid[mesh_hit] = simulation.chroma_global_host_artifact.solid_id[
        last_triangle[mesh_hit]
    ]
    boundary_kind[(hit_solid >= 1) & (hit_solid <= 162)] = 1  # PMT
    boundary_kind[hit_solid == 163] = 3  # active enclosure
    boundary_kind[hit_solid == 164] = 4  # cathode
    boundary_kind[hit_solid == 0] = 5  # cavity
    np.savez(
        output,
        position=state[0].cpu().numpy(),
        direction=state[1].cpu().numpy(),
        polarization=state[2].cpu().numpy(),
        time=state[3].cpu().numpy(),
        history=history,
        last_instance=last_instance,
        last_triangle=last_triangle,
        boundary_kind=boundary_kind,
        detected_channel=state[8].cpu().numpy(),
        process=process,
        interaction=interaction,
        draw_count=draw_count,
        interaction_cursor=audit.interaction_cursor.cpu().numpy(),
        draw_cursor=audit.draw_cursor.cpu().numpy(),
        overflow=audit.overflow.cpu().numpy().astype(np.uint32, copy=False),
        interaction_certificate=np.ascontiguousarray(
            certificate.words.cpu().numpy().view(np.uint32)
        ),
        state_certificate=np.ascontiguousarray(
            state_certificate.words.cpu().numpy().view(np.uint32)
        ),
        evidx=np.zeros(args.count, dtype=np.uint32),
        global_ids=global_ids,
        active_ids=active_ids,
        runtime_python=np.asarray(platform.python_version()),
        runtime_numpy=np.asarray(np.__version__),
        runtime_torch=np.asarray(str(torch.__version__)),
        runtime_torch_cuda=np.asarray(str(torch.version.cuda)),
        runtime_triton=np.asarray(str(triton.__version__)),
        runtime_gpu=np.asarray(torch.cuda.get_device_name(device)),
    )


def _worker_command(
    args: argparse.Namespace,
    backend: str,
    output: Path,
    cuda_input: Path | None = None,
    global_bvh: Path | None = None,
) -> list[str]:
    worker = [
        str(Path(__file__).resolve()),
        "--count", str(args.count),
        "--center", *(str(value) for value in args.center),
        "--voxel-size", str(args.voxel_size),
        "--source-seed", str(args.source_seed),
        "--tape-seed", str(args.tape_seed),
        "--tape-interactions", str(args.tape_interactions),
        "--draws-per-interaction", str(args.draws_per_interaction),
        "--max-steps", str(args.max_steps),
        "--_worker", backend,
        "--_output", str(output),
    ]
    if args.photon_ids:
        worker.extend(("--source-population", str(args.source_population)))
        for photon_id in args.photon_ids:
            worker.extend(("--photon-id", str(photon_id)))
    if cuda_input is not None:
        worker.extend(("--_cuda-input", str(cuda_input)))
    if global_bvh is not None:
        option = (
            "--_artifact-output" if backend == "cuda" else "--_artifact-input"
        )
        worker.extend((option, str(global_bvh)))
    if backend == "cuda":
        image = Path(args.chroma_container).expanduser().resolve()
        python_path = "%s:%s" % (
            REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"
        )
        return [
            "singularity", "exec", "--nv",
            "-B", "/sdf:/sdf", "-B", "/tmp:/tmp",
            "--pwd", str(REPOSITORY), str(image),
            "env", "PYTHONPATH=" + python_path,
            "PYTHONNOUSERSITE=1",
            "PYCUDA_CACHE_DIR=/tmp/chroma-pycuda-lockstep",
            "TMPDIR=/tmp", "OMP_NUM_THREADS=1",
            "python", *worker,
        ]
    return [sys.executable, *worker]


def _process_counts(values: np.ndarray) -> dict[str, int]:
    names = {
        0: "unset", 1: "bulk_absorb", 2: "bulk_scatter",
        3: "surface_absorb", 4: "surface_detect",
        5: "surface_diffuse", 6: "surface_specular",
        7: "dielectric_reflect", 8: "dielectric_transmit",
        11: "no_hit", 12: "invalid",
    }
    unique, counts = np.unique(values, return_counts=True)
    return {
        names.get(int(code), str(int(code))): int(count)
        for code, count in zip(unique, counts)
    }


def _process_name(code: int) -> str:
    return {
        0: "unset", 1: "bulk_absorb", 2: "bulk_scatter",
        3: "surface_absorb", 4: "surface_detect",
        5: "surface_diffuse", 6: "surface_specular",
        7: "dielectric_reflect", 8: "dielectric_transmit",
        11: "no_hit", 12: "invalid",
    }.get(int(code), str(int(code)))


def _certificate_comparison(
    cuda_values: Any,
    triton_values: Any,
) -> dict[str, Any]:
    """Compare every dense ledger word and independently certify draw use."""

    from chroma.triton.rng_alignment import (
        validate_interaction_certificate,
        unpack_interaction_certificate,
    )

    cuda_words = np.ascontiguousarray(
        cuda_values["interaction_certificate"], dtype=np.uint32
    )
    triton_words = np.ascontiguousarray(
        triton_values["interaction_certificate"], dtype=np.uint32
    )
    cuda_cursor = np.ascontiguousarray(
        cuda_values["interaction_cursor"], dtype=np.int32
    )
    triton_cursor = np.ascontiguousarray(
        triton_values["interaction_cursor"], dtype=np.int32
    )
    cuda_draw_cursor = np.ascontiguousarray(
        cuda_values["draw_cursor"], dtype=np.int32
    )
    triton_draw_cursor = np.ascontiguousarray(
        triton_values["draw_cursor"], dtype=np.int32
    )

    prefix_valid: dict[str, bool] = {}
    prefix_errors: dict[str, str] = {}
    for label, words, cursor in (
        ("chroma", cuda_words, cuda_cursor),
        ("triton", triton_words, triton_cursor),
    ):
        try:
            validate_interaction_certificate(words, cursor)
        except (TypeError, ValueError) as error:
            prefix_valid[label] = False
            prefix_errors[label] = str(error)
        else:
            prefix_valid[label] = True

    shapes_equal = cuda_words.shape == triton_words.shape
    if shapes_equal:
        word_equal = cuda_words == triton_words
        ledger_equal = bool(np.all(word_equal))
        exact_words = int(np.count_nonzero(word_equal))
        total_words = int(cuda_words.size)
    else:
        word_equal = np.empty((0,), dtype=np.bool_)
        ledger_equal = False
        exact_words = 0
        total_words = int(max(cuda_words.size, triton_words.size))

    interaction_cursor_equal = np.array_equal(cuda_cursor, triton_cursor)
    draw_cursor_equal = np.array_equal(cuda_draw_cursor, triton_draw_cursor)
    cursor_equal = interaction_cursor_equal and draw_cursor_equal
    overflow_free = not (
        np.any(np.asarray(cuda_values["overflow"], dtype=np.uint32))
        or np.any(np.asarray(triton_values["overflow"], dtype=np.uint32))
    )
    draw_consumption_certified = bool(
        ledger_equal
        and prefix_valid.get("chroma", False)
        and prefix_valid.get("triton", False)
        and cursor_equal
        and overflow_free
    )

    first_difference = None
    if shapes_equal and not ledger_equal:
        row, interaction = np.argwhere(~word_equal)[0]
        row = int(row)
        interaction = int(interaction)
        cuda_word = cuda_words[row, interaction]
        triton_word = triton_words[row, interaction]
        cuda_committed, cuda_process, cuda_draw = (
            value[0]
            for value in unpack_interaction_certificate(
                np.asarray([cuda_word], dtype=np.uint32)
            )
        )
        triton_committed, triton_process, triton_draw = (
            value[0]
            for value in unpack_interaction_certificate(
                np.asarray([triton_word], dtype=np.uint32)
            )
        )
        global_ids = np.asarray(cuda_values["global_ids"], dtype=np.int64)
        global_photon_id = (
            int(global_ids[row]) if row < len(global_ids) else None
        )
        first_difference = {
            "row": row,
            "global_photon_id": global_photon_id,
            "interaction": interaction,
            "chroma_word": int(cuda_word),
            "triton_word": int(triton_word),
            "chroma_committed": bool(cuda_committed),
            "triton_committed": bool(triton_committed),
            "chroma_process": (
                None if not cuda_committed else int(cuda_process)
            ),
            "triton_process": (
                None if not triton_committed else int(triton_process)
            ),
            "chroma_process_name": (
                None if not cuda_committed else _process_name(cuda_process)
            ),
            "triton_process_name": (
                None if not triton_committed else _process_name(triton_process)
            ),
            "chroma_draw_count": (
                None if not cuda_committed else int(cuda_draw)
            ),
            "triton_draw_count": (
                None if not triton_committed else int(triton_draw)
            ),
        }
    elif not shapes_equal:
        first_difference = {
            "field": "shape",
            "chroma_shape": list(cuda_words.shape),
            "triton_shape": list(triton_words.shape),
        }

    return {
        "shape": list(cuda_words.shape) if shapes_equal else None,
        "shapes_equal": bool(shapes_equal),
        "exact_words": exact_words,
        "total_words": total_words,
        "exact_fraction": (
            float(exact_words / total_words) if total_words else 1.0
        ),
        "ledger_equal": bool(ledger_equal),
        "prefix_valid": prefix_valid,
        "prefix_errors": prefix_errors,
        "interaction_cursor_equal": bool(interaction_cursor_equal),
        "draw_cursor_equal": bool(draw_cursor_equal),
        "cursor_equal": bool(cursor_equal),
        "overflow_free": bool(overflow_free),
        "draw_consumption_certified": draw_consumption_certified,
        "first_difference": first_difference,
    }


STATE_CERTIFICATE_FIELDS = (
    "position_x",
    "position_y",
    "position_z",
    "direction_x",
    "direction_y",
    "direction_z",
    "polarization_x",
    "polarization_y",
    "polarization_z",
    "wavelength",
    "time",
    "history",
    "last_triangle",
    "weight",
    "evidx",
)
STATE_CERTIFICATE_EMPTY_WORD = np.uint32(0xFFFFFFFF)


def _state_word_value(field: int, word: np.uint32) -> Any:
    """Decode a raw word only for readable diagnostics; equality stays raw."""

    raw = np.asarray([word], dtype=np.uint32)
    if field in (11, 14):
        return int(raw[0])
    if field == 12:
        return int(raw.view(np.int32)[0])
    return float(raw.view(np.float32)[0])


def _state_certificate_comparison(
    cuda_values: Any,
    triton_values: Any,
) -> dict[str, Any]:
    """Byte-compare every committed post-interaction Photon state.

    A cursor defines the committed prefix for each stable tape row.  The
    remainder must stay all-``0xffffffff``; this independently catches stale
    or speculative writes after the bounded replay.  Individual committed
    fields may legitimately equal that sentinel (notably ``last_triangle``),
    so prefix presence is checked per complete 15-word record.
    """

    from chroma.triton.rng_alignment import (
        STATE_CERTIFICATE_FIELDS as SHARED_STATE_CERTIFICATE_FIELDS,
        validate_state_certificate,
    )

    if tuple(SHARED_STATE_CERTIFICATE_FIELDS) != STATE_CERTIFICATE_FIELDS:
        raise RuntimeError("lockstep state-certificate field order drifted")

    cuda_words = np.ascontiguousarray(
        cuda_values["state_certificate"], dtype=np.uint32
    )
    triton_words = np.ascontiguousarray(
        triton_values["state_certificate"], dtype=np.uint32
    )
    cuda_cursor = np.ascontiguousarray(
        cuda_values["interaction_cursor"], dtype=np.int32
    )
    triton_cursor = np.ascontiguousarray(
        triton_values["interaction_cursor"], dtype=np.int32
    )

    expected_rank_shape = lambda words, cursor: (
        words.ndim == 3
        and words.shape[2] == len(STATE_CERTIFICATE_FIELDS)
        and cursor.shape == (words.shape[0],)
        and np.all(cursor >= 0)
        and np.all(cursor <= words.shape[1])
    )
    layout_valid = {
        "chroma": bool(expected_rank_shape(cuda_words, cuda_cursor)),
        "triton": bool(expected_rank_shape(triton_words, triton_cursor)),
    }
    layout_errors: dict[str, str] = {}
    if not layout_valid["chroma"]:
        layout_errors["chroma"] = (
            "expected uint32 [row, interaction, 15] with one in-range "
            "interaction cursor per row"
        )
    if not layout_valid["triton"]:
        layout_errors["triton"] = (
            "expected uint32 [row, interaction, 15] with one in-range "
            "interaction cursor per row"
        )

    suffix_valid: dict[str, bool] = {}
    prefix_records_present: dict[str, bool] = {}
    occupancy_valid: dict[str, bool] = {}
    occupancy_errors: dict[str, str] = {}
    suffix_errors: dict[str, dict[str, Any]] = {}
    prefix_errors: dict[str, dict[str, Any]] = {}
    for label, words, cursor, process_words in (
        (
            "chroma",
            cuda_words,
            cuda_cursor,
            cuda_values["interaction_certificate"],
        ),
        (
            "triton",
            triton_words,
            triton_cursor,
            triton_values["interaction_certificate"],
        ),
    ):
        if not layout_valid[label]:
            suffix_valid[label] = False
            prefix_records_present[label] = False
            occupancy_valid[label] = False
            continue
        try:
            validate_state_certificate(words, process_words, cursor)
        except (TypeError, ValueError) as error:
            occupancy_valid[label] = False
            occupancy_errors[label] = str(error)
        else:
            occupancy_valid[label] = True
        interaction = np.arange(words.shape[1])[None, :]
        committed = interaction < cursor[:, None]
        suffix_bad = (~committed)[..., None] & (
            words != STATE_CERTIFICATE_EMPTY_WORD
        )
        suffix_valid[label] = not bool(np.any(suffix_bad))
        if not suffix_valid[label]:
            row, step, field = np.argwhere(suffix_bad)[0]
            suffix_errors[label] = {
                "row": int(row),
                "interaction": int(step),
                "field_index": int(field),
                "field": STATE_CERTIFICATE_FIELDS[int(field)],
                "word": int(words[row, step, field]),
            }
        empty_record = np.all(
            words == STATE_CERTIFICATE_EMPTY_WORD, axis=2
        )
        missing = committed & empty_record
        prefix_records_present[label] = not bool(np.any(missing))
        if not prefix_records_present[label]:
            row, step = np.argwhere(missing)[0]
            prefix_errors[label] = {
                "row": int(row),
                "interaction": int(step),
            }

    shapes_equal = cuda_words.shape == triton_words.shape
    cursor_equal = np.array_equal(cuda_cursor, triton_cursor)
    comparable_layout = bool(
        shapes_equal and all(layout_valid.values())
    )
    committed_mask = None
    committed_equal = False
    exact_words = 0
    compared_words = 0
    first_difference = None
    if comparable_layout:
        interaction = np.arange(cuda_words.shape[1])[None, :]
        # Use the union so a cursor mismatch cannot hide one backend's extra
        # committed record behind the other's sentinel suffix.
        committed_union = (
            (interaction < cuda_cursor[:, None])
            | (interaction < triton_cursor[:, None])
        )
        committed_mask = np.broadcast_to(
            committed_union[..., None], cuda_words.shape
        )
        equal = cuda_words == triton_words
        compared_words = int(np.count_nonzero(committed_mask))
        exact_words = int(np.count_nonzero(equal & committed_mask))
        committed_equal = bool(np.all(equal[committed_mask]))
        mismatch = committed_mask & ~equal
        if np.any(mismatch):
            row, step, field = np.argwhere(mismatch)[0]
            row, step, field = int(row), int(step), int(field)
            cuda_word = cuda_words[row, step, field]
            triton_word = triton_words[row, step, field]
            global_ids = np.asarray(
                cuda_values.get("global_ids", ()), dtype=np.int64
            )
            first_difference = {
                "row": row,
                "global_photon_id": (
                    int(global_ids[row]) if row < len(global_ids) else None
                ),
                "interaction": step,
                "field_index": field,
                "field": STATE_CERTIFICATE_FIELDS[field],
                "chroma_word": int(cuda_word),
                "triton_word": int(triton_word),
                "chroma_word_hex": f"0x{int(cuda_word):08x}",
                "triton_word_hex": f"0x{int(triton_word):08x}",
                "chroma_value": _state_word_value(field, cuda_word),
                "triton_value": _state_word_value(field, triton_word),
            }
        elif not cursor_equal:
            row = int(np.flatnonzero(cuda_cursor != triton_cursor)[0])
            first_difference = {
                "field": "interaction_cursor",
                "row": row,
                "chroma_cursor": int(cuda_cursor[row]),
                "triton_cursor": int(triton_cursor[row]),
            }
    else:
        first_difference = {
            "field": "shape_or_layout",
            "chroma_shape": list(cuda_words.shape),
            "triton_shape": list(triton_words.shape),
            "layout_valid": layout_valid,
        }

    if first_difference is None:
        for label in ("chroma", "triton"):
            if label in suffix_errors:
                first_difference = {
                    "kind": "sentinel_suffix",
                    "backend": label,
                    **suffix_errors[label],
                }
                break
            if label in prefix_errors:
                first_difference = {
                    "kind": "missing_committed_record",
                    "backend": label,
                    **prefix_errors[label],
                }
                break
            if label in occupancy_errors:
                first_difference = {
                    "kind": "invalid_occupancy",
                    "backend": label,
                    "error": occupancy_errors[label],
                }
                break

    certified = bool(
        comparable_layout
        and cursor_equal
        and committed_equal
        and all(suffix_valid.values())
        and all(prefix_records_present.values())
        and all(occupancy_valid.values())
    )
    return {
        "field_order": list(STATE_CERTIFICATE_FIELDS),
        "shape": list(cuda_words.shape) if shapes_equal else None,
        "shapes_equal": bool(shapes_equal),
        "layout_valid": layout_valid,
        "layout_errors": layout_errors,
        "committed_records": {
            "chroma": (
                int(cuda_cursor.sum()) if layout_valid["chroma"] else None
            ),
            "triton": (
                int(triton_cursor.sum()) if layout_valid["triton"] else None
            ),
        },
        "compared_committed_words": compared_words,
        "exact_committed_words": exact_words,
        "exact_committed_fraction": (
            float(exact_words / compared_words) if compared_words else 1.0
        ),
        "committed_words_equal": bool(committed_equal),
        "cursor_equal": bool(cursor_equal),
        "suffix_valid": suffix_valid,
        "suffix_errors": suffix_errors,
        "prefix_records_present": prefix_records_present,
        "prefix_errors": prefix_errors,
        "occupancy_valid": occupancy_valid,
        "occupancy_errors": occupancy_errors,
        "post_interaction_state_certified": certified,
        "first_difference": first_difference,
    }


def _active_set_comparison(
    cuda_values: Any,
    triton_values: Any,
) -> dict[str, Any]:
    """Compare scheduler survivors in the stable global-photon namespace."""

    cuda_active = np.sort(
        np.ascontiguousarray(cuda_values["active_ids"], dtype=np.int64)
    )
    triton_active = np.sort(
        np.ascontiguousarray(triton_values["active_ids"], dtype=np.int64)
    )
    cuda_unique = len(np.unique(cuda_active)) == len(cuda_active)
    triton_unique = len(np.unique(triton_active)) == len(triton_active)
    exact = (
        cuda_unique
        and triton_unique
        and np.array_equal(cuda_active, triton_active)
    )
    return {
        "chroma": int(len(cuda_active)),
        "triton": int(len(triton_active)),
        "chroma_unique": bool(cuda_unique),
        "triton_unique": bool(triton_unique),
        "exact": bool(exact),
    }


def _last_instance_invariant(triton_values: Any) -> dict[str, Any]:
    """Global-BVH compatibility uses CUDA's global triangle namespace only."""

    values = np.ascontiguousarray(triton_values["last_instance"], dtype=np.int32)
    valid = values == -1
    return {
        "required_value": -1,
        "exact": int(np.count_nonzero(valid)),
        "total": int(values.size),
        "satisfied": bool(np.all(valid)),
    }


def _single_event_evidx_invariant(
    cuda_values: Any,
    triton_values: Any,
) -> dict[str, Any]:
    """Require the specialized one-event replay to preserve event index zero."""

    chroma = np.ascontiguousarray(cuda_values["evidx"], dtype=np.uint32)
    triton = np.ascontiguousarray(triton_values["evidx"], dtype=np.uint32)
    shape_valid = chroma.ndim == 1 and triton.shape == chroma.shape
    chroma_zero = shape_valid and bool(np.all(chroma == np.uint32(0)))
    triton_zero = shape_valid and bool(np.all(triton == np.uint32(0)))
    endpoint_equal = shape_valid and bool(np.array_equal(chroma, triton))
    return {
        "required_value": 0,
        "shape_valid": bool(shape_valid),
        "photons": int(chroma.size) if chroma.ndim == 1 else None,
        "chroma_zero": bool(chroma_zero),
        "triton_zero": bool(triton_zero),
        "endpoint_equal": bool(endpoint_equal),
        "satisfied": bool(chroma_zero and triton_zero and endpoint_equal),
    }


def _comparison(
    args: argparse.Namespace,
    cuda_values: Any,
    triton_values: Any,
) -> dict[str, Any]:
    from chroma.triton.lockstep import (
        capture_trace,
        chroma_draw_slot_stage,
        compare_traces,
    )
    from chroma.triton.rng_alignment import RandomTape, RandomTapeSpec, TapeAudit

    global_ids = _global_ids(args)
    tape = RandomTape.generate(
        global_ids,
        RandomTapeSpec(
            args.tape_interactions,
            args.draws_per_interaction,
            seed=args.tape_seed,
        ),
    )

    def capture(values: Any):
        process = np.ascontiguousarray(values["process"], dtype=np.int32)
        draw_count = np.ascontiguousarray(values["draw_count"], dtype=np.int32)
        audit = TapeAudit(
            np.ascontiguousarray(values["interaction_cursor"], dtype=np.int32),
            np.ascontiguousarray(values["draw_cursor"], dtype=np.int32),
            np.ascontiguousarray(values["overflow"], dtype=np.uint32),
        )

        def stage(record: int, slot: int, coarse: int) -> int:
            del coarse
            return int(chroma_draw_slot_stage(
                int(process[record]), int(draw_count[record]), slot
            ))

        return capture_trace(
            global_photon_ids=global_ids,
            positions=values["position"],
            directions=values["direction"],
            polarizations=values["polarization"],
            times=values["time"],
            histories=values["history"],
            process=process,
            audit=audit,
            interaction_indices=np.ascontiguousarray(
                values["interaction"], dtype=np.int32
            ),
            draw_counts=draw_count,
            step_index=0,
            tape=tape,
            slot_stage_resolver=stage,
            extras={
                "detected_channel": np.ascontiguousarray(
                    values["detected_channel"], dtype=np.int32
                )
            },
        )

    cuda_trace = capture(cuda_values)
    triton_trace = capture(triton_values)
    result = compare_traces(
        cuda_trace, triton_trace,
        left_label="Chroma CUDA", right_label="Triton",
    )
    fields: dict[str, Any] = {}
    for name in ("position", "direction", "polarization"):
        left = np.ascontiguousarray(cuda_values[name]).view(np.uint32)
        right = np.ascontiguousarray(triton_values[name]).view(np.uint32)
        exact_rows = np.all(left == right, axis=1)
        fields[name] = {
            "exact_rows": int(np.count_nonzero(exact_rows)),
            "exact_fraction": float(np.mean(exact_rows)),
            "maximum_absolute_difference": float(np.max(np.abs(
                np.asarray(cuda_values[name], dtype=np.float64)
                - np.asarray(triton_values[name], dtype=np.float64)
            ))),
        }
    for name in (
        "time", "history", "process", "draw_count",
        "interaction", "interaction_cursor", "draw_cursor", "overflow",
        "detected_channel", "boundary_kind", "last_triangle",
        "evidx",
        "global_ids",
    ):
        left = np.ascontiguousarray(cuda_values[name])
        right = np.ascontiguousarray(triton_values[name])
        if left.dtype.kind == "f":
            equal = left.view(np.uint32) == right.view(np.uint32)
        else:
            equal = left == right
        fields[name] = {
            "exact": int(np.count_nonzero(equal)),
            "exact_fraction": float(np.mean(equal)),
        }

    process_agreement: dict[str, Any] = {}
    left_process = np.asarray(cuda_values["process"], dtype=np.int32)
    for code in np.unique(left_process):
        selected = left_process == code
        count = int(np.count_nonzero(selected))
        summary: dict[str, Any] = {"count": count}
        for name in ("position", "direction", "polarization"):
            left = np.ascontiguousarray(cuda_values[name]).view(np.uint32)
            right = np.ascontiguousarray(triton_values[name]).view(np.uint32)
            row_equal = np.all(left == right, axis=1)
            summary[name + "_exact"] = int(np.count_nonzero(
                selected & row_equal
            ))
        left_time = np.ascontiguousarray(cuda_values["time"]).view(np.uint32)
        right_time = np.ascontiguousarray(triton_values["time"]).view(np.uint32)
        summary["time_exact"] = int(np.count_nonzero(
            selected & (left_time == right_time)
        ))
        process_agreement[_process_name(int(code))] = summary

    first = None
    if result.difference is not None:
        difference = result.difference
        first = {
            "global_photon_id": difference.global_photon_id,
            "step_index": difference.step_index,
            "interaction_index": difference.interaction_index,
            "field": difference.field,
            "left_value": difference.left_value,
            "right_value": difference.right_value,
            "left_word": difference.left_word,
            "right_word": difference.right_word,
            "draw_slot": difference.draw_slot,
            "left_draw_word": difference.left_draw_word,
            "right_draw_word": difference.right_draw_word,
            "formatted": difference.format(),
        }
    certificate = _certificate_comparison(cuda_values, triton_values)
    state_certificate = _state_certificate_comparison(
        cuda_values, triton_values
    )
    active_set = _active_set_comparison(cuda_values, triton_values)
    last_instance_invariant = _last_instance_invariant(triton_values)
    single_event_evidx_invariant = _single_event_evidx_invariant(
        cuda_values, triton_values
    )
    required_fields_exact = all(
        int(summary.get("exact_rows", summary.get("exact", -1)))
        == args.count
        for summary in fields.values()
    )
    matched = bool(
        result.matched
        and required_fields_exact
        and active_set["exact"]
        and last_instance_invariant["satisfied"]
        and single_event_evidx_invariant["satisfied"]
        and certificate["draw_consumption_certified"]
        and state_certificate["post_interaction_state_certified"]
    )
    if first is None and certificate["first_difference"] is not None:
        first = {
            "field": "interaction_certificate",
            **certificate["first_difference"],
        }
    if first is None and state_certificate["first_difference"] is not None:
        first = {
            "field": "state_certificate",
            **state_certificate["first_difference"],
        }
    return {
        "schema_version": 6,
        "detector": "detector_config_reflect_reflect3wires",
        "scope": "one complete physical interaction from identical input/tape words",
        "photons": args.count,
        "source_population": int(args.source_population or args.count),
        "global_photon_ids": global_ids.tolist(),
        "center": [float(value) for value in args.center],
        "voxel_size": float(args.voxel_size),
        "source_seed": args.source_seed,
        "tape_seed": args.tape_seed,
        "tape_interactions": int(args.tape_interactions),
        "draws_per_interaction": int(args.draws_per_interaction),
        "legacy_specular_reflection": True,
        "chroma_mesh_box_compatibility": True,
        "chroma_global_bvh_compatibility": True,
        "representable_progress_guard": False,
        "wire_scan_source_indices": [0, 1, 2, 3, 4, 5],
        "matched_bitwise": matched,
        "draw_consumption_certified": bool(
            certificate["draw_consumption_certified"]
        ),
        "compared_records_before_first_difference": int(
            result.compared_records
        ),
        "first_difference": first,
        "field_agreement": fields,
        "interaction_certificate": certificate,
        "state_certificate": state_certificate,
        "post_interaction_state_certified": bool(
            state_certificate["post_interaction_state_certified"]
        ),
        "last_instance_invariant": last_instance_invariant,
        "single_event_evidx_invariant": single_event_evidx_invariant,
        "agreement_by_process": process_agreement,
        "process_counts": {
            "chroma": _process_counts(cuda_values["process"]),
            "triton": _process_counts(triton_values["process"]),
        },
        "active_set": active_set,
    }


def _comparison_multistep(
    args: argparse.Namespace,
    cuda_values: Any,
    triton_values: Any,
) -> dict[str, Any]:
    """Compare complete resident state after several physical interactions."""

    fields: dict[str, Any] = {}
    matched = True
    for name in ("position", "direction", "polarization"):
        left = np.ascontiguousarray(cuda_values[name], dtype=np.float32)
        right = np.ascontiguousarray(triton_values[name], dtype=np.float32)
        equal = left.view(np.uint32) == right.view(np.uint32)
        row_equal = np.all(equal, axis=1)
        fields[name] = {
            "exact_rows": int(np.count_nonzero(row_equal)),
            "exact_fraction": float(np.mean(row_equal)),
            "maximum_absolute_difference": float(
                np.max(np.abs(left - right), initial=0.0)
            ),
        }
        matched &= bool(np.all(equal))
    for name in (
        "time",
        "history",
        "detected_channel",
        "boundary_kind",
        "last_triangle",
        "evidx",
        "interaction_cursor",
        "draw_cursor",
        "overflow",
        "global_ids",
    ):
        left = np.ascontiguousarray(cuda_values[name])
        right = np.ascontiguousarray(triton_values[name])
        equal = (
            left.view(np.uint32) == right.view(np.uint32)
            if left.dtype.kind == "f"
            else left == right
        )
        fields[name] = {
            "exact": int(np.count_nonzero(equal)),
            "exact_fraction": float(np.mean(equal)),
        }
        matched &= bool(np.all(equal))

    active_set = _active_set_comparison(cuda_values, triton_values)
    certificate = _certificate_comparison(cuda_values, triton_values)
    state_certificate = _state_certificate_comparison(
        cuda_values, triton_values
    )
    last_instance_invariant = _last_instance_invariant(triton_values)
    single_event_evidx_invariant = _single_event_evidx_invariant(
        cuda_values, triton_values
    )
    matched &= bool(active_set["exact"])
    matched &= bool(last_instance_invariant["satisfied"])
    matched &= bool(single_event_evidx_invariant["satisfied"])
    matched &= bool(certificate["draw_consumption_certified"])
    matched &= bool(state_certificate["post_interaction_state_certified"])
    return {
        "schema_version": 6,
        "detector": "detector_config_reflect_reflect3wires",
        "scope": "complete resident state after bounded full-history replay",
        "photons": args.count,
        "source_population": int(args.source_population or args.count),
        "global_photon_ids": _global_ids(args).tolist(),
        "max_steps": args.max_steps,
        "center": [float(value) for value in args.center],
        "voxel_size": float(args.voxel_size),
        "source_seed": args.source_seed,
        "tape_seed": args.tape_seed,
        "tape_interactions": int(args.tape_interactions),
        "draws_per_interaction": int(args.draws_per_interaction),
        "legacy_specular_reflection": True,
        "chroma_mesh_box_compatibility": True,
        "chroma_global_bvh_compatibility": True,
        "representable_progress_guard": False,
        "wire_scan_source_indices": [0, 1, 2, 3, 4, 5],
        "matched_bitwise": bool(matched),
        "draw_consumption_certified": bool(
            certificate["draw_consumption_certified"]
        ),
        "field_agreement": fields,
        "interaction_certificate": certificate,
        "first_certificate_difference": certificate["first_difference"],
        "state_certificate": state_certificate,
        "post_interaction_state_certified": bool(
            state_certificate["post_interaction_state_certified"]
        ),
        "first_state_difference": state_certificate["first_difference"],
        "last_instance_invariant": last_instance_invariant,
        "single_event_evidx_invariant": single_event_evidx_invariant,
        "active_set": active_set,
        "interaction_cursor_range": {
            "minimum": int(np.min(cuda_values["interaction_cursor"])),
            "maximum": int(np.max(cuda_values["interaction_cursor"])),
        },
        "tape_overflow_free": bool(certificate["overflow_free"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument(
        "--source-population", type=int, default=None,
        help="original source count used with --photon-id (default: --count)",
    )
    photon_selection = parser.add_mutually_exclusive_group()
    photon_selection.add_argument(
        "--photon-id", dest="photon_ids", type=int, action="append",
        help="replay one original global photon ID; repeat to select several",
    )
    photon_selection.add_argument(
        "--photon-id-file",
        type=Path,
        help="replay photon IDs from a JSON list or one integer per line",
    )
    photon_selection.add_argument(
        "--photon-id-from-npz",
        type=Path,
        help="replay the global active_ids tail stored in a prior worker NPZ",
    )
    parser.add_argument("--center", nargs=3, type=float, default=(-1000.0, 0.0, 0.0))
    parser.add_argument("--voxel-size", type=float, default=30.0)
    parser.add_argument("--source-seed", type=int, default=8123)
    parser.add_argument("--tape-seed", type=int, default=99173)
    parser.add_argument(
        "--tape-interactions",
        type=int,
        default=None,
        help=(
            "random-tape interaction rows per photon (default: max-steps + 1; "
            "the extra sentinel row prevents end-of-run cursor overflow)"
        ),
    )
    parser.add_argument("--draws-per-interaction", type=int, default=64)
    parser.add_argument(
        "--max-steps", type=int, default=1,
        help="physical interactions to execute per photon (one enables the detailed first-divergence report)",
    )
    parser.add_argument("--json", default=None)
    parser.add_argument(
        "--npz-prefix", default=None,
        help="retain worker arrays as PREFIX.cuda.npz and PREFIX.triton.npz",
    )
    parser.add_argument("--chroma-container", default=str(DEFAULT_CHROMA_CONTAINER))
    parser.add_argument("--_worker", choices=("cuda", "triton"), default=None)
    parser.add_argument("--_output", default=None)
    parser.add_argument("--_cuda-input", default=None)
    parser.add_argument("--_artifact-output", default=None)
    parser.add_argument("--_artifact-input", default=None)
    args = parser.parse_args()
    try:
        if args.photon_id_file is not None:
            args.photon_ids = _photon_ids_from_file(args.photon_id_file)
        elif args.photon_id_from_npz is not None:
            args.photon_ids = _photon_ids_from_npz(args.photon_id_from_npz)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if (
        args.photon_id_file is not None
        or args.photon_id_from_npz is not None
    ) and not args.photon_ids:
        parser.error("photon-ID selection is empty")
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.photon_ids:
        if len(set(args.photon_ids)) != len(args.photon_ids):
            parser.error("--photon-id values must be unique")
        population = int(args.source_population or args.count)
        if population <= 0:
            parser.error("--source-population must be positive")
        if min(args.photon_ids) < 0 or max(args.photon_ids) >= population:
            parser.error("--photon-id must be within the source population")
        args.source_population = population
        args.count = len(args.photon_ids)
    elif args.source_population is not None:
        parser.error("--source-population requires at least one --photon-id")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.tape_interactions is None:
        # Both tape implementations advance to the next interaction after a
        # completed physical step and immediately diagnose equality with the
        # tape extent as overflow.  Keep one unused row beyond max_steps so a
        # bounded replay can end with a valid cursor.
        args.tape_interactions = args.max_steps + 1
    if args.tape_interactions <= 0 or args.draws_per_interaction <= 0:
        parser.error("tape dimensions must be positive")
    if args.tape_interactions <= args.max_steps:
        parser.error("--tape-interactions must exceed --max-steps")

    if args._worker == "cuda":
        _cuda_worker(
            args, Path(args._output), Path(args._artifact_output)
        )
        return 0
    if args._worker == "triton":
        _triton_worker(
            args,
            Path(args._cuda_input),
            Path(args._artifact_input),
            Path(args._output),
        )
        return 0

    with tempfile.TemporaryDirectory(prefix="detector-lockstep-", dir="/tmp") as temp:
        temporary = Path(temp)
        cuda_path = temporary / "cuda.npz"
        triton_path = temporary / "triton.npz"
        global_bvh_path = temporary / "chroma-global-bvh.npz"
        cuda_run = subprocess.run(
            _worker_command(
                args, "cuda", cuda_path, global_bvh=global_bvh_path
            ),
            text=True, capture_output=True, check=False,
        )
        if cuda_run.returncode:
            raise RuntimeError(
                "CUDA worker failed with status %d:\n%s\n%s"
                % (cuda_run.returncode, cuda_run.stdout, cuda_run.stderr)
            )
        triton_run = subprocess.run(
            _worker_command(
                args,
                "triton",
                triton_path,
                cuda_input=cuda_path,
                global_bvh=global_bvh_path,
            ),
            text=True, capture_output=True, check=False,
        )
        if triton_run.returncode:
            raise RuntimeError(
                "Triton worker failed with status %d:\n%s\n%s"
                % (triton_run.returncode, triton_run.stdout, triton_run.stderr)
            )
        with np.load(cuda_path) as cuda_values, np.load(
            triton_path
        ) as triton_values:
            report = (
                _comparison(args, cuda_values, triton_values)
                if args.max_steps == 1
                else _comparison_multistep(args, cuda_values, triton_values)
            )
            report["provenance"] = _proof_provenance(
                args, cuda_values, triton_values
            )
        from chroma_lar.triton_scene import load_chroma_global_bvh_artifact

        certificate = load_chroma_global_bvh_artifact(global_bvh_path)
        report["global_bvh_certificate"] = {
            "mesh_md5": certificate.mesh_md5,
            "traversal_sha256": certificate.traversal_sha256,
            "optical_semantics_sha256": (
                certificate.optical_semantics_sha256
            ),
            "artifact_sha256": certificate.sha256,
        }
        if args.npz_prefix:
            prefix = Path(args.npz_prefix)
            prefix.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cuda_path, Path(str(prefix) + ".cuda.npz"))
            shutil.copy2(triton_path, Path(str(prefix) + ".triton.npz"))
            shutil.copy2(
                global_bvh_path,
                Path(str(prefix) + ".chroma-global-bvh.npz"),
            )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        destination = Path(args.json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    return 0 if report["matched_bitwise"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
