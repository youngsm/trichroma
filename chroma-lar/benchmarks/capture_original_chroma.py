"""Capture native Chroma XORWOW draws, auditing recorder transparency first.

Run inside the local Chroma container with an explicit --reference-root.
This is a reference/capture audit, NOT by itself a Triton parity certificate.
The default population fits one ordinary GPUPhotons.propagate launch, retaining
the original public API's normalization, RNG assignment and physical functions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np

FIELDS = (
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
    "event_index",
)


def photon_words(photons):
    """No float comparison, NaN canonicalization, or signed-zero normalization."""
    pieces = [
        getattr(photons, key).view(np.uint32).reshape(len(photons), -1)
        for key in (
            "pos",
            "dir",
            "pol",
            "wavelengths",
            "t",
            "flags",
            "last_hit_triangles",
            "weights",
            "evidx",
        )
    ]
    return np.ascontiguousarray(np.concatenate(pieces, axis=1))


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def export_scene(geometry, device, grid):
    """Capture exact uploaded table words, not a second interpolation of them."""
    import pycuda.driver as cuda

    def read(pointer, count, dtype):
        out = np.empty(count, dtype)
        if out.nbytes:
            cuda.memcpy_dtoh(out, int(pointer))
        return out

    data = {
        "nodes": geometry.bvh.nodes.view(np.uint32).reshape(-1, 4),
        "vertices": np.asarray(geometry.mesh.vertices, np.float32),
        "triangles": np.asarray(geometry.mesh.triangles, np.int32),
        "world_origin": np.asarray(geometry.bvh.world_coords.world_origin, np.float32),
        "world_scale": np.float32(geometry.bvh.world_coords.world_scale),
        "solid_ids": np.asarray(geometry.solid_id, np.int32),
        "channel_ids": np.asarray(geometry.solid_id_to_channel_index[geometry.solid_id], np.int32),
        "surface_ids": np.asarray(geometry.surface_index, np.int32),
        "material_inner": np.asarray(geometry.material1_index, np.int32),
        "material_outer": np.asarray(geometry.material2_index, np.int32),
        "wavelength_grid": np.asarray(grid, np.float32),
    }
    if getattr(geometry, "wireplanes", None):
        from chroma.cuda import srcdir

        if "u_norm" not in (Path(srcdir) / "geometry_types.h").read_text():
            raise NotImplementedError("this snapshot adapter expects the installed FP32 wire ABI")
        data["wire_words"] = np.stack(
            [read(pointer, 31, np.uint32) for pointer in device.wireplane_ptrs]
        )
    else:
        data["wire_words"] = np.empty((0, 31), np.uint32)
    material_fields = ("rindex", "absorption", "scattering")
    for i, field in enumerate(material_fields):
        data[field] = np.stack(
            [
                read(read(pointer, 7, np.uint64)[i], len(grid), np.float32)
                for pointer in device.material_ptrs
            ]
        )
    data["material_components"] = np.asarray(
        [read(int(pointer) + 7 * 8, 1, np.uint32)[0] for pointer in device.material_ptrs], np.int32
    )
    offsets = np.r_[0, np.cumsum(data["material_components"])].astype(np.int32)
    data["component_offsets"] = offsets
    total = int(offsets[-1])
    material_headers = [read(int(p) + 7 * 8, 7, np.uint32) for p in device.material_ptrs]
    time_header = material_headers[0][4:].copy()
    if any(not np.array_equal(h[4:], time_header) for h in material_headers):
        raise ValueError("capture expects the original common time grid")
    data["time_grid_spec"] = time_header
    for field, index, width in (
        ("component_prob", 3, len(grid)),
        ("component_wavelength_cdf", 4, len(grid)),
        ("component_time_cdf", 5, int(time_header[0])),
        ("component_absorption", 6, len(grid)),
    ):
        arrays = []
        for pointer, count in zip(device.material_ptrs, data["material_components"]):
            if count:
                pointers = read(read(pointer, 7, np.uint64)[index], int(count), np.uint64)
                arrays.extend(read(p, width, np.float32) for p in pointers)
        data[field] = np.stack(arrays) if total else np.zeros((1, width), np.float32)
    surface_fields = (
        "detect",
        "absorb",
        "reemit",
        "diffuse",
        "specular",
        "eta",
        "k",
        "reemission_cdf",
    )
    for i, field in enumerate(surface_fields):
        data["surface_" + field] = np.stack(
            [
                (
                    read(read(pointer, 10, np.uint64)[i], len(grid), np.float32)
                    if int(pointer)
                    else np.zeros(len(grid), np.float32)
                )
                for pointer in device.surface_ptrs
            ]
        )
    data["surface_models"] = np.asarray(
        [
            read(int(pointer) + 10 * 8, 1, np.uint32)[0] if int(pointer) else -1
            for pointer in device.surface_ptrs
        ],
        np.int32,
    )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=99173)
    parser.add_argument("--draws-per-interaction", type=int, default=64)
    parser.add_argument(
        "--max-steps", type=int, help="override the fixture's bounded interaction limit"
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "ballistic",
            "absorption",
            "competing_bulk",
            "rayleigh",
            "default_surface",
            "diffuse",
            "specular",
            "fresnel_p_30",
            "wls",
            "wls_loss",
        ],
    )
    args = parser.parse_args()
    if not 1 <= args.count < 16384 or args.count % 128:
        parser.error(
            "count must be a positive multiple of 128 below 16384 to preserve one native launch"
        )
    if args.draws_per_interaction < 1:
        parser.error("draw capacity must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    reference = args.reference_root.resolve()
    if not (reference / "chroma/cuda/photon.h").is_file():
        parser.error("reference-root must contain the original chroma package")
    sys.path.insert(0, str(reference))
    # Use this checkout's detector fixtures alongside the explicitly selected
    # original transport, rather than an unrelated container installation.
    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("PYCUDA_CACHE_DIR", "/tmp/trichroma-native-capture-cache")
    os.environ.setdefault("CHROMA_FORCE_SCATTER_AT_PASS", "0")
    Path(os.environ["PYCUDA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    import chroma
    from chroma import gpu
    from chroma.bvh.grid import make_recursive_grid_bvh
    from chroma.cuda import srcdir
    from chroma.gpu.tools import cuda_options
    import pycuda
    import pycuda.driver as cuda
    from pycuda import characterize, compiler, gpuarray as ga
    from original_chroma_cases import GRID, make_case

    if not Path(chroma.__file__).resolve().is_relative_to(reference):
        raise RuntimeError("the requested reference installation was not imported")
    if not Path(srcdir).resolve().is_relative_to(reference):
        raise RuntimeError("CUDA headers do not belong to the requested reference")
    if not hasattr(np.linalg, "linalg"):
        np.linalg.linalg = np.linalg
    args.output.mkdir(parents=True, exist_ok=True)
    package = reference / "chroma"
    source_hashes = {
        str(p.relative_to(reference)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*"))
        if p.is_file()
        and p.suffix in (".py", ".h", ".cu")
        and "__pycache__" not in p.parts
        and "_build_ext" not in p.parts
    }
    capture_source = Path(__file__).with_name("native_chroma_capture.cu")
    report = dict(
        reference=str(reference),
        imported_chroma=chroma.__file__,
        cuda_headers=str(srcdir),
        source_sha256=source_hashes,
        wrapper_sha256=hashlib.sha256(capture_source.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("optical_comparison_cases.py", "original_chroma_cases.py")
        },
        python=platform.python_version(),
        numpy=np.__version__,
        pycuda=pycuda.VERSION_TEXT,
        cuda_options=list(cuda_options),
        fields=FIELDS,
        scope="original public GPUPhotons.propagate versus native-XORWOW recorder; no Triton claim",
        count=args.count,
        seed=args.seed,
        cases=[],
    )
    context = gpu.create_cuda_context()
    try:
        report["device"] = cuda.Context.get_device().name()
        module = compiler.SourceModule(
            capture_source.read_text(),
            no_extern_c=True,
            options=[*cuda_options, "-I" + str(srcdir)],
        )
        capture = module.get_function("capture_original_chroma")
        rng_slots = 16384
        rng_bytes = rng_slots * characterize.sizeof(
            "curandStateXORWOW", "#include <curand_kernel.h>"
        )
        initial_rng = gpu.get_rng_states(rng_slots, seed=args.seed)
        initial_rng_words = np.empty(rng_bytes, np.uint8)
        cuda.memcpy_dtoh(initial_rng_words, initial_rng)
        report["initial_rng_sha256"] = digest(initial_rng_words)

        for name in args.cases:
            geometry, source, steps, _ = make_case(name, args.count)
            if args.max_steps is not None:
                steps = args.max_steps
            geometry.flatten()
            # Complete the fixture's explicit optical tables. The installed
            # original uploader validates wire-only objects against these
            # flattened lists even though it also builds its own pointer list.
            for plane in getattr(geometry, "wireplanes", None) or []:
                for table_name, keys in (
                    ("unique_materials", ("material_inner", "material_outer")),
                    ("unique_surfaces", ("surface",)),
                ):
                    table = list(getattr(geometry, table_name))
                    for key in keys:
                        value = plane[key]
                        if value not in table:
                            table.append(value)
                    setattr(geometry, table_name, np.asarray(table, dtype=object))
            geometry.bvh = make_recursive_grid_bvh(geometry.mesh, target_degree=3)
            device = gpu.GPUDetector(geometry, wavelengths=GRID)
            scene_data = export_scene(geometry, device, GRID)
            expected, actual = gpu.GPUPhotons(source), gpu.GPUPhotons(source)
            native_rng, capture_rng = cuda.mem_alloc(rng_bytes), cuda.mem_alloc(rng_bytes)
            cuda.memcpy_dtod(native_rng, initial_rng, rng_bytes)
            cuda.memcpy_dtod(capture_rng, initial_rng, rng_bytes)
            expected.propagate(
                device, native_rng, nthreads_per_block=128, max_blocks=128, max_steps=steps
            )
            tape = ga.empty((args.count, steps, args.draws_per_interaction), np.float32)
            tape.fill(np.nan)
            initial = ga.empty((args.count, len(FIELDS)), np.uint32)
            states = ga.empty((args.count, steps, len(FIELDS)), np.uint32)
            states.fill(np.uint32(0xFFFFFFFF))
            draws = ga.empty((args.count, steps), np.int32)
            draws.fill(-1)
            interactions = ga.zeros(args.count, np.int32)
            overflow = ga.zeros(args.count, np.uint32)
            capture(
                np.int32(args.count),
                capture_rng,
                actual.pos,
                actual.dir,
                actual.wavelengths,
                actual.pol,
                actual.t,
                actual.flags,
                actual.last_hit_triangles,
                actual.weights,
                actual.evidx,
                np.int32(steps),
                np.int32(0),
                np.int32(0),
                device.gpudata,
                tape,
                initial,
                states,
                draws,
                interactions,
                overflow,
                np.int32(args.draws_per_interaction),
                block=(128, 1, 1),
                grid=((args.count + 127) // 128, 1, 1),
            )
            cuda.Context.synchronize()
            native_words, captured_words = photon_words(expected.get()), photon_words(actual.get())
            native_rng_words = np.empty(rng_bytes, np.uint8)
            capture_rng_words = np.empty(rng_bytes, np.uint8)
            cuda.memcpy_dtoh(native_rng_words, native_rng)
            cuda.memcpy_dtoh(capture_rng_words, capture_rng)
            state_words, draw_counts = states.get(), draws.get()
            committed, overflows = interactions.get(), overflow.get()
            normalized_words = initial.get()
            recorded_tape = tape.get()
            # Replay the ordinary public API from the SAME initial input at
            # every prefix length. This ties each captured intermediate state
            # to an uninstrumented original execution, not merely the endpoint.
            prefix_checks = []
            for prefix in range(1, int(committed.max(initial=0)) + 1):
                plain = gpu.GPUPhotons(source)
                prefix_rng = cuda.mem_alloc(rng_bytes)
                cuda.memcpy_dtod(prefix_rng, initial_rng, rng_bytes)
                plain.propagate(
                    device, prefix_rng, nthreads_per_block=128, max_blocks=128, max_steps=prefix
                )
                plain_words = photon_words(plain.get())
                indices = np.minimum(committed, prefix) - 1
                if np.any(indices < 0):
                    raise NotImplementedError(
                        "prefix audit for initially terminal photons is pending"
                    )
                recorded_words = state_words[np.arange(args.count), indices]
                prefix_checks.append(
                    {
                        "steps": prefix,
                        "mismatched_words": int(np.count_nonzero(plain_words != recorded_words)),
                        "original_words_sha256": digest(plain_words),
                    }
                )
                del plain, prefix_rng
            difference = np.argwhere(native_words != captured_words)
            row = dict(
                name=name,
                max_steps=steps,
                original_words_sha256=digest(native_words),
                captured_words_sha256=digest(captured_words),
                initial_source_sha256=digest(photon_words(source)),
                normalized_source_sha256=digest(normalized_words),
                random_tape_sha256=digest(recorded_tape),
                geometry_sha256=digest(geometry.mesh.vertices)
                + ":"
                + digest(geometry.mesh.triangles),
                mismatched_words=int(len(difference)),
                first_mismatch=(
                    None
                    if not len(difference)
                    else dict(
                        photon=int(difference[0, 0]),
                        field=FIELDS[difference[0, 1]],
                        original_word=int(native_words[tuple(difference[0])]),
                        captured_word=int(captured_words[tuple(difference[0])]),
                    )
                ),
                native_rng_bytes_equal=bool(np.array_equal(native_rng_words, capture_rng_words)),
                capture_overflow=int(np.count_nonzero(overflows)),
                committed_interactions=int(committed.sum()),
                native_draws=int(np.maximum(draw_counts, 0).sum()),
                maximum_draw_count=int(draw_counts.max(initial=0)),
                unfinished_photons=int(np.count_nonzero((native_words[:, 11] & 32783) == 0)),
                prefix_checks=prefix_checks,
                scene_array_sha256={key: digest(value) for key, value in scene_data.items()},
            )
            row["passed"] = (
                not len(difference)
                and row["native_rng_bytes_equal"]
                and not row["capture_overflow"]
                and not any(p["mismatched_words"] for p in prefix_checks)
            )
            report["cases"].append(row)
            report["passed"] = all(r["passed"] for r in report["cases"])
            (args.output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
            np.savez_compressed(
                args.output / f"{name}.npz",
                original_words=native_words,
                captured_words=captured_words,
                source_words=photon_words(source),
                initial_words=normalized_words,
                state_words=state_words,
                draw_counts=draw_counts,
                interaction_counts=committed,
                random_tape=recorded_tape,
                capture_overflow=overflows,
                original_rng_words=native_rng_words.reshape(rng_slots, -1).view(np.uint32)[
                    : args.count, :6
                ],
                initial_rng_words=initial_rng_words.reshape(rng_slots, -1).view(np.uint32)[
                    : args.count, :6
                ],
                **scene_data,
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in ("prefix_checks", "scene_array_sha256")
                    }
                ),
                flush=True,
            )
            if not row["passed"]:
                raise AssertionError(
                    "recorder differs from original native Chroma; inspect saved evidence"
                )
            del (
                expected,
                actual,
                device,
                native_rng,
                capture_rng,
                tape,
                initial,
                states,
                draws,
                interactions,
                overflow,
            )
    finally:
        context.pop()


if __name__ == "__main__":
    main()
