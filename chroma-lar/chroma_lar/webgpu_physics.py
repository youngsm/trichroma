"""Trusted scene export and CPU oracles for the browser spectral playground.

This module does not implement a second optical model. It exports the tables
compiled for the native playground and runs its existing CPU spectral oracle.
JSON null means positive infinity only in interaction-length tables; every
other floating-point value is finite. Browser transport uses f32, so validation
reports pathwise discrepancies separately from ensemble agreement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from chroma.event import (
    BULK_ABSORB,
    NO_HIT,
    RAYLEIGH_SCATTER,
    REFLECT_DIFFUSE,
    REFLECT_SPECULAR,
    SURFACE_ABSORB,
    SURFACE_DETECT,
    SURFACE_REEMIT,
    SURFACE_TRANSMIT,
)
from chroma.triton.optical_response import uniform
from chroma.triton.spectral import SpectralScene, SpectralSimulation, STEP_LIMIT
from .optical_showcase import SCENES, build_playground_scene, trace_photon_subset

FORMAT = "trichroma-webgpu-spectral-v1"
FLAGS = dict(
    escaped=NO_HIT,
    bulk_absorbed=BULK_ABSORB,
    detected=SURFACE_DETECT,
    surface_absorbed=SURFACE_ABSORB,
    scattered=RAYLEIGH_SCATTER,
    diffuse=REFLECT_DIFFUSE,
    reflected=REFLECT_SPECULAR,
    reemitted=SURFACE_REEMIT,
    transmitted=SURFACE_TRANSMIT,
    step_limit=int(STEP_LIMIT),
    aborted=1 << 31,
)
TERMINAL = NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | int(STEP_LIMIT) | (1 << 31)


def _length_json(array):
    values = np.asarray(array, float)
    if np.any(np.isnan(values) | (values < 0)):
        raise ValueError("length tables must be nonnegative and not NaN")
    result = values.astype(object)
    result[np.isposinf(values)] = None
    return result.tolist()


def _rng_goldens():
    ids = np.array([0, 1, 2, 7, 2**32 - 1, 2**32, 2**40 + 71], np.uint64)
    rows = []
    for seed in (901, 2**32 + 17):
        for stream in (0, 17, 32, 0x10000000, 0x10000003):
            values = uniform(ids, seed, stream)
            for photon, value, word in zip(ids, values, values.view(np.uint32)):
                rows.append(
                    dict(
                        id_low=int(photon & np.uint64(0xFFFFFFFF)),
                        id_high=int(photon >> np.uint64(32)),
                        seed_low=seed & 0xFFFFFFFF,
                        seed_high=seed >> 32,
                        stream=stream,
                        value=float(value),
                        float32_bits=int(word),
                    )
                )
    return rows


def physics_manifest(name, *, fixture=None):
    """Return portable table data and the corresponding independent CPU scene."""
    fixture = build_playground_scene(name) if fixture is None else fixture
    if fixture.name != name:
        raise ValueError("fixture does not match requested experiment")
    scene = SpectralScene.compile(fixture.geometry, wavelengths=fixture.wavelengths)
    host, mat, surf = scene.host, scene.host.optics.materials, scene.host.optics.surfaces
    grid = host.optics.wavelength_grid
    source = fixture.photons(32, seed=901)
    selected = np.array([0, 1, 2, 7, 31])
    source_golden = {
        key: np.asarray(getattr(source, field))[selected].tolist()
        for key, field in (
            ("pos", "pos"),
            ("direction", "direction"),
            ("polarization", "polarization"),
            ("wavelengths", "wavelengths"),
            ("times", "times"),
            ("photon_ids", "global_photon_ids"),
        )
    }
    manifest = dict(
        format=FORMAT,
        version=1,
        name=name,
        title=fixture.title,
        explanation=fixture.explanation,
        scene_fingerprint=scene.fingerprint,
        max_steps=256,
        units=dict(length="mm", wavelength="nm", time="ns"),
        wavelengths=dict(start=float(grid.start), step=float(grid.step), count=int(grid.count)),
        triangles=scene.bvh.triangle_vertices.tolist(),
        normals=scene.normals.tolist(),
        material1=host.material1_index.tolist(),
        material2=host.material2_index.tolist(),
        surface=host.surface_index.tolist(),
        channels=host.triangle_channel_index.tolist(),
        materials=dict(
            names=list(mat.names),
            refractive_index=mat.refractive_index.tolist(),
            absorption_length=_length_json(mat.absorption_length),
            scattering_length=_length_json(mat.scattering_length),
            group_velocity=scene.velocities.tolist(),
        ),
        surfaces=dict(
            names=list(surf.names),
            present=surf.present.tolist(),
            model=surf.model.tolist(),
            detect=surf.detect.tolist(),
            absorb=surf.absorb.tolist(),
            reflect_diffuse=surf.reflect_diffuse.tolist(),
            reflect_specular=surf.reflect_specular.tolist(),
            reemit=surf.reemit.tolist(),
            reemission_cdf=surf.reemission_cdf.tolist(),
            time_offsets=scene.time_offsets.tolist(),
            time_x=scene.time_x.tolist(),
            time_cdf=scene.time_cdf.tolist(),
            time_pdf=scene.time_pdf.tolist(),
            reemit_to_material1=scene.reemit_to_material1.tolist(),
        ),
        source=dict(
            center=list(fixture.source_center),
            band=list(fixture.source_band),
            beam_width=2.0,
            direction=[1, 0, 0],
            time0=0,
            polarization_options=["random", "y", "z"],
            random_polarization="uniform linear angle in the y/z plane",
            streams=dict(
                y=0x10000000, z=0x10000001, polarization=0x10000002, wavelength=0x10000003
            ),
        ),
        display=dict(
            monitor_box=[600, 400, 240],
            outline=None if fixture.outline is None else fixture.outline.tolist(),
            projection="x/y",
            ultraviolet="violet false color",
            radiance=False,
        ),
        flags=FLAGS,
        rng=dict(name="philox4x32-10-id64-stream32-seed64-v1", goldens=_rng_goldens()),
        source_golden=dict(seed=901, polarization_mode="random", **source_golden),
        scope="Synthetic dispersive dielectric, polarized Rayleigh and effective fluorescent surface; no bulk reemission",
        precision="Float32 geometry/state/tables; WGSL float32 arithmetic. CPU oracle uses float64 triangle queries.",
        length_encoding="null is +Infinity only in absorption_length and scattering_length",
    )
    # No non-standard JSON Infinity/NaN tokens may silently reach the browser.
    json.dumps(manifest, allow_nan=False)
    return manifest, fixture, scene


def export_physics_scene(name, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest, fixture, scene = physics_manifest(name)
    payload = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
    filename = f"physics-{name}.json"
    (destination / filename).write_bytes(payload)
    entry = dict(
        name=name,
        title=manifest["title"],
        manifest=filename,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_length=len(payload),
        scene_fingerprint=scene.fingerprint,
    )
    return entry, fixture, scene


def export_physics_catalog(destination):
    destination = Path(destination)
    entries = [export_physics_scene(name, destination)[0] for name in SCENES]
    catalog = dict(format=FORMAT, scenes=entries)
    (destination / "physics-catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog


def terminal_summary(state, name):
    """Full-count observables with explicit histogram overflow and terminal audit."""
    flags = np.asarray(state["flags"], np.uint32)
    detected = (flags & SURFACE_DETECT) != 0
    summary = dict(
        scene=name,
        count=len(flags),
        flags={key: int(np.count_nonzero(flags & flag)) for key, flag in FLAGS.items()},
        unfinished=int(np.count_nonzero((flags & TERMINAL) == 0)),
    )
    summary["finite"] = all(
        np.isfinite(state[key]).all()
        for key in ("pos", "direction", "polarization", "wavelengths", "times")
    )
    for key, edges in (
        ("wavelengths", np.linspace(280, 740, 93)),
        ("times", np.linspace(0, 220 if name == "fluorescence" else 20, 221)),
    ):
        values = np.asarray(state[key])[detected]
        counts, _ = np.histogram(values, bins=edges)
        summary[key] = dict(
            edges=edges.tolist(),
            counts=counts.tolist(),
            underflow=int(np.count_nonzero(values < edges[0])),
            overflow=int(np.count_nonzero(values > edges[-1])),
            invalid=int(np.count_nonzero(~np.isfinite(values))),
            mean=float(values.mean()) if len(values) and np.isfinite(values).all() else None,
            std=float(values.std()) if len(values) and np.isfinite(values).all() else None,
        )
    return summary


def cpu_reference(
    name,
    count=4096,
    *,
    seed=901,
    paths=0,
    polarization="random",
    max_steps=256,
    fixture=None,
    scene=None,
):
    """Reference results from existing CPU transport, not the browser shader."""
    if (fixture is None) != (scene is None):
        raise ValueError("supply both fixture and compiled scene, or neither")
    if fixture is None:
        _, fixture, scene = physics_manifest(name)
    if fixture.name != name:
        raise ValueError("reference fixture does not match the requested experiment")
    batch = fixture.photons(count, seed=seed, polarization=polarization)
    # These fixtures contain only12–20triangles. Exhaustive FP64 triangle
    # queries are both a simpler oracle and faster than Python BVH traversal.
    simulation = SpectralSimulation(scene, backend="reference")
    result = simulation.simulate(batch, seed=seed, max_steps=max_steps)
    if result.step_limit_count:
        raise RuntimeError("CPU reference reached max_steps; increase the oracle step limit")
    summary = terminal_summary(result.final_state, name)
    trajectory = None
    if paths:
        trajectory = trace_photon_subset(
            simulation, batch, seed=seed, count=min(paths, count), max_steps=max_steps
        )
    return dict(
        name=name,
        seed=seed,
        polarization=polarization,
        scene_fingerprint=scene.fingerprint,
        summary=summary,
        result=result,
        paths=trajectory,
        batch=batch,
    )


def compare_terminal_states(
    actual,
    expected,
    *,
    position_atol=0.02,
    time_atol=0.002,
    wavelength_atol=0.002,
    vector_atol=0.001,
):
    """Report paired agreement honestly; do not silently drop diverging photons.

    Tolerances cover f32 arithmetic in these millimetre-scale scenes. Discrete
    flags/channels and integer IDs remain exact comparisons. Returned physical
    audit failures and disagreement counts must be visible in validation output.
    """
    expected = expected.final_state if hasattr(expected, "final_state") else expected
    fields = ("pos", "direction", "polarization", "wavelengths", "times", "flags", "channels")
    missing = [key for key in fields if key not in actual or key not in expected]
    if missing:
        raise ValueError("terminal debug state is missing fields: " + ", ".join(missing))
    n = len(expected["flags"])
    report = dict(
        count=n,
        mismatches={},
        tolerances=dict(
            position_mm=position_atol,
            time_ns=time_atol,
            wavelength_nm=wavelength_atol,
            vector=vector_atol,
        ),
    )
    actual = {key: np.asarray(value) for key, value in actual.items()}
    for key in fields:
        if actual[key].shape != np.asarray(expected[key]).shape:
            raise ValueError(f"{key} shape differs between browser and CPU")
    for key in ("flags", "channels"):
        report["mismatches"][key] = int(np.count_nonzero(actual[key] != expected[key]))
    for key, tolerance in (
        ("pos", position_atol),
        ("times", time_atol),
        ("wavelengths", wavelength_atol),
        ("direction", vector_atol),
        ("polarization", vector_atol),
    ):
        delta = np.abs(actual[key].astype(float) - expected[key])
        bad = ~np.isfinite(delta) | (delta > tolerance)
        per_photon = bad if bad.ndim == 1 else bad.any(axis=1)
        report["mismatches"][key] = int(np.count_nonzero(per_photon))
        report[key + "_max_abs"] = (
            (float(delta.max()) if delta.size else 0.0) if np.isfinite(delta).all() else None
        )
    if "photon_ids" in actual:
        report["mismatches"]["photon_ids"] = int(
            np.count_nonzero(actual["photon_ids"] != expected["photon_ids"])
        )
    report["all_within_tolerances"] = not any(report["mismatches"].values())
    report["finite"] = all(np.isfinite(actual[key]).all() for key in fields)
    for key in ("direction", "polarization"):
        error = np.abs(np.linalg.norm(actual[key], axis=1) - 1)
        report[key + "_norm_max_error"] = float(error.max()) if np.isfinite(error).all() else None
    dot = np.abs(np.sum(actual["direction"] * actual["polarization"], axis=1))
    report["polarization_dot_max_abs"] = float(dot.max()) if np.isfinite(dot).all() else None
    return report


def compare_ensemble(actual, expected, *, family_alpha=1e-6):
    """Conservative Hoeffding/DKW checks for counts and binned distributions.

    This evaluates marginal agreement, not event-by-event equivalence. The
    Bonferroni allocation controls the family error bound of the listed tests.
    Report paired comparisons separately when stable photon IDs are available.
    """
    names = tuple(FLAGS) + ("wavelengths", "times")
    n, m = actual["count"], expected["count"]
    if n <= 0 or m <= 0 or not 0 < family_alpha < 1:
        raise ValueError("comparison needs positive counts and alpha in (0,1)")
    per_test_alpha = family_alpha / len(names)
    bound = np.sqrt(np.log(4 / per_test_alpha) / (2 * n)) + np.sqrt(
        np.log(4 / per_test_alpha) / (2 * m)
    )
    checks = {}
    for key in FLAGS:
        difference = abs(actual["flags"][key] / n - expected["flags"][key] / m)
        checks[key] = dict(
            difference=difference, bound=float(bound), passed=bool(difference <= bound)
        )
    for key in ("wavelengths", "times"):
        a, b = actual[key], expected[key]
        if not np.array_equal(a["edges"], b["edges"]):
            raise ValueError("histogram edges must match for ensemble comparison")
        ca = np.r_[a["underflow"], a["counts"], a["overflow"]].astype(float)
        cb = np.r_[b["underflow"], b["counts"], b["overflow"]].astype(float)
        if not ca.sum() or not cb.sum():
            checks[key] = dict(
                passed=bool(ca.sum() == cb.sum()), reason="empty detected population"
            )
            continue
        distribution_bound = np.sqrt(np.log(4 / per_test_alpha) / (2 * ca.sum())) + np.sqrt(
            np.log(4 / per_test_alpha) / (2 * cb.sum())
        )
        distance = float(np.max(np.abs(np.cumsum(ca) / ca.sum() - np.cumsum(cb) / cb.sum())))
        checks[key] = dict(
            cdf_distance=distance,
            bound=float(distribution_bound),
            passed=bool(distance <= distribution_bound),
        )
    # These three scenes have a closed ideal monitor and no bulk absorption.
    # A rare leak or inactive-medium absorption is a failure even when too
    # infrequent for the finite-sample statistical tests to detect.
    audit = bool(
        actual["finite"]
        and actual["unfinished"] == 0
        and actual["flags"]["step_limit"] == 0
        and actual["flags"]["aborted"] == 0
        and actual["flags"]["escaped"] == 0
        and actual["flags"]["bulk_absorbed"] == 0
    )
    return dict(
        family_alpha=family_alpha,
        checks=checks,
        physical_audit=audit,
        passed=audit and all(value["passed"] for value in checks.values()),
        meaning="Marginal ensemble compatibility; does not establish paired or universal bitwise equivalence",
    )


def compare_recorded_paths(
    actual, expected, terminal_state, *, position_atol=0.02, time_atol=0.002, wavelength_atol=0.002
):
    """Check every browser path vertex against CPU interaction-prefix results.

    Also require recorded endpoints to equal the browser's own debug state,
    which establishes that the visualization belongs to the simulated event.
    """
    if len(actual) != len(expected.photon_ids):
        raise ValueError("browser and oracle path counts differ")
    mismatches = dict(
        photon_ids=0,
        vertex_count=0,
        position=0,
        time=0,
        wavelength=0,
        flags=0,
        endpoint=0,
        arrival_time=0,
    )
    vertices_checked = 0
    for index, path in enumerate(actual):
        photon = int(expected.photon_ids[index])
        mismatches["photon_ids"] += path["photon_id"] != photon
        terminal = np.flatnonzero(expected.flags[:, index] & TERMINAL)
        if not len(terminal):
            raise ValueError("CPU reference path is incomplete")
        count = int(terminal[0]) + 1
        vertices = path["vertices"]
        mismatches["vertex_count"] += len(vertices) != count
        if not vertices:
            mismatches["endpoint"] += 1
            continue
        if all("arrival_time" in vertex for vertex in vertices):
            arrivals = np.asarray([vertex["arrival_time"] for vertex in vertices])
            interaction_times = np.asarray([vertex["time"] for vertex in vertices])
            previous_times = np.r_[0, interaction_times[:-1]]
            mismatches["arrival_time"] += int(
                np.count_nonzero(
                    ~np.isfinite(arrivals)
                    | (arrivals > interaction_times + time_atol)
                    | (arrivals < previous_times - time_atol)
                )
            )
        stop = min(count, len(vertices))
        for key, field, tolerance in (
            ("position", expected.positions[:stop, index], position_atol),
            ("time", expected.times[:stop, index], time_atol),
            ("wavelength", expected.wavelengths[:stop, index], wavelength_atol),
        ):
            values = np.asarray([vertex[key] for vertex in vertices[:stop]])
            delta = np.abs(values - field)
            bad = ~np.isfinite(delta) | (delta > tolerance)
            mismatches[key] += int(np.count_nonzero(bad.any(axis=1) if bad.ndim > 1 else bad))
        mismatches["flags"] += int(
            np.count_nonzero(
                np.asarray([vertex["flags"] for vertex in vertices[:stop]], np.uint32)
                != expected.flags[:stop, index]
            )
        )
        vertices_checked += stop
        last = vertices[-1]
        endpoint = all(
            np.array_equal(last[a], np.asarray(terminal_state[b])[photon])
            for a, b in (
                ("position", "pos"),
                ("time", "times"),
                ("wavelength", "wavelengths"),
                ("flags", "flags"),
            )
        )
        mismatches["endpoint"] += not endpoint
    return dict(
        paths=len(actual),
        vertices_checked=vertices_checked,
        mismatches={key: int(value) for key, value in mismatches.items()},
        passed=bool(expected.complete and not any(mismatches.values())),
    )


def check_browser_histograms(result, state):
    """Recount browser GPU atomics from its independently downloaded state.

    Bins intentionally reproduce the browser's documented clamped edges. The
    separate statistical oracle uses explicit underflow/overflow bins instead.
    """
    state = {key: np.asarray(value) for key, value in state.items()}
    flags = state["flags"].astype(np.uint32)
    detected = (flags & SURFACE_DETECT) != 0
    reemitted = (flags & SURFACE_REEMIT) != 0
    scattered = (flags & RAYLEIGH_SCATTER) != 0
    checks = {}

    def check(key, actual, expected):
        checks[key] = dict(
            passed=bool(np.array_equal(actual, expected)),
            mismatched_bins=int(np.count_nonzero(np.asarray(actual) != expected)),
        )

    def histogram(key, values, low, high, count):
        values = np.asarray(values, np.float32)
        # Supported bin edges are exactly representable. Direct comparisons
        # avoid an independent floating-point quotient-rounding accident.
        bins = np.clip(
            np.searchsorted(np.linspace(low, high, count + 1), values, side="right") - 1,
            0,
            count - 1,
        )
        check(key, result["histograms"][key], np.bincount(bins, minlength=count))

    for name in result["counts"]:
        if name in FLAGS:
            check("count_" + name, result["counts"][name], np.count_nonzero(flags & FLAGS[name]))
    time_max = result["histograms"]["time_max"]
    check(
        "time_overflow",
        result["counts"]["time_overflow"],
        np.count_nonzero(state["times"][detected] >= time_max),
    )
    check("max_steps", result["counts"]["max_steps"], int(state["steps"].max()))
    histogram("source", state["source_wavelengths"], 280, 740, 92)
    histogram("detected", state["wavelengths"][detected], 280, 740, 92)
    histogram("arrival", state["times"][detected], 0, time_max, 128)
    histogram("delay", state["fluorescence_delay"][reemitted], 0, 200, 80)
    histogram("scattered", state["source_wavelengths"][scattered], 390, 710, 32)
    histogram("scatter_source", state["source_wavelengths"], 390, 710, 32)
    forward = detected & (state["pos"][:, 0] > 299) & (result["scene"] == "prism")
    check("forward_monitor", result["counts"]["forward_monitor"], np.count_nonzero(forward))
    wavelengths = state["wavelengths"][forward].astype(np.float32)
    direction = state["direction"][forward].astype(np.float32)
    angles = np.arctan2(direction[:, 1], direction[:, 0]) * np.float32(57.29577951308232)
    wavelength_bin = np.clip((wavelengths - np.float32(380)) / np.float32(4), 0, 84).astype(int)
    angle_bin = np.clip(angles + np.float32(90), 0, 179).astype(int)
    check(
        "dispersion",
        result["histograms"]["dispersion"],
        np.bincount(angle_bin * 85 + wavelength_bin, minlength=15300),
    )
    return dict(passed=all(row["passed"] for row in checks.values()), checks=checks)


def compare_browser_source(state, batch):
    """Check all browser source coordinates/wavelengths with stable photon IDs."""
    checks = {}
    for key, expected, tolerance in (
        ("source_wavelengths", batch.wavelengths, 0.0001),
        ("source_y", batch.pos[:, 1], 0.00001),
        ("source_z", batch.pos[:, 2], 0.00001),
    ):
        actual = np.asarray(state[key])
        if actual.shape != expected.shape:
            raise ValueError(f"browser source {key} shape differs from the CPU source")
        delta = np.abs(actual - expected)
        finite = bool(np.isfinite(delta).all())
        checks[key] = dict(
            tolerance=tolerance,
            max_abs=float(delta.max()) if finite else None,
            mismatches=int(np.count_nonzero(~np.isfinite(delta) | (delta > tolerance))),
        )
    return dict(passed=all(row["mismatches"] == 0 for row in checks.values()), checks=checks)
