"""The browser gets native optical tables, with independently checked oracles."""

import json

import numpy as np
import pytest

from chroma_lar.webgpu_physics import (
    compare_ensemble,
    compare_recorded_paths,
    compare_terminal_states,
    cpu_reference,
    export_physics_catalog,
    physics_manifest,
    terminal_summary,
)


@pytest.mark.parametrize("name", ["prism", "fluorescence", "rayleigh"])
def test_export_preserves_compiled_triangle_and_optical_tables(name):
    manifest, fixture, scene = physics_manifest(name)
    parsed = json.loads(json.dumps(manifest, allow_nan=False))
    np.testing.assert_array_equal(parsed["triangles"], scene.bvh.triangle_vertices)
    np.testing.assert_array_equal(parsed["normals"], scene.normals)
    for name, array in (
        ("material1", scene.host.material1_index),
        ("material2", scene.host.material2_index),
        ("surface", scene.host.surface_index),
        ("channels", scene.host.triangle_channel_index),
    ):
        np.testing.assert_array_equal(parsed[name], array)
    for field in ("absorption_length", "scattering_length"):
        decoded = np.array(
            [
                [np.inf if value is None else value for value in row]
                for row in parsed["materials"][field]
            ],
            np.float32,
        )
        np.testing.assert_array_equal(decoded, getattr(scene.host.optics.materials, field))
    np.testing.assert_array_equal(parsed["materials"]["group_velocity"], scene.velocities)
    np.testing.assert_array_equal(
        parsed["surfaces"]["reemission_cdf"], scene.host.optics.surfaces.reemission_cdf
    )
    np.testing.assert_array_equal(parsed["surfaces"]["time_cdf"], scene.time_cdf)
    assert fixture.name == parsed["name"]


def test_catalog_files_have_actual_hashes_and_source_goldens(tmp_path):
    import hashlib

    catalog = export_physics_catalog(tmp_path)
    assert len(catalog["scenes"]) == 3
    for entry in catalog["scenes"]:
        contents = (tmp_path / entry["manifest"]).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == entry["sha256"]
        manifest = json.loads(contents)
        golden = manifest["source_golden"]
        _, fixture, _ = physics_manifest(entry["name"])
        batch = fixture.photons(32, seed=golden["seed"])
        ids = np.asarray(golden["photon_ids"])
        for field in ("pos", "direction", "polarization", "wavelengths", "times"):
            np.testing.assert_array_equal(golden[field], np.asarray(getattr(batch, field))[ids])
        for draw in manifest["rng"]["goldens"]:
            assert np.float32(draw["value"]).view(np.uint32) == draw["float32_bits"]


def test_comparison_cannot_hide_wrong_flags_or_nonfinite_state():
    oracle = cpu_reference("fluorescence", 192, paths=8)
    expected = oracle["result"]
    actual = {name: np.array(value, copy=True) for name, value in expected.final_state.items()}
    assert compare_terminal_states(actual, expected)["all_within_tolerances"]
    original_summary = terminal_summary(actual, "fluorescence")
    assert compare_ensemble(original_summary, original_summary)["passed"]
    actual["flags"][0] ^= np.uint32(4)
    actual["pos"][1, 0] = np.nan
    report = compare_terminal_states(actual, expected)
    assert not report["all_within_tolerances"]
    assert report["mismatches"]["flags"] == 1
    assert report["mismatches"]["pos"] == 1
    assert not report["finite"]
    # Failure evidence must remain serializable rather than hiding the actual
    # discrepancy behind a JSON NaN error in the browser validation harness.
    json.dumps(report, allow_nan=False)
    actual["wavelengths"][2] = np.nan
    summary = terminal_summary(actual, "fluorescence")
    json.dumps(summary, allow_nan=False)
    assert not compare_ensemble(summary, original_summary)["passed"]


def test_ensemble_check_rejects_a_large_fluorescence_bias():
    oracle = cpu_reference("fluorescence", 256)
    correct = oracle["summary"]
    biased = json.loads(json.dumps(correct))
    biased["flags"]["reemitted"] = 0
    report = compare_ensemble(biased, correct)
    assert not report["passed"]
    assert not report["checks"]["reemitted"]["passed"]


def test_recorded_paths_are_checked_at_every_vertex_and_match_event_endpoints():
    oracle = cpu_reference("prism", 32, paths=8)
    paths = oracle["paths"]
    records = []
    for index, photon in enumerate(paths.photon_ids):
        count = np.flatnonzero(paths.flags[:, index] & 15)[0] + 1
        records.append(
            dict(
                photon_id=int(photon),
                vertices=[
                    dict(
                        position=paths.positions[step, index].tolist(),
                        wavelength=float(paths.wavelengths[step, index]),
                        time=float(paths.times[step, index]),
                        flags=int(paths.flags[step, index]),
                    )
                    for step in range(count)
                ],
            )
        )
    final = oracle["result"].final_state
    assert compare_recorded_paths(records, paths, final)["passed"]
    records[0]["vertices"][1]["position"][0] += 1
    report = compare_recorded_paths(records, paths, final)
    assert not report["passed"]
    assert report["mismatches"]["position"] == 1
    records[1]["vertices"][-1]["flags"] ^= 4
    assert compare_recorded_paths(records, paths, final)["mismatches"]["endpoint"] == 1
