"""Independent conservation and normalization checks for photon-camera maps."""

import json

import numpy as np
import pytest

from chroma_lar.photon_camera import (
    camera_color_table,
    camera_source_energy_sum,
    camera_source_is_fill,
    camera_source_scales,
    camera_manifest,
    collect_camera_reference,
    compare_camera_maps,
    fluorescence_radiance,
    integrate_constant_segment,
    packet_energy,
    polarization_moments,
    rayleigh_source,
    wall_radiance,
    wall_bin_boundary_oracle,
)
from chroma_lar.webgpu_physics import physics_manifest


@pytest.mark.parametrize("name", ["prism", "fluorescence", "rayleigh"])
def test_camera_scenes_export_haze_separately_from_diagnostic_fixtures(name):
    manifest, _, scene = camera_manifest(name)
    json.dumps(manifest, allow_nan=False)
    assert manifest["camera"]["wavelength_bins"] == 64
    assert manifest["camera"]["volume_shape"] == [64, 40, 24]
    assert manifest["camera"]["default_eye"][0] < 0  # Faces transmitted prism caustic.
    original, _, _ = physics_manifest(name)
    if name != "rayleigh":
        air = manifest["materials"]["names"].index("synthetic_air")
        assert np.isfinite(manifest["materials"]["scattering_length"][air]).all()
        original_air = original["materials"]["names"].index("synthetic_air")
        assert all(
            value is None for value in original["materials"]["scattering_length"][original_air]
        )
    if name == "fluorescence":
        glass = manifest["materials"]["names"].index("synthetic_fluorescent_camera_glass")
        assert np.all(scene.host.optics.materials.absorption_length[glass] == 60)
        assert manifest["source"]["beam_width"] == 20
    else:
        assert np.all(np.isinf(scene.host.optics.materials.absorption_length))


def test_radiance_does_not_double_when_photon_budget_doubles():
    packets = packet_energy([410, 450, 520, 610])
    assert wall_radiance(packets.sum(), 4, 20) == wall_radiance(2 * packets.sum(), 8, 20)
    m = polarization_moments([[0, 1, 0]] * 4, packets).sum(axis=0)
    a = rayleigh_source(m, [0, 0, 1], 4, 10)
    assert a == rayleigh_source(2 * m, [0, 0, 1], 8, 10)
    assert fluorescence_radiance(3, 4, 20, 0.5) == fluorescence_radiance(6, 8, 20, 0.5)
    assert packet_energy(305) > packet_energy(485)  # WLS loses radiant energy.


def test_polarized_rayleigh_source_integrates_to_deposited_energy():
    cosine, weight = np.polynomial.legendre.leggauss(8)
    phi = np.arange(16) * (2 * np.pi / 16)
    directions = np.stack(
        np.broadcast_arrays(
            np.sqrt(1 - cosine[:, None] ** 2) * np.cos(phi),
            np.sqrt(1 - cosine[:, None] ** 2) * np.sin(phi),
            cosine[:, None],
        ),
        axis=-1,
    )
    moment = polarization_moments([0, 1, 0], 7)
    source = rayleigh_source(moment, directions, 100, 20)
    integral = np.sum(source * weight[:, None]) * 2 * np.pi / 16
    assert integral == pytest.approx(7 / (100 * 20), abs=1e-15)
    assert rayleigh_source(moment, [0, 1, 0], 100, 20) == 0


def test_random_transverse_camera_polarization_recovers_unpolarized_phase():
    moment = polarization_moments([1, 2, 3] / np.sqrt(14), 9)
    angle = np.arange(16) * 2 * np.pi / 16
    transverse = np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=-1)
    polarized = 2 * 3 / (8 * np.pi) * np.einsum("ni,ij,nj->n", transverse, moment, transverse)
    assert polarized.mean() == pytest.approx(rayleigh_source(moment, [0, 0, 1], 1, 1))


def test_attenuated_segment_is_invariant_to_step_subdivision():
    full = integrate_constant_segment(2.4, 0.01, 90)
    first = integrate_constant_segment(2.4, 0.01, 40)
    second = np.exp(-0.01 * 40) * integrate_constant_segment(2.4, 0.01, 50)
    assert full == pytest.approx(first + second, rel=1e-14)
    assert integrate_constant_segment(2.4, 0, 90) == pytest.approx(216)


def test_uniform_hemisphere_emitter_conserves_projected_flux():
    cosine, weight = np.polynomial.legendre.leggauss(16)
    cosine = (cosine + 1) / 2
    radiance = fluorescence_radiance(11, 100, 20, cosine)
    integrated = np.sum(radiance * cosine * weight / 2) * 2 * np.pi
    assert integrated == pytest.approx(11 / (100 * 20))


def test_camera_color_response_does_not_render_uv_as_visible_violet():
    colors = camera_color_table()
    assert np.array_equal(colors["linear_srgb"][1], [0, 0, 0])  # 300nm source.
    assert np.isfinite(colors["linear_srgb"]).all()
    assert np.any(np.asarray(colors["linear_srgb"]) < 0)  # Retain out-of-gamut basis until sum.


def test_collision_oracle_records_incoming_polarization_and_all_wall_arrivals():
    oracle = collect_camera_reference("rayleigh", 32)
    assert len(oracle["records"]["wall"]) == 32
    assert oracle["records"]["volume"]
    source = oracle["fixture"].photons(32, seed=oracle["seed"])
    first = next(record for record in oracle["records"]["volume"] if record["step"] == 1)
    np.testing.assert_array_equal(
        first["incoming_polarization"], source.polarization[first["photon_id"]]
    )
    assert not oracle["records"]["fluorescence"]


def test_weighted_area_source_is_transverse_stable_and_energy_normalized():
    manifest, fixture, _ = camera_manifest("prism")
    small, large = fixture.photons(257, seed=91), fixture.photons(513, seed=91)
    for field in ("pos", "direction", "polarization", "wavelengths", "weights"):
        np.testing.assert_array_equal(getattr(small, field), getattr(large, field)[:257])
    fill = camera_source_is_fill(257, 91)
    assert fill.any() and (~fill).any()
    assert np.all(small.direction[fill, 2] < 0)
    assert np.all(small.pos[fill, 2] == np.float32(119.999))
    assert np.max(np.abs(np.sum(small.direction * small.polarization, axis=1))) < 1e-6
    energy = packet_energy(small.wavelengths) * camera_source_scales(257, 91, small.wavelengths)
    assert energy.sum() == pytest.approx(camera_source_energy_sum("prism", 257, 91), rel=1e-7)
    assert np.allclose(
        energy[fill],
        manifest["camera"]["fill_light"]["integrated_radiance"] * 180 * 80 * np.pi / 0.2,
        rtol=1e-7,
    )


def test_map_audit_rejects_missing_signed_cells_and_nonfinite_values():
    expected = dict(wall=[[3, 2.0]], fill_wall=[], volume=[[9, -0.25]], wls=[])
    actual = dict(wall=[[3, 2.0]], fill_wall=[], volume=[], wls=[])
    assert not compare_camera_maps(actual, expected)["passed"]
    actual["volume"] = [[9, float("nan")]]
    report = compare_camera_maps(actual, expected)
    assert not report["passed"]
    assert report["comparisons"]["volume"]["browser_sum"] is None
    json.dumps(report, allow_nan=False)


def test_curved_pmt_charts_and_vuv_conversion_match_repository_calibration():
    manifest, fixture, scene = camera_manifest("pmt")
    pmt = manifest["camera"]["pmt"]
    roles = np.asarray(pmt["triangle_role"])
    charts = np.asarray(pmt["triangle_chart"])
    assert np.sum(np.asarray(pmt["triangle_area_mm2"])[roles == 0]) == pytest.approx(960000)
    assert pmt["wls_charts"] == 624
    np.testing.assert_array_equal(charts[roles == 1], np.arange(624))
    assert np.all(charts[roles != 1] == -1)
    assert np.all(np.asarray(pmt["triangle_area_mm2"]) > 0)
    assert pmt["calibration_status"] == "synthetic"
    assert pmt["clip_plane_normal"] == [0, 1, 0]
    assert not pmt["ultraviolet"]["default_visible"]
    assert pmt["profile_rings"] == 20 and pmt["azimuth_segments"] == 48
    bvh = manifest["camera"]["geometry_bvh"]
    np.testing.assert_array_equal(bvh["nodes"], scene.bvh.nodes)
    assert len(bvh["escape_links"]) == bvh["node_count"]
    assert bvh["bounds_padding_quantization_units"] == 1
    assert manifest["wavelengths"]["start"] == 120
    assert fixture.source_band == (126, 130)
    assert fixture.fluorescence_time.x[-1] == 10000
    oracle = collect_camera_reference("pmt", 128, fixture=fixture, scene=scene)
    converted = oracle["records"]["fluorescence"]
    assert converted and oracle["records"]["photocathode"]
    assert all(400 <= event["wavelength"] <= 460 for event in converted)
    assert all(event["normal"] is not None for event in converted)
    beam = ~camera_source_is_fill(128, 901)
    for event in oracle["records"]["photocathode"]:
        if beam[event["photon_id"]]:
            assert 400 <= event["wavelength"] <= 460  # VUV cannot cross the absorbing glass.


def test_wall_quantization_audit_reports_edges_without_hiding_flight_errors():
    manifest, _, _ = camera_manifest("prism")
    record = dict(
        photon_id=0,
        position=[-84.3748, -200, -16],
        wavelength=450,
        energy=1.25,
        source_is_fill=False,
        normal=[0, -1, 0],
        triangle=4,
    )
    oracle = dict(records=dict(wall=[record], volume=[], fluorescence=[], photocathode=[]))
    browser = dict(pos=[[-84.3752, -200, -16]], last_hit_triangles=[4])
    result = wall_bin_boundary_oracle(browser, oracle, manifest)
    assert result["valid"] and len(result["ambiguities"]) == 1
    crossing = result["ambiguities"][0]
    assert crossing["crossed_edges"] == [dict(axis=0, edge_mm=-84.375)]
    assert crossing["cpu_index"] != crossing["browser_position_index"]
    assert record["position"][0] == -84.3748  # Reference trajectory is unchanged.
    browser["pos"][0][0] = -84.5
    assert not wall_bin_boundary_oracle(browser, oracle, manifest)["valid"]
    browser["pos"][0][0] = -90
    assert not wall_bin_boundary_oracle(browser, oracle, manifest)["valid"]
    browser["pos"][0][0] = -84.3752
    browser["last_hit_triangles"][0] = 5
    assert not wall_bin_boundary_oracle(browser, oracle, manifest)["valid"]


@pytest.mark.parametrize("name", ["prism", "fluorescence", "rayleigh", "pmt"])
def test_compacted_oracle_preserves_every_collision_and_terminal_state(name):
    _, fixture, scene = camera_manifest(name)
    full = collect_camera_reference(name, 128, fixture=fixture, scene=scene, compact_prefixes=False)
    compact = collect_camera_reference(name, 128, fixture=fixture, scene=scene)
    assert full["records"] == compact["records"]
    for field, expected in full["result"].final_state.items():
        np.testing.assert_array_equal(compact["result"].final_state[field], expected)
