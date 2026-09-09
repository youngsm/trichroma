"""Independent source, handoff, WLS and waveform checks for accelerated optics."""

from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="local CUDA GPU required")


@pytest.fixture(scope="module")
def pipeline():
    from chroma_lar.optical_calibration import OpticalCalibration
    from chroma_lar.optical_simulation import FastOpticalSimulation

    calibration = OpticalCalibration.load(
        Path(__file__).parents[1]
        / "benchmarks/optical_validation/full_detector_synthetic_calibration.json"
    )
    return FastOpticalSimulation(calibration)


@pytest.fixture(scope="module")
def simulation(pipeline):
    return pipeline.transport


@pytest.mark.parametrize("seed", [1729, 2**63 + 189])
def test_gpu_scintillation_matches_cpu_source(simulation, seed):
    n, first = 10003, 2**35 + 7
    s = simulation
    state = s._empty_state(n)
    s._source(state, n, (-1000, 0, 0), 0, seed, first, 17, 29.0)
    positions = state.pos.cpu().numpy()
    expected = s.calibration.source.photons(
        positions, seed=seed, photon_id_base=first, event_indices=17, times=29.0
    )
    for index, value, atol in (
        (1, expected.direction, 9.0e-7),
        (2, expected.polarization, 1.2e-6),
        (3, expected.times, 0.003),
        (10, expected.wavelengths, 2.0e-5),
    ):
        np.testing.assert_allclose(state[index].cpu().numpy(), value, rtol=3.0e-7, atol=atol)
    np.testing.assert_array_equal(state.photon_ids.cpu().numpy(), expected.global_photon_ids)
    np.testing.assert_array_equal(state.event_indices.cpu().numpy(), expected.event_indices)
    np.testing.assert_array_equal(state.flags.cpu().numpy(), expected.flags)
    assert np.max(abs((state.direction * state.polarization).sum(1).cpu().numpy())) < 5.0e-7


@pytest.mark.parametrize("varying", [False, True])
def test_bulk_epoch_commits_only_certified_collisions(simulation, varying):
    import triton
    from chroma.triton.optical_response import uniform
    from chroma.triton.physics import rayleigh_scatter
    from chroma.triton.transport import DeviceQueue
    from chroma_lar.spectral_kernels import bulk_epoch

    s, n, seed = simulation, 4097, 631
    state = s._empty_state(n)
    s._source(state, n, (-1000, 0, 0), 0, seed, 0, 0, 0)
    absorption, scattering = (
        s.model.properties.absorption_length.clone(),
        s.model.properties.scattering_length.clone(),
    )
    if varying:
        state.wavelengths.copy_(torch.linspace(120, 500, n, device="cuda"))
        absorption[s.lar_index].copy_(
            torch.linspace(1000, 100000, s.model.optics.wavelength_grid.count, device="cuda")
        )
        scattering[s.lar_index].copy_(
            torch.linspace(100, 2000, s.model.optics.wavelength_grid.count, device="cuda")
        )
    initial = [a.cpu().numpy().copy() for a in state]
    q, out, boundary = [DeviceQueue.allocate(n) for _ in range(3)]
    q.buffer.copy_(torch.arange(n, dtype=torch.int32, device="cuda"))
    q.count.fill_(n)
    g = s.model.optics.wavelength_grid
    bulk_epoch[(triton.cdiv(n, 128),)](
        *state,
        q.buffer,
        q.count,
        n,
        out.buffer,
        out.count,
        boundary.buffer,
        boundary.count,
        absorption,
        scattering,
        s.model.properties.group_velocity,
        seed,
        200,
        *map(float, s.query.safe_lower),
        *map(float, s.query.safe_upper),
        float(g.start),
        float(g.step),
        LAR=s.lar_index,
        NW=g.count,
        HISTORY=1,
        BLOCK=128,
        enable_fp_fusion=False
    )
    final = [a.cpu().numpy() for a in state]
    ids = np.arange(n, dtype=np.int64)
    alen = np.interp(initial[10], s.calibration.wavelengths, absorption[s.lar_index].cpu().numpy())
    slen = np.interp(initial[10], s.calibration.wavelengths, scattering[s.lar_index].cpu().numpy())
    da, ds = -alen * np.log(uniform(ids, seed, 0)), -slen * np.log(uniform(ids, seed, 1))
    direction, position = initial[1].astype(float), initial[0].astype(float)
    with np.errstate(divide="ignore"):
        distances = np.where(
            direction > 0,
            (s.query.safe_upper - position) / direction,
            np.where(direction < 0, (s.query.safe_lower - position) / direction, np.inf),
        )
    exit_distance = distances.min(1)
    safe = np.maximum(0, exit_distance - np.maximum(0.01, 2.0e-6 * abs(exit_distance)))
    collision = np.minimum(da, ds) < safe
    np.testing.assert_array_equal(final[9], collision.astype(np.int32))
    np.testing.assert_array_equal(
        np.sort(boundary.tensor().cpu().numpy()), np.flatnonzero(~collision)
    )
    np.testing.assert_array_equal(final[0][~collision], initial[0][~collision])
    np.testing.assert_array_equal(final[3][~collision], initial[3][~collision])
    scatter = collision & (ds < da)
    np.testing.assert_array_equal(np.sort(out.tensor().cpu().numpy()), np.flatnonzero(scatter))
    expected = rayleigh_scatter(
        initial[1][scatter],
        initial[2][scatter],
        *(uniform(ids, seed, i)[scatter] for i in range(2, 6))
    )
    np.testing.assert_allclose(final[1][scatter], expected[0], atol=2.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(final[2][scatter], expected[1], atol=2.0e-6, rtol=1.0e-6)
    np.testing.assert_allclose(
        final[0][collision],
        initial[0][collision] + np.minimum(da, ds)[collision, None] * initial[1][collision],
        rtol=3.0e-6,
        atol=0.0005,
    )
    velocity = np.interp(
        initial[10],
        s.calibration.wavelengths,
        s.model.properties.group_velocity[s.lar_index].cpu().numpy(),
    )
    np.testing.assert_allclose(
        final[3][collision],
        (initial[3] + np.minimum(da, ds) / velocity)[collision],
        rtol=3.0e-7,
        atol=0.003,
    )


def test_wls_boundary_retains_spectrum_time_and_material_side(simulation):
    import triton
    from chroma.triton.transport import DeviceQueue
    from chroma.triton.optical_response import uniform, TabulatedCDF
    from chroma_lar.spectral_kernels import boundary_step

    s, n, seed = simulation, 32768, 391
    state = s._empty_state(n)
    s._source(state, n, (-1000, 0, 0), 0, seed, 0, 0, 0)
    state.direction.copy_(torch.tensor([0.0, 0.0, 1.0], device="cuda"))
    state.polarization.copy_(torch.tensor([1.0, 0.0, 0.0], device="cuda"))
    state.times.zero_()
    state.wavelengths.fill_(128)
    sid = s.scene.tables.surface_names.index(s.calibration.manifest["roles"]["tpb"])
    glass = s.scene.tables.material_names.index(s.calibration.manifest["roles"]["glass"])
    tri = int(np.flatnonzero(s.scene.pmt.scene_surface_index == sid)[0])
    q, output = [DeviceQueue.allocate(n) for _ in range(2)]
    q.buffer.copy_(torch.arange(n, dtype=torch.int32, device="cuda"))
    q.count.fill_(n)

    def full(value, dtype=torch.int32):
        return torch.full((n,), value, dtype=dtype, device="cuda")

    normals = torch.tensor([0.0, 0.0, -1.0], device="cuda").repeat(n, 1)
    hit = (
        full(1.0, torch.float32),
        normals,
        full(s.lar_index),
        full(glass),
        full(sid),
        full(0),
        full(tri),
        full(-1),
    )
    arrays = list(s.model.properties)
    arrays[1] = torch.full_like(arrays[1], 1.0e30)
    arrays[2] = torch.full_like(arrays[2], 1.0e30)
    g = s.model.optics.wavelength_grid
    boundary_step[(triton.cdiv(n, 128),)](
        *state,
        q.buffer,
        q.count,
        n,
        output.buffer,
        output.count,
        *hit,
        s.query.scene_device["pmt_scene_material1_index"],
        *arrays,
        seed,
        200,
        float(g.start),
        float(g.step),
        NW=g.count,
        BLOCK=128,
        enable_fp_fusion=False
    )
    flags, wavelength, time, direction = (state[i].cpu().numpy() for i in (4, 10, 3, 1))
    ids = np.arange(n, dtype=np.int64)
    reemit_probability = np.interp(
        128, s.calibration.wavelengths, s.model.optics.surfaces.reemit[sid]
    )
    expected = uniform(ids, seed, 7) < reemit_probability
    np.testing.assert_array_equal((flags & 128) != 0, expected)
    cdf = TabulatedCDF(s.calibration.wavelengths, s.model.optics.surfaces.reemission_cdf[sid])
    np.testing.assert_allclose(
        wavelength[expected], cdf.sample(uniform(ids, seed, 8))[expected], atol=4.0e-5, rtol=1.0e-7
    )
    delay = s.model.surfaces[sid].reemission_time_cdf.sample(uniform(ids, seed, 9))
    velocity = np.interp(
        128, s.calibration.wavelengths, s.model.properties.group_velocity[s.lar_index].cpu().numpy()
    )
    np.testing.assert_allclose(
        time[expected], (1 / velocity + delay)[expected], atol=3.0e-5, rtol=2.0e-6
    )
    into_glass = uniform(ids, seed, 10) < s.model.surfaces[sid].reemission_to_material1
    np.testing.assert_array_equal(direction[expected, 2] > 0, into_glass[expected])
    np.testing.assert_array_equal(np.sort(output.tensor().cpu().numpy()), np.flatnonzero(expected))


@pytest.mark.parametrize("noise,adc_bits", [(0.0, None), (0.4, None), (0.4, 12)])
def test_gpu_waveforms_match_fractional_overlapping_cpu_pulses(noise, adc_bits):
    from chroma.triton.optical_response import Photoelectrons, digitize
    from chroma.triton.digitizer_kernels import digitize_gpu

    rng = np.random.default_rng(917)
    n = 2048
    times = rng.uniform(-40, 160, n)
    pe = Photoelectrons(
        times,
        rng.uniform(0.2, 1.8, n),
        rng.integers(0, 5, n),
        np.arange(n),
        rng.choice([11, 37], n),
        times,
    )
    options = dict(
        event_indices=[37, 99, 11],
        channel_count=5,
        start_ns=-20.123,
        sample_period_ns=0.3,
        sample_count=600,
        pulse_times_ns=[0.0, 0.7, 3.1, 6.2, 12.3],
        pulse_adc_per_pe=[0.0, 2.0, 1.3, 0.4, 0.0],
        baseline=100.0,
        noise_rms=noise,
        adc_bits=adc_bits,
        seed=991,
    )
    cpu, gpu = digitize(pe, **options), digitize_gpu(pe, **options)
    np.testing.assert_array_equal(cpu.event_indices, gpu.event_indices)
    np.testing.assert_array_equal(cpu.sample_times, gpu.sample_times)
    if adc_bits:
        np.testing.assert_array_equal(cpu.samples, gpu.samples)
    else:
        np.testing.assert_allclose(cpu.samples, gpu.samples, atol=2.0e-11, rtol=1.0e-13)


def test_geometry_query_has_no_transport_entrypoint(simulation):
    from chroma_lar.triton_scene.detector_query import DetectorBoundaryQuery

    assert isinstance(simulation.query, DetectorBoundaryQuery)
    assert not hasattr(simulation.query, "simulate")


def test_full_fast_pipeline_preserves_empty_event_and_wls(simulation, pipeline):
    result = pipeline.simulate_voxel(
        100000, event_id=11, event_indices=[11, 99], seed=1729, keep_final_states=True
    )
    state = result.transport.final_state
    assert not result.transport.diagnostics["escaped"]
    assert len(result.photoelectrons.times) > 200
    assert np.all((state["flags"][(state["flags"] & 4) != 0] & 128) != 0)
    assert np.all(result.transport.hits.wavelengths > 200)
    np.testing.assert_array_equal(
        result.waveforms.samples[1], simulation.calibration.digitizer["baseline"]
    )
    from chroma.triton.optical_response import digitize

    cpu = digitize(
        result.photoelectrons,
        event_indices=[11, 99],
        channel_count=162,
        seed=1729,
        **simulation.calibration.digitizer
    )
    np.testing.assert_array_equal(result.waveforms.samples, cpu.samples)


def test_gpu_deposition_counts_and_zero_light_event(simulation, pipeline):
    positions = [[-1000, 0, 0], [-900, 20, 0], [-1000, 0, 0]]
    energies = [0.03, 0.02, 0.0]
    options = dict(event_indices=[11, 37, 99], times=[3.0, 5.0, 7.0], seed=872)
    cpu = simulation.calibration.source.from_depositions(positions, energies, **options)
    result = pipeline.simulate_depositions(positions, energies, keep_final_states=True, **options)
    assert result.transport.photon_count == cpu.photon_count
    np.testing.assert_array_equal(result.transport.final_state["event_indices"], cpu.event_indices)
    np.testing.assert_array_equal(result.transport.final_state["photon_ids"], cpu.global_photon_ids)
    np.testing.assert_array_equal(result.waveforms.event_indices, [11, 37, 99])
    np.testing.assert_array_equal(
        result.waveforms.samples[2], simulation.calibration.digitizer["baseline"]
    )
    empty = pipeline.simulate_depositions([[-1000, 0, 0]], [0.0], event_indices=[99])
    assert empty.transport.photon_count == 0 and len(empty.photoelectrons.times) == 0
    np.testing.assert_array_equal(
        empty.waveforms.samples, simulation.calibration.digitizer["baseline"]
    )


def test_analytic_wire_calibration_replacements(simulation):
    c = simulation.calibration
    geometry = c.build_detector("detector_config_reflect_reflect3wires", analytic_wires=True)
    assert len(geometry.wireplanes) == 6
    for wire in geometry.wireplanes:
        for name in ("material_inner", "material_outer"):
            assert wire[name] is c.materials[wire[name].name]
        assert wire["surface"] is c.surfaces[wire["surface"].name]
    assert simulation.scene.wires.count == 6
    assert simulation.scene.instances.count == 162
    assert np.all(simulation.scene.boxes.reachable_face_mask)


def test_spectral_reachability_checks_every_wavelength(simulation):
    from copy import deepcopy
    from chroma_lar.config.detector_config_reflect_reflect3wires import get_config
    from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene

    c = deepcopy(simulation.calibration)
    barrier = c.surfaces[get_config()["cathode_surface"].name]
    for field in ("absorb", "detect", "reflect_diffuse"):
        barrier.set(field, 0.0)
    barrier.set("reflect_specular", [0.5, 1, 1], [120, 450, 500])
    with pytest.raises(ValueError, match="entire wavelength grid"):
        compile_reflect3wires_scene(calibration=c)


def test_source_outside_compiled_component_rejected(simulation):
    from chroma.event import Photons

    p = Photons([[10000.0, 0, 0]], [[1.0, 0, 0]], [[0.0, 1.0, 0]], [450.0])
    with pytest.raises(ValueError, match="negative-x"):
        simulation.simulate(p)


def test_discovered_regions_enable_positive_component_sources(simulation, pipeline):
    result = pipeline.simulate_voxel(10000, center=(1000, 0, 0), seed=763, keep_final_states=True)
    assert not result.transport.diagnostics["escaped"]
    assert len(result.transport.hits) > 10
    assert simulation.query.safe_lower[0] > 0
    assert np.all(np.isfinite(result.transport.final_state["pos"]))
    # A subsequent negative-side batch selects its own region.
    simulation.simulate_voxel(1000, center=(-1000, 0, 0), seed=762)
    assert simulation.query.safe_upper[0] < 0


def test_discovered_certificates_use_real_accelerator_bounds(simulation):
    from dataclasses import replace
    from chroma.triton.regions import compile_regions
    from chroma_lar.triton_scene.primitive_adapter import detector_primitives

    query = simulation.query
    regions = query.regions.regions
    assert len(regions) > 2
    assert any(r.bounds.contains([-1000, 0, 0]) for r in regions)
    assert any(r.bounds.contains([1000, 0, 0]) for r in regions)
    for region in regions:
        assert not any(
            region.bounds.overlaps(o.bounds) for o in query.primitives.bounded_obstacles()
        )
    # Detector names are not input to the certificate algorithm.
    renamed = replace(
        simulation.scene, boxes=replace(simulation.scene.boxes, kinds=("a", "b", "c"))
    )
    primitive_scene = detector_primitives(
        renamed,
        query.pmt_accelerator.host_bounds_min,
        query.pmt_accelerator.host_bounds_max,
        mesh_exterior_material=simulation.lar_index,
    )
    assert compile_regions(primitive_scene).fingerprint == query.regions.fingerprint


def test_lossless_detector_with_absorbing_steel_has_no_false_bulk_crossings(simulation):
    from copy import deepcopy
    from chroma_lar.spectral_backend import SpectralDetectorSimulation

    c = deepcopy(simulation.calibration)
    for name, material in c.materials.items():
        material.set("absorption_length", 0.0 if name == "steel" else 1.0e30)
        material.set("scattering_length", 1.0e30)
        material.set("refractive_index", 1.0)
    for name, surface in c.surfaces.items():
        surface.model = 0
        for field in ("absorb", "detect", "reflect_diffuse", "reflect_specular", "reemit"):
            surface.set(field, 0.0)
        surface.set(
            (
                "detect"
                if name in ("validation_tpb", "perfect_pmt_photocathode", "glossy_surface")
                else "reflect_diffuse"
            ),
            1.0,
        )
    c.surfaces["reflect00"].set("reflect_diffuse", 0.0)
    c.surfaces["reflect00"].set("absorb", 1.0)
    full = SpectralDetectorSimulation(c)
    result = full.simulate_voxel(
        10000, voxel_size=0.0, seed=11, max_steps=2048, keep_final_states=True
    )
    flags = result.final_state["flags"]
    assert np.count_nonzero(flags & (4 | 8)) == result.photon_count
    assert not np.any(flags & (1 | 2 | (1 << 30) | (1 << 31)))
    position = result.final_state["pos"][(flags & 8) != 0]
    assert np.all(np.any(np.isclose(abs(position), 3240.0, atol=0.002, rtol=0), axis=1))


def test_separated_pmt_bounds_preserve_complete_geometry_query(simulation):
    from dataclasses import replace
    from chroma_lar.triton_scene.instances import nearest_pmt_hit

    accelerator = simulation.query.pmt_accelerator
    assert accelerator.coarse_box_count == 2
    rng = np.random.default_rng(937)
    pos = rng.uniform(-3000, 3000, (20000, 3)).astype(np.float32)
    direction = rng.normal(size=pos.shape)
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    origin = torch.from_numpy(pos).cuda()
    rays = torch.from_numpy(direction.astype(np.float32)).cuda()
    separated = nearest_pmt_hit(accelerator, origin, rays, ray_tile=None)
    original = nearest_pmt_hit(
        replace(accelerator, coarse_bounds_min=None, coarse_bounds_max=None, coarse_box_count=1),
        origin,
        rays,
        ray_tile=None,
    )
    for name in ("distances", "triangle_ids", "instance_ids", "channel_ids", "world_normals"):
        np.testing.assert_array_equal(
            getattr(separated, name).cpu().numpy(), getattr(original, name).cpu().numpy()
        )
    channels = separated.channel_ids.cpu().numpy()
    assert np.any((channels >= 0) & (channels < 81)) and np.any(channels >= 81)
