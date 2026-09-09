"""Correctness tests for the algebraic Triton optical primitives."""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt

from chroma.triton.physics import (
    BULK_ABSORB,
    BULK_NONE,
    BULK_SCATTER,
    fresnel_coefficients,
    fresnel_step,
    median3,
    rayleigh_cosine_cdf,
    rayleigh_cosine_from_uniforms,
    rayleigh_scatter,
    reflect_specular,
    refract_direction,
    sample_bulk_collision,
)


def _random_photon_frames(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    helper = rng.normal(size=(n, 3))
    polarization = np.cross(direction, helper)
    polarization /= np.linalg.norm(polarization, axis=1, keepdims=True)
    return direction, polarization


def test_median3_is_elementwise_median_and_permutation_invariant():
    rng = np.random.default_rng(1293)
    values = rng.random((3, 10_000))
    expected = np.sort(values, axis=0)[1]
    npt.assert_array_equal(median3(*values), expected)
    npt.assert_array_equal(median3(values[2], values[0], values[1]), expected)


def test_rayleigh_cosine_has_exact_beta22_moments_and_cdf():
    rng = np.random.default_rng(99181)
    n = 500_000
    cosine = rayleigh_cosine_from_uniforms(
        rng.random(n), rng.random(n), rng.random(n)
    )

    # For p(c)=3/4(1-c^2): E[c]=0, E[c^2]=1/5, E[c^4]=3/35.
    assert abs(np.mean(cosine)) < 0.003
    assert abs(np.mean(cosine**2) - 0.2) < 0.0015
    assert abs(np.mean(cosine**4) - 3.0 / 35.0) < 0.001

    for point in (-0.8, -0.4, 0.0, 0.4, 0.8):
        empirical = np.mean(cosine <= point)
        assert abs(empirical - rayleigh_cosine_cdf(point)) < 0.0025


def test_rayleigh_direct_update_preserves_frame_and_sampled_cosine():
    rng = np.random.default_rng(3821)
    n = 100_000
    direction, polarization = _random_photon_frames(rng, n)
    uniforms = rng.random((4, n))
    expected_cosine = rayleigh_cosine_from_uniforms(*uniforms[:3])

    direction_new, polarization_new = rayleigh_scatter(
        direction, polarization, *uniforms
    )
    npt.assert_allclose(np.linalg.norm(direction_new, axis=1), 1.0, atol=2e-15)
    npt.assert_allclose(np.linalg.norm(polarization_new, axis=1), 1.0, atol=2e-15)
    npt.assert_allclose(
        np.sum(direction_new * polarization_new, axis=1), 0.0, atol=2e-15
    )
    npt.assert_allclose(
        np.sum(direction_new * polarization, axis=1), expected_cosine, atol=2e-15
    )


def test_rayleigh_uses_polarization_axis_after_oblique_specular_reflection():
    """Regression for Chroma's non-orthogonal post-reflection photon frame."""

    direction = np.array([0.6, 0.0, -0.8])
    polarization = np.array([0.0, 0.0, 1.0])
    reflected = reflect_specular(direction, np.array([0.0, 0.0, 1.0]))
    assert abs(np.dot(reflected, polarization)) > 0.5

    uniforms = (0.17, 0.81, 0.42, 0.63)
    out_direction, out_polarization = rayleigh_scatter(
        reflected, polarization, *uniforms
    )
    # Chroma's pick_new_direction depends only on its polarization axis.  An
    # arbitrary old direction must therefore give the identical sampled ray.
    other_direction = np.array([-0.2, 0.9, 0.1])
    other_out_direction, other_out_polarization = rayleigh_scatter(
        other_direction, polarization, *uniforms
    )
    npt.assert_allclose(out_direction, other_out_direction, atol=0.0, rtol=0.0)
    npt.assert_allclose(out_polarization, other_out_polarization, atol=0.0, rtol=0.0)
    npt.assert_allclose(np.linalg.norm(out_direction), 1.0, atol=2e-15)
    npt.assert_allclose(np.linalg.norm(out_polarization), 1.0, atol=2e-15)
    npt.assert_allclose(np.dot(out_direction, out_polarization), 0.0, atol=2e-15)
    npt.assert_allclose(
        np.dot(out_direction, polarization),
        rayleigh_cosine_from_uniforms(*uniforms[:3]),
        atol=2e-15,
    )


def test_total_hazard_bulk_collision_matches_competing_exponentials():
    rng = np.random.default_rng(87192)
    n = 500_000
    absorption_length = 1_500.0
    scattering_length = 950.0
    result = sample_bulk_collision(
        absorption_length,
        scattering_length,
        rng.random(n),
        rng.random(n),
    )

    total_rate = 1.0 / absorption_length + 1.0 / scattering_length
    expected_mean = 1.0 / total_rate
    expected_scatter_fraction = (1.0 / scattering_length) / total_rate
    assert abs(np.mean(result.distance) / expected_mean - 1.0) < 0.004
    assert abs(np.mean(result.process == BULK_SCATTER) - expected_scatter_fraction) < 0.002
    assert np.all((result.process == BULK_SCATTER) | (result.process == BULK_ABSORB))

    for multiples in (0.25, 0.5, 1.0, 2.0):
        distance = multiples * expected_mean
        empirical = np.mean(result.distance <= distance)
        expected = 1.0 - np.exp(-total_rate * distance)
        assert abs(empirical - expected) < 0.002


def test_zero_hazard_bulk_collision_is_none_at_infinity():
    result = sample_bulk_collision(
        np.array([np.inf, -1.0, np.inf]),
        np.array([np.inf, np.inf, 10.0]),
        np.array([0.5, 0.5, 0.5]),
        np.array([0.5, 0.5, 0.5]),
    )
    assert np.isinf(result.distance[0]) and result.process[0] == BULK_NONE
    assert np.isinf(result.distance[1]) and result.process[1] == BULK_NONE
    assert np.isfinite(result.distance[2]) and result.process[2] == BULK_SCATTER


def test_algebraic_specular_reflection_invariants():
    rng = np.random.default_rng(442)
    direction = rng.normal(size=(20_000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    normal = rng.normal(size=(20_000, 3))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    reflected = reflect_specular(direction, normal)
    npt.assert_allclose(np.linalg.norm(reflected, axis=1), 1.0, atol=2e-15)
    npt.assert_allclose(
        np.sum(reflected * normal, axis=1),
        -np.sum(direction * normal, axis=1),
        atol=2e-15,
    )
    npt.assert_allclose(reflect_specular(reflected, normal), direction, atol=3e-15)


def test_angle_free_fresnel_special_cases_and_angle_formula():
    # Normal incidence.
    n1, n2 = 1.0, 1.5
    normal_r = ((n1 - n2) / (n1 + n2)) ** 2
    coeff = fresnel_coefficients(1.0, n1, n2)
    npt.assert_allclose(coeff.reflect_s, normal_r, rtol=1e-14)
    npt.assert_allclose(coeff.reflect_p, normal_r, rtol=1e-14)
    assert not bool(coeff.total_internal_reflection)

    # Brewster angle has zero p reflection.
    theta_b = math.atan(n2 / n1)
    coeff = fresnel_coefficients(math.cos(theta_b), n1, n2)
    assert coeff.reflect_p < 1.0e-28

    # Compare to the angle-based formulas Chroma currently evaluates.
    theta_i = np.linspace(0.001, math.radians(80.0), 2_000)
    theta_t = np.arcsin(np.sin(theta_i) * n1 / n2)
    expected_s = (
        -np.sin(theta_i - theta_t) / np.sin(theta_i + theta_t)
    ) ** 2
    expected_p = (np.tan(theta_i - theta_t) / np.tan(theta_i + theta_t)) ** 2
    coeff = fresnel_coefficients(np.cos(theta_i), n1, n2)
    npt.assert_allclose(coeff.reflect_s, expected_s, rtol=2e-12, atol=1e-15)
    npt.assert_allclose(coeff.reflect_p, expected_p, rtol=2e-12, atol=1e-15)


def test_vector_snell_law_and_total_internal_reflection():
    theta = np.linspace(0.0, math.radians(75.0), 1_000)
    direction = np.stack((np.sin(theta), np.zeros_like(theta), -np.cos(theta)), axis=1)
    normal = np.zeros_like(direction)
    normal[:, 2] = 1.0
    refracted, tir = refract_direction(direction, normal, 1.0, 1.5)
    assert not np.any(tir)
    npt.assert_allclose(np.linalg.norm(refracted, axis=1), 1.0, atol=2e-15)
    npt.assert_allclose(1.5 * refracted[:, 0], np.sin(theta), atol=2e-15)

    theta_tir = math.radians(50.0)  # above asin(1/1.5)
    direction_tir = np.array([math.sin(theta_tir), 0.0, -math.cos(theta_tir)])
    out, tir = refract_direction(direction_tir, np.array([0.0, 0.0, 1.0]), 1.5, 1.0)
    assert bool(tir)
    npt.assert_allclose(out, reflect_specular(direction_tir, [0.0, 0.0, 1.0]))


def test_fresnel_step_preserves_direction_and_polarization_frame():
    rng = np.random.default_rng(8921)
    n = 100_000
    theta = rng.uniform(0.0, math.radians(75.0), n)
    direction = np.stack((np.sin(theta), np.zeros(n), -np.cos(theta)), axis=1)
    normal = np.zeros_like(direction)
    normal[:, 2] = 1.0
    # Uniform linear-polarization angle in the transverse plane.
    s_axis = np.tile([0.0, -1.0, 0.0], (n, 1))
    p_axis = np.cross(s_axis, direction)
    alpha = rng.uniform(0.0, 2.0 * np.pi, n)
    polarization = np.cos(alpha)[:, None] * s_axis + np.sin(alpha)[:, None] * p_axis

    step = fresnel_step(
        direction,
        polarization,
        normal,
        1.0,
        1.5,
        rng.random(n),
        rng.random(n),
    )
    npt.assert_allclose(np.linalg.norm(step.direction, axis=1), 1.0, atol=2e-15)
    npt.assert_allclose(np.linalg.norm(step.polarization, axis=1), 1.0, atol=2e-15)
    npt.assert_allclose(
        np.sum(step.direction * step.polarization, axis=1), 0.0, atol=2e-15
    )
    assert np.all((step.selected_reflectance >= 0.0) & (step.selected_reflectance <= 1.0))


def test_chroma_cross_preserves_positive_zero_on_exact_cancellation():
    import pytest

    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    import triton.language as tl

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from chroma.triton.physics_kernels import cross_chroma_fast

    globals().update(tl=tl, cross_chroma_fast=cross_chroma_fast)

    @triton.jit
    def probe(output):
        one = tl.full((1,), 1.0, tl.float32)
        x, y, z = cross_chroma_fast(
            one, one, 2.0 * one, -3.0 * one, -2.0 * one, -4.0 * one
        )
        lane = tl.arange(0, 1)
        tl.store(output + lane, x)
        tl.store(output + 1 + lane, y)
        tl.store(output + 2 + lane, z)

    output = torch.empty(3, dtype=torch.float32, device="cuda")
    probe[(1,)](output)
    words = output.cpu().numpy().view(np.uint32)
    npt.assert_array_equal(
        words, np.asarray((0x00000000, 0xC0000000, 0x3F800000), np.uint32)
    )


def test_triton_photon_source_is_deterministic_tile_invariant_and_isotropic():
    import pytest

    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from chroma.triton.physics_kernels import generate_photon_bomb

    n = 262_144
    kwargs = dict(
        seed=0x12345678,
        center=(-1000.0, 15.0, -7.0),
        voxel_size=30.0,
        wavelength=(440.0, 460.0),
    )
    whole = generate_photon_bomb(n, **kwargs)
    repeat = generate_photon_bomb(n, **kwargs)
    for left, right in zip(whole.tensors, repeat.tensors):
        assert torch.equal(left, right)

    split = n // 2
    first = generate_photon_bomb(split, photon_id_base=0, **kwargs)
    second = generate_photon_bomb(n - split, photon_id_base=split, **kwargs)
    for full, left, right in zip(whole.tensors, first.tensors, second.tensors):
        assert torch.equal(full, torch.cat((left, right)))

    direction = whole.direction()
    polarization = whole.polarization()
    position = whole.position()
    torch.testing.assert_close(
        torch.linalg.vector_norm(direction, dim=1),
        torch.ones(n, device="cuda"),
        atol=2.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(polarization, dim=1),
        torch.ones(n, device="cuda"),
        atol=2.0e-6,
        rtol=0.0,
    )
    assert torch.max(torch.abs(torch.sum(direction * polarization, dim=1))) < 2.0e-6
    assert torch.max(torch.abs(torch.mean(direction, dim=0))) < 0.005
    assert torch.all(position >= torch.tensor([-1015.0, 0.0, -22.0], device="cuda"))
    assert torch.all(position <= torch.tensor([-985.0, 30.0, 8.0], device="cuda"))
    assert torch.all((whole.wavelength >= 440.0) & (whole.wavelength <= 460.0))


def test_triton_optical_primitives_match_cpu_references():
    import pytest

    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from chroma.triton.physics_kernels import (
        bulk_collision_kernel,
        fresnel_step_kernel,
        rayleigh_scatter_kernel,
    )

    rng = np.random.default_rng(67291)
    n = 65_537
    block = 256
    grid = (triton.cdiv(n, block),)

    # Rayleigh frame update.
    direction, polarization = _random_photon_frames(rng, n)
    uniforms = rng.random((4, n))
    expected_direction, expected_polarization = rayleigh_scatter(
        direction.astype(np.float32), polarization.astype(np.float32), *uniforms.astype(np.float32)
    )
    rayleigh_inputs = [
        torch.as_tensor(x, dtype=torch.float32, device="cuda")
        for x in (
            *direction.astype(np.float32).T,
            *polarization.astype(np.float32).T,
            *uniforms.astype(np.float32),
        )
    ]
    rayleigh_outputs = [torch.empty(n, dtype=torch.float32, device="cuda") for _ in range(6)]
    rayleigh_scatter_kernel[grid](
        *rayleigh_inputs, *rayleigh_outputs, n, BLOCK_SIZE=block
    )
    actual_direction = torch.stack(rayleigh_outputs[:3], dim=1).cpu().numpy()
    actual_polarization = torch.stack(rayleigh_outputs[3:], dim=1).cpu().numpy()
    npt.assert_allclose(actual_direction, expected_direction, rtol=2e-5, atol=2e-6)
    npt.assert_allclose(actual_polarization, expected_polarization, rtol=2e-5, atol=2e-6)

    # Competing bulk hazards.
    absorption = rng.uniform(500.0, 5_000.0, n).astype(np.float32)
    scattering = rng.uniform(250.0, 2_000.0, n).astype(np.float32)
    bulk_uniforms = rng.random((2, n)).astype(np.float32)
    expected_bulk = sample_bulk_collision(
        absorption, scattering, bulk_uniforms[0], bulk_uniforms[1]
    )
    bulk_inputs = [
        torch.as_tensor(x, dtype=torch.float32, device="cuda")
        for x in (absorption, scattering, *bulk_uniforms)
    ]
    distance_out = torch.empty(n, dtype=torch.float32, device="cuda")
    process_out = torch.empty(n, dtype=torch.int32, device="cuda")
    bulk_collision_kernel[grid](
        *bulk_inputs, distance_out, process_out, n, BLOCK_SIZE=block
    )
    npt.assert_allclose(
        distance_out.cpu().numpy(), expected_bulk.distance, rtol=3e-6, atol=2e-4
    )
    npt.assert_array_equal(process_out.cpu().numpy(), expected_bulk.process)

    # Fresnel sampling over a range of incident angles and polarizations.
    theta = rng.uniform(0.0, math.radians(80.0), n).astype(np.float32)
    direction_f = np.stack(
        (np.sin(theta), np.zeros(n, dtype=np.float32), -np.cos(theta)), axis=1
    )
    normal = np.zeros_like(direction_f)
    normal[:, 2] = 1.0
    s_axis = np.tile(np.array([0.0, -1.0, 0.0], np.float32), (n, 1))
    p_axis = np.cross(s_axis, direction_f)
    alpha = rng.uniform(0.0, 2.0 * np.pi, n).astype(np.float32)
    polarization_f = np.cos(alpha)[:, None] * s_axis + np.sin(alpha)[:, None] * p_axis
    n1 = rng.uniform(1.0, 1.4, n).astype(np.float32)
    n2 = rng.uniform(1.35, 1.7, n).astype(np.float32)
    fresnel_uniforms = rng.random((2, n)).astype(np.float32)
    expected_f = fresnel_step(
        direction_f,
        polarization_f,
        normal,
        n1,
        n2,
        fresnel_uniforms[0],
        fresnel_uniforms[1],
    )
    fresnel_inputs_np = (
        *direction_f.T,
        *polarization_f.T,
        *normal.T,
        n1,
        n2,
        *fresnel_uniforms,
    )
    fresnel_inputs = [
        torch.as_tensor(x, dtype=torch.float32, device="cuda") for x in fresnel_inputs_np
    ]
    vector_outputs = [torch.empty(n, dtype=torch.float32, device="cuda") for _ in range(6)]
    reflected_out = torch.empty(n, dtype=torch.bool, device="cuda")
    reflectance_out = torch.empty(n, dtype=torch.float32, device="cuda")
    s_fraction_out = torch.empty(n, dtype=torch.float32, device="cuda")
    tir_out = torch.empty(n, dtype=torch.bool, device="cuda")
    fresnel_step_kernel[grid](
        *fresnel_inputs,
        *vector_outputs,
        reflected_out,
        reflectance_out,
        s_fraction_out,
        tir_out,
        n,
        BLOCK_SIZE=block,
    )
    actual_f_direction = torch.stack(vector_outputs[:3], dim=1).cpu().numpy()
    actual_f_polarization = torch.stack(vector_outputs[3:], dim=1).cpu().numpy()
    npt.assert_allclose(actual_f_direction, expected_f.direction, rtol=3e-5, atol=3e-6)
    npt.assert_allclose(actual_f_polarization, expected_f.polarization, rtol=3e-5, atol=3e-6)
    npt.assert_array_equal(reflected_out.cpu().numpy(), expected_f.reflected)
    npt.assert_allclose(
        reflectance_out.cpu().numpy(), expected_f.selected_reflectance, rtol=3e-5, atol=3e-7
    )
    npt.assert_allclose(
        s_fraction_out.cpu().numpy(), expected_f.s_fraction, rtol=3e-5, atol=3e-7
    )
    npt.assert_array_equal(tir_out.cpu().numpy(), expected_f.total_internal_reflection)
