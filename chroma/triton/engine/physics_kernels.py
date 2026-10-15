"""Triton kernels for source generation and algebraic optical primitives.

The small ``@triton.jit`` functions are the public integration surface for the
transport backend.  The standalone kernels and host wrappers exist for unit
tests, profiling, and source generation before the full transport is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ``from __future__ import annotations`` would stringify annotations, so use
# explicit constexpr objects for globals consumed by Triton 3.1's frontend.
TWO_PI = tl.constexpr(6.2831853071795864769)
SOURCE_DOMAIN_0 = tl.constexpr(0x243F6A88)
SOURCE_DOMAIN_1 = tl.constexpr(0x13198A2E)

BULK_NONE = tl.constexpr(0)
BULK_ABSORB = tl.constexpr(1)
BULK_SCATTER = tl.constexpr(2)


@triton.jit
def median3(u0, u1, u2):
    """Elementwise median of three values."""

    return tl.maximum(tl.minimum(u0, u1), tl.minimum(tl.maximum(u0, u1), u2))


@triton.jit
def rayleigh_cosine(u0, u1, u2):
    """Exact ``3/4 * (1-c*c)`` Rayleigh cosine via ``Beta(2, 2)``."""

    return 2.0 * median3(u0, u1, u2) - 1.0


@triton.jit
def rayleigh_scatter(
    direction_x,
    direction_y,
    direction_z,
    polarization_x,
    polarization_y,
    polarization_z,
    u0,
    u1,
    u2,
    u_phi,
):
    """Scatter around normalized polarization without inverse trigonometry.

    Chroma's specular surface path does not keep direction perpendicular to
    polarization.  Consequently the old direction must *not* define the
    azimuthal tangent.  It remains in this signature for integration symmetry,
    while a robust arbitrary tangent basis is built solely from polarization.
    """

    polarization_norm2 = (
        polarization_x * polarization_x
        + polarization_y * polarization_y
        + polarization_z * polarization_z
    )
    inv_polarization = tl.rsqrt(tl.maximum(polarization_norm2, 1.0e-20))
    px = polarization_x * inv_polarization
    py = polarization_y * inv_polarization
    pz = polarization_z * inv_polarization

    # b = normalize(reference x p), choosing z except near its poles.
    use_z = tl.abs(pz) < 0.9
    bx_raw = tl.where(use_z, -py, 0.0)
    by_raw = tl.where(use_z, px, -pz)
    bz_raw = tl.where(use_z, 0.0, py)
    inv_b = tl.rsqrt(
        tl.maximum(bx_raw * bx_raw + by_raw * by_raw + bz_raw * bz_raw, 1.0e-20)
    )
    bx = bx_raw * inv_b
    by = by_raw * inv_b
    bz = bz_raw * inv_b
    qx = py * bz - pz * by
    qy = pz * bx - px * bz
    qz = px * by - py * bx

    phi = TWO_PI * u_phi
    cos_phi = tl.cos(phi)
    sin_phi = tl.sin(phi)
    tx = cos_phi * bx + sin_phi * qx
    ty = cos_phi * by + sin_phi * qy
    tz = cos_phi * bz + sin_phi * qz

    cosine = rayleigh_cosine(u0, u1, u2)
    sine = tl.sqrt(tl.maximum(0.0, 1.0 - cosine * cosine))
    direction_new_x = cosine * px + sine * tx
    direction_new_y = cosine * py + sine * ty
    direction_new_z = cosine * pz + sine * tz
    polarization_new_x = sine * px - cosine * tx
    polarization_new_y = sine * py - cosine * ty
    polarization_new_z = sine * pz - cosine * tz
    return (
        direction_new_x,
        direction_new_y,
        direction_new_z,
        polarization_new_x,
        polarization_new_y,
        polarization_new_z,
    )


@triton.jit
def sample_bulk_collision(
    absorption_length, scattering_length, u_distance, u_process
):
    """Sample competing exponential hazards with one logarithm.

    Returns ``(distance, process)`` where process is ``BULK_NONE``,
    ``BULK_ABSORB``, or ``BULK_SCATTER``.
    """

    valid_a = absorption_length > 0.0
    valid_s = scattering_length > 0.0
    rate_a = tl.where(valid_a, 1.0 / tl.maximum(absorption_length, 1.0e-30), 0.0)
    rate_s = tl.where(valid_s, 1.0 / tl.maximum(scattering_length, 1.0e-30), 0.0)
    rate = rate_a + rate_s
    distance = tl.where(
        rate > 0.0,
        -tl.log(tl.maximum(u_distance, 1.0e-30)) / tl.maximum(rate, 1.0e-30),
        float("inf"),
    )
    scatter_probability = tl.where(rate > 0.0, rate_s / tl.maximum(rate, 1.0e-30), 0.0)
    process = tl.where(
        rate <= 0.0,
        BULK_NONE,
        tl.where(u_process < scatter_probability, BULK_SCATTER, BULK_ABSORB),
    )
    return distance, process


@triton.jit
def exponential_distance_chroma_fast(length, uniform):
    """Reproduce Chroma's CUDA fast-math ``-length * logf(uniform)``.

    This is deliberately PTX, rather than an algebraically equivalent Triton
    expression.  CUDA 12.4 lowers the expression in ``photon.h`` to an
    approximate base-2 logarithm followed by two separately rounded
    multiplications and a negation.  Keeping the sequence opaque also prevents
    a compile-time material length from being folded into the ln(2) constant.
    """

    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 log2_value;
            .reg .f32 log_value;
            lg2.approx.ftz.f32 log2_value, $1;
            mul.ftz.f32 log_value, log2_value, 0f3F317218;
            mul.ftz.f32 $0, $2, log_value;
            neg.ftz.f32 $0, $0;
        }
        """,
        constraints="=f,f,f",
        args=[uniform, length],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def advance_position_chroma_fast(position, distance, direction):
    """Match CUDA's fused ``position += distance * direction`` update."""

    return tl.inline_asm_elementwise(
        asm="fma.rn.ftz.f32 $0, $2, $3, $1;",
        constraints="=f,f,f,f",
        args=[position, distance, direction],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def advance_time_chroma_fast(photon_time, distance, refractive_index):
    """Match Chroma's two fast divisions and rounded time addition."""

    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 velocity;
            .reg .f32 delta_time;
            div.approx.ftz.f32 velocity, 0f4395E56F, $3;
            div.approx.ftz.f32 delta_time, $2, velocity;
            add.ftz.f32 $0, delta_time, $1;
        }
        """,
        constraints="=f,f,f,f",
        args=[photon_time, distance, refractive_index],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def fresnel_incident_cosine_chroma_fast(
    direction_x,
    direction_y,
    direction_z,
    normal_x,
    normal_y,
    normal_z,
):
    """Match the non-contracted dot tree in CUDA ``propagate_at_boundary``."""

    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 product_x;
            .reg .f32 product_y;
            .reg .f32 partial;
            .reg .f32 product_z;
            mul.ftz.f32 product_x, $1, $4;
            mul.ftz.f32 product_y, $2, $5;
            neg.ftz.f32 partial, product_y;
            sub.ftz.f32 partial, partial, product_x;
            mul.ftz.f32 product_z, $3, $6;
            sub.ftz.f32 $0, partial, product_z;
        }
        """,
        constraints="=f,f,f,f,f,f,f",
        args=[
            direction_x,
            direction_y,
            direction_z,
            normal_x,
            normal_y,
            normal_z,
        ],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def cross_chroma_fast(
    left_x,
    left_y,
    left_z,
    right_x,
    right_y,
    right_z,
):
    """Match CUDA's three separately rounded cross-product components.

    Keeping the multiply/subtract pairs opaque is observable when two products
    cancel exactly: CUDA produces ``+0.0``, whereas contraction to a negated
    FMA can produce ``-0.0``.  The sign is part of compatibility state when
    the incidence axis becomes the outgoing s-polarization.
    """

    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 first;
            .reg .f32 second;
            mul.ftz.f32 first, $4, $8;
            mul.ftz.f32 second, $5, $7;
            sub.ftz.f32 $0, first, second;
            mul.ftz.f32 first, $5, $6;
            mul.ftz.f32 second, $3, $8;
            sub.ftz.f32 $1, first, second;
            mul.ftz.f32 first, $3, $7;
            mul.ftz.f32 second, $4, $6;
            sub.ftz.f32 $2, first, second;
        }
        """,
        constraints="=f,=f,=f,f,f,f,f,f,f",
        args=[left_x, left_y, left_z, right_x, right_y, right_z],
        dtype=(tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def reflect_specular(direction_x, direction_y, direction_z, normal_x, normal_y, normal_z):
    """Algebraic specular reflection ``d - 2*dot(d,n)*n``.

    This deliberately avoids Chroma's equivalent acos/Rodrigues construction.
    Under CUDA fast-math, that longer path can round a wire reflection just
    inside its cylinder and cause a nonphysical zero-distance steel absorb.
    The vector identity is both the physically exact operation and faster.
    """

    dot_dn = direction_x * normal_x + direction_y * normal_y + direction_z * normal_z
    scale = 2.0 * dot_dn
    return (
        direction_x - scale * normal_x,
        direction_y - scale * normal_y,
        direction_z - scale * normal_z,
    )


@triton.jit
def acos_chroma_fast(value):
    """Reproduce CUDA 12.4 ``__acosf`` as emitted by ``--use_fast_math``.

    Chroma compiles ``acosf`` with ``--use_fast_math``.  NVCC therefore
    inlines the approximate device intrinsic instead of calling
    ``__nv_acosf``.  The sequence and coefficient words below are transcribed
    from the PTX generated for ``chroma/cuda/specular_probe.cu``.  Keeping
    this as a separately probed compatibility primitive makes the historical
    numerical behavior explicit; production reflection does not use it.
    """

    threshold = tl.full(value.shape, 0x3F0F5C29, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_0 = tl.full(value.shape, 0x3C8B1ABB, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_1 = tl.full(value.shape, 0x3D10ECEF, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_2 = tl.full(value.shape, 0x3CFC028C, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_3 = tl.full(value.shape, 0x3D372139, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_4 = tl.full(value.shape, 0x3D9993DB, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_5 = tl.full(value.shape, 0x3E2AAAC6, tl.uint32).to(
        tl.float32, bitcast=True
    )
    split_pi_over_two_hi = tl.full(value.shape, 0x3FD774EB, tl.uint32).to(
        tl.float32, bitcast=True
    )
    split_pi_over_two_lo = tl.full(value.shape, 0x3F6EE581, tl.uint32).to(
        tl.float32, bitcast=True
    )

    absolute = tl.abs(value)
    half_complement = tl.fma(0.5, -absolute, 0.5)
    inverse_root = tl.rsqrt(half_complement)
    root_seed = half_complement * inverse_root
    half_inverse_root = inverse_root * 0.5
    root_correction = tl.fma(-root_seed, half_inverse_root, 0.5)
    root = tl.fma(root_seed, root_correction, root_seed)
    root = tl.where(absolute == 1.0, 0.0, root)

    large = absolute > threshold
    polynomial_argument_magnitude = tl.where(large, root, absolute)
    value_words = value.to(tl.uint32, bitcast=True)
    magnitude_words = polynomial_argument_magnitude.to(
        tl.uint32, bitcast=True
    )
    sign_mask = tl.full(value.shape, 0x80000000, tl.uint32)
    signed_argument = (
        (value_words & sign_mask) | magnitude_words
    ).to(tl.float32, bitcast=True)

    squared = signed_argument * signed_argument
    polynomial = tl.fma(coefficient_1, squared, coefficient_0)
    polynomial = tl.fma(polynomial, squared, coefficient_2)
    polynomial = tl.fma(polynomial, squared, coefficient_3)
    polynomial = tl.fma(polynomial, squared, coefficient_4)
    polynomial = tl.fma(polynomial, squared, coefficient_5)
    polynomial_times_square = polynomial * squared
    approximation = tl.fma(
        polynomial_times_square, signed_argument, signed_argument
    )
    signed_approximation = tl.where(large, approximation, -approximation)
    pi_over_two_adjusted = tl.fma(
        split_pi_over_two_lo,
        split_pi_over_two_hi,
        signed_approximation,
    )
    positive_large = value > threshold
    selected = tl.where(positive_large, approximation, pi_over_two_adjusted)
    doubled = selected + selected
    return tl.where(large, doubled, selected)


@triton.jit
def asin_chroma_fast(value):
    """Reproduce CUDA 12.4 ``__asinf`` under Chroma's fast-math flags."""

    threshold = tl.full(value.shape, 0x3F0F5C29, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_0 = tl.full(value.shape, 0x3C99CA97, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_1 = tl.full(value.shape, 0x3D4DD2F7, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_2 = tl.full(value.shape, 0x3D3F90E8, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_3 = tl.full(value.shape, 0x3D993CCF, tl.uint32).to(
        tl.float32, bitcast=True
    )
    coefficient_4 = tl.full(value.shape, 0x3E2AAC04, tl.uint32).to(
        tl.float32, bitcast=True
    )
    split_pi_over_two_hi = tl.full(value.shape, 0x3FD774EB, tl.uint32).to(
        tl.float32, bitcast=True
    )
    split_pi_over_two_lo = tl.full(value.shape, 0x3F6EE581, tl.uint32).to(
        tl.float32, bitcast=True
    )

    absolute = tl.abs(value)
    half_complement = tl.fma(0.5, -absolute, 0.5)
    inverse_root = tl.rsqrt(half_complement)
    root_seed = half_complement * inverse_root
    half_inverse_root = inverse_root * 0.5
    root_correction = tl.fma(-root_seed, half_inverse_root, 0.5)
    root = tl.fma(root_seed, root_correction, root_seed)
    root = tl.where(absolute == 1.0, 0.0, root)

    large = absolute > threshold
    argument = tl.where(large, root, absolute)
    squared = argument * argument
    polynomial = tl.fma(coefficient_1, squared, coefficient_0)
    polynomial = tl.fma(polynomial, squared, coefficient_2)
    polynomial = tl.fma(polynomial, squared, coefficient_3)
    polynomial = tl.fma(polynomial, squared, coefficient_4)
    polynomial_times_square = squared * polynomial
    approximation = tl.fma(
        polynomial_times_square, argument, argument
    )
    large_approximation = tl.fma(
        split_pi_over_two_lo,
        split_pi_over_two_hi,
        approximation * -2.0,
    )
    magnitude = tl.where(large, large_approximation, approximation)
    value_words = value.to(tl.uint32, bitcast=True)
    magnitude_words = magnitude.to(tl.uint32, bitcast=True)
    sign_mask = tl.full(value.shape, 0x80000000, tl.uint32)
    signed = (
        (value_words & sign_mask) | magnitude_words
    ).to(tl.float32, bitcast=True)
    return tl.where(magnitude <= float("inf"), signed, magnitude)


@triton.jit
def rotate_chroma_fast(
    vector_x,
    vector_y,
    vector_z,
    angle,
    axis_x,
    axis_y,
    axis_z,
):
    """Literal CUDA fast-math lowering of ``rotate.h::rotate``."""

    cosine = libdevice.fast_cosf(angle)
    sine = libdevice.fast_sinf(angle)
    projection = tl.fma(
        axis_z,
        vector_z,
        tl.fma(axis_x, vector_x, axis_y * vector_y),
    )
    projected_x = axis_x * projection
    projected_y = axis_y * projection
    projected_z = axis_z * projection
    one_minus_cosine = 1.0 - cosine
    blend_x = one_minus_cosine * projected_x
    blend_y = one_minus_cosine * projected_y
    blend_z = one_minus_cosine * projected_z
    blend_x = tl.fma(cosine, vector_x, blend_x)
    blend_y = tl.fma(cosine, vector_y, blend_y)
    blend_z = tl.fma(cosine, vector_z, blend_z)
    cross_x = vector_y * axis_z - vector_z * axis_y
    cross_y = vector_z * axis_x - vector_x * axis_z
    cross_z = vector_x * axis_y - vector_y * axis_x
    return (
        tl.fma(sine, cross_x, blend_x),
        tl.fma(sine, cross_y, blend_y),
        tl.fma(sine, cross_z, blend_z),
    )


@triton.jit
def reflect_specular_chroma(
    direction_x,
    direction_y,
    direction_z,
    normal_x,
    normal_y,
    normal_z,
):
    """Literal Chroma ``acosf`` plus fast ``sinf/cosf`` Rodrigues path.

    This is a validation primitive, not the preferred production operation.
    It intentionally retains Chroma's unguarded incidence-axis normalization
    and CUDA fast-math trigonometric functions so raw-word probes can separate
    legacy numerical behavior from the corrected algebraic reflection.
    """

    # NVCC lowers ``dot(normal, -direction)`` under --use_fast_math to a
    # left-to-right FFMA chain: x is rounded once, then y and z are fused.
    # Triton's ordinary three-term expression uses a different association.
    # Spell out the CUDA tree so compatibility mode is word-for-word stable.
    incident_dot = tl.fma(
        normal_z,
        -direction_z,
        tl.fma(
            normal_y,
            -direction_y,
            normal_x * -direction_x,
        ),
    )
    incident_cosine = tl.maximum(-1.0, tl.minimum(1.0, incident_dot))
    incident_angle = acos_chroma_fast(incident_cosine)
    axis_x, axis_y, axis_z = cross_chroma_fast(
        direction_x,
        direction_y,
        direction_z,
        normal_x,
        normal_y,
        normal_z,
    )
    axis_length = tl.sqrt(
        axis_x * axis_x + axis_y * axis_y + axis_z * axis_z
    )
    # Preserve linalg.h's three component-wise ``operator/=`` operations.
    # Computing one reciprocal and multiplying is mathematically equivalent
    # but changes raw float32 words under CUDA's approximate division mode.
    axis_x /= axis_length
    axis_y /= axis_length
    axis_z /= axis_length
    cosine = libdevice.fast_cosf(incident_angle)
    sine = libdevice.fast_sinf(incident_angle)
    projection = (
        normal_x * axis_x + normal_y * axis_y + normal_z * axis_z
    )
    one_minus_cosine = 1.0 - cosine
    cross_x = normal_y * axis_z - normal_z * axis_y
    cross_y = normal_z * axis_x - normal_x * axis_z
    cross_z = normal_x * axis_y - normal_y * axis_x
    return (
        normal_x * cosine + axis_x * projection * one_minus_cosine
        + cross_x * sine,
        normal_y * cosine + axis_y * projection * one_minus_cosine
        + cross_y * sine,
        normal_z * cosine + axis_z * projection * one_minus_cosine
        + cross_z * sine,
    )


@triton.jit
def fresnel_step_chroma(
    direction_x,
    direction_y,
    direction_z,
    polarization_x,
    polarization_y,
    polarization_z,
    normal_x,
    normal_y,
    normal_z,
    refractive_index1,
    refractive_index2,
    u_polarization,
    u_reflect,
):
    """Literal ``photon.h::propagate_at_boundary`` compatibility path."""

    # Match the non-fused multiply/subtract tree emitted for
    # get_theta(surface_normal, -direction).
    incident_cosine = fresnel_incident_cosine_chroma_fast(
        direction_x,
        direction_y,
        direction_z,
        normal_x,
        normal_y,
        normal_z,
    )
    incident_cosine = tl.maximum(-1.0, tl.minimum(1.0, incident_cosine))
    incident_angle = acos_chroma_fast(incident_cosine)
    refracted_argument = (
        libdevice.fast_sinf(incident_angle) * refractive_index1
    ) / refractive_index2
    refracted_angle = asin_chroma_fast(refracted_argument)

    axis_x, axis_y, axis_z = cross_chroma_fast(
        direction_x,
        direction_y,
        direction_z,
        normal_x,
        normal_y,
        normal_z,
    )
    axis_norm2 = tl.fma(
        axis_z, axis_z, tl.fma(axis_x, axis_x, axis_y * axis_y)
    )
    axis_length = tl.sqrt(axis_norm2)
    normal_incidence = axis_length < 1.0e-6
    normalized_axis_x = axis_x / axis_length
    normalized_axis_y = axis_y / axis_length
    normalized_axis_z = axis_z / axis_length
    axis_x = tl.where(normal_incidence, polarization_x, normalized_axis_x)
    axis_y = tl.where(normal_incidence, polarization_y, normalized_axis_y)
    axis_z = tl.where(normal_incidence, polarization_z, normalized_axis_z)

    normal_coefficient = tl.fma(
        axis_z,
        polarization_z,
        tl.fma(
            axis_x, polarization_x, axis_y * polarization_y
        ),
    )
    normal_probability = normal_coefficient * normal_coefficient
    choose_normal = u_polarization < normal_probability

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
    reflectance = reflection_coefficient * reflection_coefficient
    total_internal_reflection = refracted_angle != refracted_angle
    reflected = (u_reflect < reflectance) | total_internal_reflection

    pi = tl.full(direction_x.shape, 0x40490FDB, tl.uint32).to(
        tl.float32, bitcast=True
    )
    outgoing_angle = tl.where(
        reflected, incident_angle, pi - refracted_angle
    )
    out_dx, out_dy, out_dz = rotate_chroma_fast(
        normal_x,
        normal_y,
        normal_z,
        outgoing_angle,
        axis_x,
        axis_y,
        axis_z,
    )

    parallel_px, parallel_py, parallel_pz = cross_chroma_fast(
        axis_x,
        axis_y,
        axis_z,
        out_dx,
        out_dy,
        out_dz,
    )
    parallel_norm2 = tl.fma(
        parallel_pz,
        parallel_pz,
        tl.fma(
            parallel_px, parallel_px, parallel_py * parallel_py
        ),
    )
    parallel_norm = tl.sqrt(parallel_norm2)
    parallel_px /= parallel_norm
    parallel_py /= parallel_norm
    parallel_pz /= parallel_norm
    out_px = tl.where(choose_normal, axis_x, parallel_px)
    out_py = tl.where(choose_normal, axis_y, parallel_py)
    out_pz = tl.where(choose_normal, axis_z, parallel_pz)
    return (
        out_dx,
        out_dy,
        out_dz,
        out_px,
        out_py,
        out_pz,
        reflected,
        reflectance,
        normal_probability,
        total_internal_reflection,
    )


@triton.jit
def fresnel_coefficients(cos_incident, refractive_index1, refractive_index2):
    """Angle-free dielectric ``(R_s, R_p, cos_t, TIR)``."""

    ci = tl.minimum(1.0, tl.maximum(0.0, cos_incident))
    eta = refractive_index1 / refractive_index2
    sin_transmitted2 = eta * eta * tl.maximum(0.0, 1.0 - ci * ci)
    tir = sin_transmitted2 > 1.0
    cos_transmitted = tl.sqrt(tl.maximum(0.0, 1.0 - sin_transmitted2))

    denom_s = refractive_index1 * ci + refractive_index2 * cos_transmitted
    denom_p = refractive_index2 * ci + refractive_index1 * cos_transmitted
    amp_s = tl.where(
        tl.abs(denom_s) > 1.0e-20,
        (refractive_index1 * ci - refractive_index2 * cos_transmitted)
        / tl.maximum(tl.abs(denom_s), 1.0e-20),
        0.0,
    )
    amp_p = tl.where(
        tl.abs(denom_p) > 1.0e-20,
        (refractive_index2 * ci - refractive_index1 * cos_transmitted)
        / tl.maximum(tl.abs(denom_p), 1.0e-20),
        0.0,
    )
    reflect_s = tl.where(tir, 1.0, amp_s * amp_s)
    reflect_p = tl.where(tir, 1.0, amp_p * amp_p)
    return reflect_s, reflect_p, cos_transmitted, tir


@triton.jit
def refract_direction(
    direction_x,
    direction_y,
    direction_z,
    normal_x,
    normal_y,
    normal_z,
    cos_incident,
    cos_transmitted,
    refractive_index1,
    refractive_index2,
):
    """Vector form of Snell's law (caller handles TIR)."""

    eta = refractive_index1 / refractive_index2
    normal_scale = eta * cos_incident - cos_transmitted
    return (
        eta * direction_x + normal_scale * normal_x,
        eta * direction_y + normal_scale * normal_y,
        eta * direction_z + normal_scale * normal_z,
    )


@triton.jit
def fresnel_step(
    direction_x,
    direction_y,
    direction_z,
    polarization_x,
    polarization_y,
    polarization_z,
    normal_x,
    normal_y,
    normal_z,
    refractive_index1,
    refractive_index2,
    u_polarization,
    u_reflect,
):
    """Chroma-compatible polarization selection and dielectric transition."""

    cos_incident = -(
        direction_x * normal_x + direction_y * normal_y + direction_z * normal_z
    )
    cos_incident = tl.minimum(1.0, tl.maximum(0.0, cos_incident))
    reflect_s, reflect_p, cos_transmitted, tir = fresnel_coefficients(
        cos_incident, refractive_index1, refractive_index2
    )

    # s is normal to the plane of incidence.  At normal incidence Chroma uses
    # the incoming polarization because that plane is not unique.
    sx_raw = direction_y * normal_z - direction_z * normal_y
    sy_raw = direction_z * normal_x - direction_x * normal_z
    sz_raw = direction_x * normal_y - direction_y * normal_x
    s_norm2 = sx_raw * sx_raw + sy_raw * sy_raw + sz_raw * sz_raw
    use_polarization_axis = s_norm2 < 1.0e-12
    inv_s = tl.rsqrt(tl.maximum(s_norm2, 1.0e-20))
    sx = tl.where(use_polarization_axis, polarization_x, sx_raw * inv_s)
    sy = tl.where(use_polarization_axis, polarization_y, sy_raw * inv_s)
    sz = tl.where(use_polarization_axis, polarization_z, sz_raw * inv_s)

    normal_coefficient = polarization_x * sx + polarization_y * sy + polarization_z * sz
    s_fraction = tl.minimum(1.0, tl.maximum(0.0, normal_coefficient * normal_coefficient))
    choose_s = u_polarization < s_fraction
    selected_reflectance = tl.where(choose_s, reflect_s, reflect_p)
    reflected = tir | (u_reflect < selected_reflectance)

    rdx, rdy, rdz = reflect_specular(
        direction_x, direction_y, direction_z, normal_x, normal_y, normal_z
    )
    tdx, tdy, tdz = refract_direction(
        direction_x,
        direction_y,
        direction_z,
        normal_x,
        normal_y,
        normal_z,
        cos_incident,
        cos_transmitted,
        refractive_index1,
        refractive_index2,
    )
    out_dx = tl.where(reflected, rdx, tdx)
    out_dy = tl.where(reflected, rdy, tdy)
    out_dz = tl.where(reflected, rdz, tdz)

    # p is in the plane of incidence and transverse to the selected ray.
    px = sy * out_dz - sz * out_dy
    py = sz * out_dx - sx * out_dz
    pz = sx * out_dy - sy * out_dx
    inv_p = tl.rsqrt(tl.maximum(px * px + py * py + pz * pz, 1.0e-20))
    px *= inv_p
    py *= inv_p
    pz *= inv_p
    out_px = tl.where(choose_s, sx, px)
    out_py = tl.where(choose_s, sy, py)
    out_pz = tl.where(choose_s, sz, pz)
    return (
        out_dx,
        out_dy,
        out_dz,
        out_px,
        out_py,
        out_pz,
        reflected,
        selected_reflectance,
        s_fraction,
        tir,
    )


@triton.jit(do_not_specialize=[10, 11, 12])
def generate_photon_bomb_kernel(
    position_x,
    position_y,
    position_z,
    direction_x,
    direction_y,
    direction_z,
    polarization_x,
    polarization_y,
    polarization_z,
    wavelengths,
    n_photons,
    seed,
    photon_id_base,
    center_x,
    center_y,
    center_z,
    voxel_size,
    wavelength_min,
    wavelength_max,
    BLOCK_SIZE: tl.constexpr,
):
    """Generate source photons directly into structure-of-arrays buffers.

    The global photon ID is the Philox counter, so output is invariant to block
    size, launch order, and tiled generation.  Two domain-separated Philox4x32
    calls provide all seven required uniforms.
    """

    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_photons
    photon_ids = offsets + photon_id_base
    u0, u1, u2, _ = tl.rand4x(seed + SOURCE_DOMAIN_0, photon_ids)
    u4, u5, u6, u7 = tl.rand4x(seed + SOURCE_DOMAIN_1, photon_ids)

    # Isotropic direction.
    dz = 2.0 * u0 - 1.0
    radial = tl.sqrt(tl.maximum(0.0, 1.0 - dz * dz))
    azimuth = TWO_PI * u1
    dx = radial * tl.cos(azimuth)
    dy = radial * tl.sin(azimuth)

    # A robust tangent frame, followed by one uniform polarization angle.
    # cross(z, d) is preferred except close to the z poles, where cross(x, d)
    # avoids a vanishing vector.
    use_z = tl.abs(dz) < 0.9
    bx_raw = tl.where(use_z, -dy, 0.0)
    by_raw = tl.where(use_z, dx, -dz)
    bz_raw = tl.where(use_z, 0.0, dy)
    inv_b = tl.rsqrt(
        tl.maximum(bx_raw * bx_raw + by_raw * by_raw + bz_raw * bz_raw, 1.0e-20)
    )
    bx = bx_raw * inv_b
    by = by_raw * inv_b
    bz = bz_raw * inv_b
    cx = dy * bz - dz * by
    cy = dz * bx - dx * bz
    cz = dx * by - dy * bx
    pol_angle = TWO_PI * u2
    cos_pol = tl.cos(pol_angle)
    sin_pol = tl.sin(pol_angle)
    px = cos_pol * bx + sin_pol * cx
    py = cos_pol * by + sin_pol * cy
    pz = cos_pol * bz + sin_pol * cz

    half_voxel = 0.5 * voxel_size
    x = center_x + voxel_size * u4 - half_voxel
    y = center_y + voxel_size * u5 - half_voxel
    z = center_z + voxel_size * u6 - half_voxel
    wavelength = wavelength_min + (wavelength_max - wavelength_min) * u7

    tl.store(position_x + offsets, x, mask=mask)
    tl.store(position_y + offsets, y, mask=mask)
    tl.store(position_z + offsets, z, mask=mask)
    tl.store(direction_x + offsets, dx, mask=mask)
    tl.store(direction_y + offsets, dy, mask=mask)
    tl.store(direction_z + offsets, dz, mask=mask)
    tl.store(polarization_x + offsets, px, mask=mask)
    tl.store(polarization_y + offsets, py, mask=mask)
    tl.store(polarization_z + offsets, pz, mask=mask)
    tl.store(wavelengths + offsets, wavelength, mask=mask)


@triton.jit
def rayleigh_scatter_kernel(
    direction_x,
    direction_y,
    direction_z,
    polarization_x,
    polarization_y,
    polarization_z,
    u0,
    u1,
    u2,
    u_phi,
    out_direction_x,
    out_direction_y,
    out_direction_z,
    out_polarization_x,
    out_polarization_y,
    out_polarization_z,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    out_dx, out_dy, out_dz, out_px, out_py, out_pz = rayleigh_scatter(
        tl.load(direction_x + offsets, mask=mask, other=0.0),
        tl.load(direction_y + offsets, mask=mask, other=0.0),
        tl.load(direction_z + offsets, mask=mask, other=0.0),
        tl.load(polarization_x + offsets, mask=mask, other=0.0),
        tl.load(polarization_y + offsets, mask=mask, other=0.0),
        tl.load(polarization_z + offsets, mask=mask, other=0.0),
        tl.load(u0 + offsets, mask=mask, other=0.0),
        tl.load(u1 + offsets, mask=mask, other=0.0),
        tl.load(u2 + offsets, mask=mask, other=0.0),
        tl.load(u_phi + offsets, mask=mask, other=0.0),
    )
    tl.store(out_direction_x + offsets, out_dx, mask=mask)
    tl.store(out_direction_y + offsets, out_dy, mask=mask)
    tl.store(out_direction_z + offsets, out_dz, mask=mask)
    tl.store(out_polarization_x + offsets, out_px, mask=mask)
    tl.store(out_polarization_y + offsets, out_py, mask=mask)
    tl.store(out_polarization_z + offsets, out_pz, mask=mask)


@triton.jit
def bulk_collision_kernel(
    absorption_length,
    scattering_length,
    u_distance,
    u_process,
    out_distance,
    out_process,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    distance, process = sample_bulk_collision(
        tl.load(absorption_length + offsets, mask=mask, other=1.0e30),
        tl.load(scattering_length + offsets, mask=mask, other=1.0e30),
        tl.load(u_distance + offsets, mask=mask, other=1.0),
        tl.load(u_process + offsets, mask=mask, other=0.0),
    )
    tl.store(out_distance + offsets, distance, mask=mask)
    tl.store(out_process + offsets, process, mask=mask)


@triton.jit
def fresnel_step_kernel(
    direction_x,
    direction_y,
    direction_z,
    polarization_x,
    polarization_y,
    polarization_z,
    normal_x,
    normal_y,
    normal_z,
    refractive_index1,
    refractive_index2,
    u_polarization,
    u_reflect,
    out_direction_x,
    out_direction_y,
    out_direction_z,
    out_polarization_x,
    out_polarization_y,
    out_polarization_z,
    out_reflected,
    out_reflectance,
    out_s_fraction,
    out_tir,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    (
        out_dx,
        out_dy,
        out_dz,
        out_px,
        out_py,
        out_pz,
        reflected,
        reflectance,
        s_fraction,
        tir,
    ) = fresnel_step(
        tl.load(direction_x + offsets, mask=mask, other=0.0),
        tl.load(direction_y + offsets, mask=mask, other=0.0),
        tl.load(direction_z + offsets, mask=mask, other=0.0),
        tl.load(polarization_x + offsets, mask=mask, other=0.0),
        tl.load(polarization_y + offsets, mask=mask, other=0.0),
        tl.load(polarization_z + offsets, mask=mask, other=0.0),
        tl.load(normal_x + offsets, mask=mask, other=0.0),
        tl.load(normal_y + offsets, mask=mask, other=0.0),
        tl.load(normal_z + offsets, mask=mask, other=0.0),
        tl.load(refractive_index1 + offsets, mask=mask, other=1.0),
        tl.load(refractive_index2 + offsets, mask=mask, other=1.0),
        tl.load(u_polarization + offsets, mask=mask, other=0.0),
        tl.load(u_reflect + offsets, mask=mask, other=0.0),
    )
    tl.store(out_direction_x + offsets, out_dx, mask=mask)
    tl.store(out_direction_y + offsets, out_dy, mask=mask)
    tl.store(out_direction_z + offsets, out_dz, mask=mask)
    tl.store(out_polarization_x + offsets, out_px, mask=mask)
    tl.store(out_polarization_y + offsets, out_py, mask=mask)
    tl.store(out_polarization_z + offsets, out_pz, mask=mask)
    tl.store(out_reflected + offsets, reflected, mask=mask)
    tl.store(out_reflectance + offsets, reflectance, mask=mask)
    tl.store(out_s_fraction + offsets, s_fraction, mask=mask)
    tl.store(out_tir + offsets, tir, mask=mask)


@dataclass
class PhotonBombSoA:
    """GPU source arrays in the layout consumed by the transport kernels."""

    position_x: torch.Tensor
    position_y: torch.Tensor
    position_z: torch.Tensor
    direction_x: torch.Tensor
    direction_y: torch.Tensor
    direction_z: torch.Tensor
    polarization_x: torch.Tensor
    polarization_y: torch.Tensor
    polarization_z: torch.Tensor
    wavelength: torch.Tensor

    def __len__(self) -> int:
        return self.position_x.numel()

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        return (
            self.position_x,
            self.position_y,
            self.position_z,
            self.direction_x,
            self.direction_y,
            self.direction_z,
            self.polarization_x,
            self.polarization_y,
            self.polarization_z,
            self.wavelength,
        )

    def position(self) -> torch.Tensor:
        """Materialize an ``(N, 3)`` view-compatible position tensor."""

        return torch.stack(self.tensors[0:3], dim=1)

    def direction(self) -> torch.Tensor:
        """Materialize an ``(N, 3)`` direction tensor for validation/I/O."""

        return torch.stack(self.tensors[3:6], dim=1)

    def polarization(self) -> torch.Tensor:
        """Materialize an ``(N, 3)`` polarization tensor for validation/I/O."""

        return torch.stack(self.tensors[6:9], dim=1)


def allocate_photon_bomb(
    n_photons: int, *, device: Union[str, torch.device] = "cuda"
) -> PhotonBombSoA:
    """Allocate reusable source buffers; no random generation is performed."""

    if n_photons < 0:
        raise ValueError("n_photons must be non-negative")
    arrays = [torch.empty(n_photons, dtype=torch.float32, device=device) for _ in range(10)]
    return PhotonBombSoA(*arrays)


def generate_photon_bomb_into(
    output: PhotonBombSoA,
    *,
    seed: int,
    center: Sequence[float],
    voxel_size: float,
    wavelength: Union[float, tuple[float, float]] = 450.0,
    photon_id_base: int = 0,
    block_size: int = 256,
) -> PhotonBombSoA:
    """Generate a reproducible isotropic voxel source into existing buffers."""

    n_photons = len(output)
    if any(t.numel() != n_photons for t in output.tensors):
        raise ValueError("all PhotonBombSoA arrays must have the same length")
    if any(t.dtype != torch.float32 for t in output.tensors):
        raise TypeError("PhotonBombSoA arrays must be float32")
    if len(center) != 3:
        raise ValueError("center must contain three coordinates")
    if voxel_size < 0.0:
        raise ValueError("voxel_size must be non-negative")
    if isinstance(wavelength, tuple):
        wavelength_min, wavelength_max = wavelength
    else:
        wavelength_min = wavelength_max = wavelength
    if wavelength_max < wavelength_min:
        raise ValueError("wavelength range must be ordered")
    if n_photons == 0:
        return output

    grid = (triton.cdiv(n_photons, block_size),)
    generate_photon_bomb_kernel[grid](
        *output.tensors,
        n_photons,
        int(seed),
        int(photon_id_base),
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(voxel_size),
        float(wavelength_min),
        float(wavelength_max),
        BLOCK_SIZE=block_size,
    )
    return output


def generate_photon_bomb(
    n_photons: int,
    *,
    seed: int,
    center: Sequence[float],
    voxel_size: float,
    wavelength: Union[float, tuple[float, float]] = 450.0,
    photon_id_base: int = 0,
    device: Union[str, torch.device] = "cuda",
    block_size: int = 256,
) -> PhotonBombSoA:
    """Allocate and generate a Philox-keyed photon bomb on the GPU."""

    output = allocate_photon_bomb(n_photons, device=device)
    return generate_photon_bomb_into(
        output,
        seed=seed,
        center=center,
        voxel_size=voxel_size,
        wavelength=wavelength,
        photon_id_base=photon_id_base,
        block_size=block_size,
    )


def benchmark_source_generation(
    n_photons: int = 15_000_000,
    *,
    warmup: int = 25,
    rep: int = 100,
) -> dict[str, float]:
    """Benchmark generation into preallocated arrays on the active CUDA GPU."""

    output = allocate_photon_bomb(n_photons)

    def launch() -> None:
        generate_photon_bomb_into(
            output,
            seed=1,
            center=(0.0, 0.0, 0.0),
            voxel_size=30.0,
            wavelength=450.0,
        )

    milliseconds = triton.testing.do_bench(launch, warmup=warmup, rep=rep)
    return {
        "photons": float(n_photons),
        "milliseconds": float(milliseconds),
        "million_photons_per_second": n_photons / (milliseconds * 1.0e3),
        "effective_gigabytes_per_second": (n_photons * 10 * 4) / (milliseconds * 1.0e6),
    }


__all__ = [
    "BULK_NONE",
    "BULK_ABSORB",
    "BULK_SCATTER",
    "PhotonBombSoA",
    "allocate_photon_bomb",
    "generate_photon_bomb",
    "generate_photon_bomb_into",
    "generate_photon_bomb_kernel",
    "rayleigh_scatter_kernel",
    "bulk_collision_kernel",
    "fresnel_step_kernel",
    "median3",
    "rayleigh_cosine",
    "rayleigh_scatter",
    "sample_bulk_collision",
    "exponential_distance_chroma_fast",
    "advance_position_chroma_fast",
    "advance_time_chroma_fast",
    "fresnel_incident_cosine_chroma_fast",
    "cross_chroma_fast",
    "reflect_specular",
    "acos_chroma_fast",
    "asin_chroma_fast",
    "rotate_chroma_fast",
    "reflect_specular_chroma",
    "fresnel_step_chroma",
    "fresnel_coefficients",
    "refract_direction",
    "fresnel_step",
]
