"""Distribution-equivalent CPU references for the Triton optical primitives.

These routines describe the probability laws used by the optimized transport
backend.  They are intentionally independent of Triton so that correctness can
be tested on machines without a GPU.

The optimized laws are equivalent in distribution to Chroma's implementation,
but do not consume the same random-number sequence:

* a median of three uniforms is ``Beta(2, 2)`` and therefore gives the exact
  Rayleigh cosine density ``3/4 * (1 - cos(theta)**2)``;
* the minimum of independent absorption and scattering exponentials is one
  exponential with their summed hazard, followed by a categorical draw;
* reflection, Snell refraction, and Fresnel coefficients need dot products and
  square roots, not angles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


ArrayLike = Union[float, np.ndarray]

BULK_NONE = np.int8(0)
BULK_ABSORB = np.int8(1)
BULK_SCATTER = np.int8(2)


@dataclass(frozen=True)
class BulkCollision:
    """A sampled homogeneous-medium collision.

    ``process`` is one of :data:`BULK_NONE`, :data:`BULK_ABSORB`, or
    :data:`BULK_SCATTER`.  ``distance`` is infinity when both hazards vanish.
    """

    distance: np.ndarray
    process: np.ndarray


@dataclass(frozen=True)
class FresnelCoefficients:
    """Angle-free dielectric-interface quantities."""

    reflect_s: np.ndarray
    reflect_p: np.ndarray
    cos_transmitted: np.ndarray
    total_internal_reflection: np.ndarray


@dataclass(frozen=True)
class FresnelStep:
    """Result of Chroma-compatible polarization-resolved Fresnel sampling."""

    direction: np.ndarray
    polarization: np.ndarray
    reflected: np.ndarray
    selected_reflectance: np.ndarray
    s_fraction: np.ndarray
    total_internal_reflection: np.ndarray


def _unit(vector: np.ndarray, *, eps: float = 1.0e-30) -> np.ndarray:
    vector = np.asarray(vector)
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.maximum(norm, eps)


def median3(u0: ArrayLike, u1: ArrayLike, u2: ArrayLike) -> np.ndarray:
    """Return the elementwise median without sorting or allocating a stack."""

    u0, u1, u2 = np.broadcast_arrays(u0, u1, u2)
    return np.maximum(np.minimum(u0, u1), np.minimum(np.maximum(u0, u1), u2))


def rayleigh_cosine_from_uniforms(
    u0: ArrayLike, u1: ArrayLike, u2: ArrayLike
) -> np.ndarray:
    """Sample the exact Chroma Rayleigh cosine law from three uniforms.

    If ``X`` is the median of three independent ``U(0, 1)`` variates then
    ``X ~ Beta(2, 2)``.  Consequently ``C = 2*X - 1`` has density
    ``p(C) = 3/4 * (1-C**2)`` on ``[-1, 1]``.
    """

    return 2.0 * median3(u0, u1, u2) - 1.0


def rayleigh_cosine_cdf(cosine: ArrayLike) -> np.ndarray:
    """Analytic CDF of :func:`rayleigh_cosine_from_uniforms`."""

    c = np.clip(np.asarray(cosine), -1.0, 1.0)
    return 0.5 + 0.75 * c - 0.25 * c * c * c


def rayleigh_scatter(
    direction: np.ndarray,
    polarization: np.ndarray,
    u0: ArrayLike,
    u1: ArrayLike,
    u2: ArrayLike,
    u_phi: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Rayleigh-scatter around polarization without inverse trig.

    Chroma samples an outgoing direction around the *polarization* axis, not
    around the old photon direction, then projects that axis into the outgoing
    transverse plane.  This distinction matters because Chroma's polished
    specular reflector changes direction while leaving polarization unchanged;
    the incoming pair is therefore not necessarily orthogonal.

    For any orthonormal tangent basis ``(b, q)`` about normalized ``p``, let

    ``t = cos(phi)*b + sin(phi)*q``
    ``d_new = c*p + sqrt(1-c*c)*t``
    ``p_new = sqrt(1-c*c)*p - c*t``.

    The old direction is accepted for API symmetry and shape checking but is
    deliberately not used to define the azimuthal frame.  A robust arbitrary
    tangent basis gives the same uniform-azimuth law as Chroma.  The outputs are
    unit length and mutually perpendicular by construction.
    """

    direction = np.asarray(direction)
    polarization = np.asarray(polarization)
    if direction.shape[-1:] != (3,) or polarization.shape[-1:] != (3,):
        raise ValueError("direction and polarization must have final dimension 3")

    p_axis = _unit(polarization)
    # Prefer z x p away from the poles; use x x p close to them.  The choice is
    # deterministic but physically irrelevant because phi is uniform.
    use_z = np.abs(p_axis[..., 2]) < 0.9
    reference = np.zeros_like(p_axis)
    reference[..., 0] = np.where(use_z, 0.0, 1.0)
    reference[..., 2] = np.where(use_z, 1.0, 0.0)
    basis = _unit(np.cross(reference, p_axis))
    quadrature = np.cross(p_axis, basis)
    phi = 2.0 * np.pi * np.asarray(u_phi)
    cp = np.cos(phi)[..., None]
    sp = np.sin(phi)[..., None]
    tangent = cp * basis + sp * quadrature

    cosine = rayleigh_cosine_from_uniforms(u0, u1, u2)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    cosine = cosine[..., None]
    sine = sine[..., None]
    direction_new = cosine * p_axis + sine * tangent
    polarization_new = sine * p_axis - cosine * tangent
    return direction_new, polarization_new


def sample_bulk_collision(
    absorption_length: ArrayLike,
    scattering_length: ArrayLike,
    u_distance: ArrayLike,
    u_process: ArrayLike,
) -> BulkCollision:
    """Sample competing absorption/scattering using their total hazard.

    A non-positive or infinite interaction length contributes zero hazard.
    This uses one exponential and one uniform categorical draw instead of two
    exponentials and is exactly equivalent for independent Poisson processes.
    """

    la, ls, ud, up = np.broadcast_arrays(
        np.asarray(absorption_length, dtype=np.float64),
        np.asarray(scattering_length, dtype=np.float64),
        np.asarray(u_distance, dtype=np.float64),
        np.asarray(u_process, dtype=np.float64),
    )
    rate_a = np.divide(
        1.0,
        la,
        out=np.zeros_like(la),
        where=np.isfinite(la) & (la > 0.0),
    )
    rate_s = np.divide(
        1.0,
        ls,
        out=np.zeros_like(ls),
        where=np.isfinite(ls) & (ls > 0.0),
    )
    rate = rate_a + rate_s

    # GPU uniforms are discrete and can contain zero.  Clamping just that atom
    # prevents log(0); it has no measurable effect on the continuous law.
    ud = np.clip(ud, np.finfo(np.float64).tiny, 1.0)
    distance = np.divide(
        -np.log(ud), rate, out=np.full_like(rate, np.inf), where=rate > 0.0
    )
    scatter_probability = np.divide(
        rate_s, rate, out=np.zeros_like(rate), where=rate > 0.0
    )
    process = np.where(
        rate <= 0.0,
        BULK_NONE,
        np.where(up < scatter_probability, BULK_SCATTER, BULK_ABSORB),
    ).astype(np.int8)
    return BulkCollision(distance=distance, process=process)


def reflect_specular(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Reflect direction about a surface normal using one dot product."""

    direction = np.asarray(direction)
    normal = np.asarray(normal)
    dot_dn = np.sum(direction * normal, axis=-1, keepdims=True)
    return direction - 2.0 * dot_dn * normal


def fresnel_coefficients(
    cos_incident: ArrayLike, refractive_index1: ArrayLike, refractive_index2: ArrayLike
) -> FresnelCoefficients:
    """Return lossless dielectric Fresnel coefficients without angles.

    ``cos_incident`` is non-negative and both indices must be positive.  The
    returned reflectances are power probabilities for s and p polarizations.
    """

    ci, n1, n2 = np.broadcast_arrays(
        np.asarray(cos_incident, dtype=np.float64),
        np.asarray(refractive_index1, dtype=np.float64),
        np.asarray(refractive_index2, dtype=np.float64),
    )
    if np.any(n1 <= 0.0) or np.any(n2 <= 0.0):
        raise ValueError("refractive indices must be positive")
    ci = np.clip(ci, 0.0, 1.0)
    eta = n1 / n2
    sin_t2 = eta * eta * np.maximum(0.0, 1.0 - ci * ci)
    tir = sin_t2 > 1.0
    ct = np.sqrt(np.maximum(0.0, 1.0 - sin_t2))

    denom_s = n1 * ci + n2 * ct
    denom_p = n2 * ci + n1 * ct
    amp_s = np.divide(
        n1 * ci - n2 * ct,
        denom_s,
        out=np.zeros_like(ci),
        where=np.abs(denom_s) > 0.0,
    )
    amp_p = np.divide(
        n2 * ci - n1 * ct,
        denom_p,
        out=np.zeros_like(ci),
        where=np.abs(denom_p) > 0.0,
    )
    rs = np.where(tir, 1.0, amp_s * amp_s)
    rp = np.where(tir, 1.0, amp_p * amp_p)
    return FresnelCoefficients(rs, rp, ct, tir)


def refract_direction(
    direction: np.ndarray,
    normal: np.ndarray,
    refractive_index1: ArrayLike,
    refractive_index2: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply vector Snell refraction; return direction and a TIR mask.

    ``normal`` must face the incoming ray, so ``-dot(direction, normal)`` is the
    non-negative incident cosine.  For TIR lanes the returned vector is the
    specularly reflected direction.
    """

    direction = np.asarray(direction)
    normal = np.asarray(normal)
    ci = np.clip(-np.sum(direction * normal, axis=-1), 0.0, 1.0)
    coeff = fresnel_coefficients(ci, refractive_index1, refractive_index2)
    eta = np.asarray(refractive_index1) / np.asarray(refractive_index2)
    eta = np.broadcast_to(eta, ci.shape)
    transmitted = (
        eta[..., None] * direction
        + (eta * ci - coeff.cos_transmitted)[..., None] * normal
    )
    reflected = reflect_specular(direction, normal)
    return np.where(coeff.total_internal_reflection[..., None], reflected, transmitted), coeff.total_internal_reflection


def fresnel_step(
    direction: np.ndarray,
    polarization: np.ndarray,
    normal: np.ndarray,
    refractive_index1: ArrayLike,
    refractive_index2: ArrayLike,
    u_polarization: ArrayLike,
    u_reflect: ArrayLike,
) -> FresnelStep:
    """Sample Chroma's polarization-resolved lossless dielectric boundary.

    Chroma first selects the s or p component according to the incoming
    polarization, then samples that component's reflectance.  Preserving this
    two-draw construction also preserves its post-boundary pure polarization.
    """

    direction = np.asarray(direction)
    polarization = np.asarray(polarization)
    normal = np.asarray(normal)
    if direction.shape[-1:] != (3,) or normal.shape[-1:] != (3,):
        raise ValueError("direction, polarization, and normal need final dimension 3")

    ci = np.clip(-np.sum(direction * normal, axis=-1), 0.0, 1.0)
    coeff = fresnel_coefficients(ci, refractive_index1, refractive_index2)

    s_axis_raw = np.cross(direction, normal)
    s_norm = np.linalg.norm(s_axis_raw, axis=-1)
    s_axis = np.where(
        (s_norm < 1.0e-6)[..., None],
        _unit(polarization),
        s_axis_raw / np.maximum(s_norm[..., None], 1.0e-30),
    )
    s_fraction = np.clip(np.sum(polarization * s_axis, axis=-1) ** 2, 0.0, 1.0)
    choose_s = np.asarray(u_polarization) < s_fraction
    reflectance = np.where(choose_s, coeff.reflect_s, coeff.reflect_p)
    reflected = coeff.total_internal_reflection | (np.asarray(u_reflect) < reflectance)

    reflected_direction = reflect_specular(direction, normal)
    refracted_direction, _ = refract_direction(
        direction, normal, refractive_index1, refractive_index2
    )
    direction_new = np.where(reflected[..., None], reflected_direction, refracted_direction)

    p_axis = _unit(np.cross(s_axis, direction_new))
    polarization_new = np.where(choose_s[..., None], s_axis, p_axis)
    return FresnelStep(
        direction=direction_new,
        polarization=polarization_new,
        reflected=reflected,
        selected_reflectance=reflectance,
        s_fraction=s_fraction,
        total_internal_reflection=coeff.total_internal_reflection,
    )
