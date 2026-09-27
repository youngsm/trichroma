// Raw-word probe for Chroma's historical dielectric boundary operation.
// Compile through chroma.gpu.tools.get_cu_module so this receives the same
// --use_fast_math flags as propagate.cu.

#include "physical_constants.h"
#include "rotate.h"

extern "C"
__global__ void
fresnel_probe(int count,
              const float3 *__restrict__ directions,
              const float3 *__restrict__ polarizations,
              const float3 *__restrict__ normals,
              const float *__restrict__ refractive_index1,
              const float *__restrict__ refractive_index2,
              const float *__restrict__ u_polarization,
              const float *__restrict__ u_reflect,
              float3 *__restrict__ output_direction,
              float3 *__restrict__ output_polarization,
              float *__restrict__ diagnostics,
              unsigned int *__restrict__ decisions)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const float3 direction = directions[index];
    const float3 polarization = polarizations[index];
    const float3 normal = normals[index];
    const float n1 = refractive_index1[index];
    const float n2 = refractive_index2[index];

    const float incident_cosine =
        fmaxf(-1.0f, fminf(1.0f, dot(normal, -direction)));
    const float incident_angle = acosf(incident_cosine);
    const float refracted_argument = sinf(incident_angle) * n1 / n2;
    const float refracted_angle = asinf(refracted_argument);

    float3 incidence_axis = cross(direction, normal);
    const float raw_axis_length = norm(incidence_axis);
    const bool normal_incidence = raw_axis_length < 1.0e-6f;
    if (normal_incidence)
        incidence_axis = polarization;
    else
        incidence_axis /= raw_axis_length;

    const float normal_coefficient = dot(polarization, incidence_axis);
    const float normal_probability = normal_coefficient * normal_coefficient;
    const bool choose_normal = u_polarization[index] < normal_probability;

    float reflection_coefficient;
    if (choose_normal) {
        reflection_coefficient =
            -sinf(incident_angle - refracted_angle)
            / sinf(incident_angle + refracted_angle);
    }
    else {
        reflection_coefficient =
            tanf(incident_angle - refracted_angle)
            / tanf(incident_angle + refracted_angle);
    }
    const float reflectance = reflection_coefficient * reflection_coefficient;
    const bool total_internal_reflection = isnan(refracted_angle);
    const bool reflected =
        (u_reflect[index] < reflectance) || total_internal_reflection;
    const float outgoing_angle =
        reflected ? incident_angle : PI - refracted_angle;

    const float3 new_direction = rotate(
        normal, outgoing_angle, incidence_axis);
    float3 new_polarization;
    if (choose_normal) {
        new_polarization = incidence_axis;
    }
    else {
        new_polarization = cross(incidence_axis, new_direction);
        new_polarization /= norm(new_polarization);
    }

    output_direction[index] = new_direction;
    output_polarization[index] = new_polarization;
    const int base = index * 12;
    diagnostics[base + 0] = incident_cosine;
    diagnostics[base + 1] = incident_angle;
    diagnostics[base + 2] = refracted_argument;
    diagnostics[base + 3] = refracted_angle;
    diagnostics[base + 4] = raw_axis_length;
    diagnostics[base + 5] = incidence_axis.x;
    diagnostics[base + 6] = incidence_axis.y;
    diagnostics[base + 7] = incidence_axis.z;
    diagnostics[base + 8] = normal_probability;
    diagnostics[base + 9] = reflection_coefficient;
    diagnostics[base + 10] = reflectance;
    diagnostics[base + 11] = outgoing_angle;
    const int decision_base = index * 3;
    decisions[decision_base + 0] = choose_normal;
    decisions[decision_base + 1] = reflected;
    decisions[decision_base + 2] = total_internal_reflection;
}
