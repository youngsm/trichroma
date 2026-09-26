// Raw-word compatibility probe for Chroma's historical surface reflection.
// Compile this through chroma.gpu.tools.get_cu_module so it receives the same
// --use_fast_math option as propagate.cu.

#include "rotate.h"

extern "C"
__global__ void
specular_probe(int count,
               const float3 *__restrict__ directions,
               const float3 *__restrict__ normals,
               float3 *__restrict__ output,
               float *__restrict__ diagnostics)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const float3 direction = directions[index];
    const float3 normal = normals[index];
    const float incident_cosine =
        fmaxf(-1.0f, fminf(1.0f, dot(normal, -direction)));
    const float incident_angle = acosf(incident_cosine);
    float3 incident_plane_normal = cross(direction, normal);
    const float3 raw_axis = incident_plane_normal;
    const float axis_length = norm(incident_plane_normal);
    incident_plane_normal /= axis_length;
    output[index] = rotate(
        normal, incident_angle, incident_plane_normal);
    const float cosine = cosf(incident_angle);
    const float sine = sinf(incident_angle);
    const float projection = dot(normal, incident_plane_normal);
    const int base = index * 12;
    diagnostics[base + 0] = incident_cosine;
    diagnostics[base + 1] = incident_angle;
    diagnostics[base + 2] = raw_axis.x;
    diagnostics[base + 3] = raw_axis.y;
    diagnostics[base + 4] = raw_axis.z;
    diagnostics[base + 5] = axis_length;
    diagnostics[base + 6] = incident_plane_normal.x;
    diagnostics[base + 7] = incident_plane_normal.y;
    diagnostics[base + 8] = incident_plane_normal.z;
    diagnostics[base + 9] = cosine;
    diagnostics[base + 10] = sine;
    diagnostics[base + 11] = projection;
}
