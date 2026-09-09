// PTX probe for Chroma fill_state's mesh-normal calculation and orientation.
// Compile with the same --use_fast_math option as propagate.cu.

#include "linalg.h"

extern "C"
__global__ void
normal_probe(int count,
             const float3 *__restrict__ directions,
             const float3 *__restrict__ vertices,
             float3 *__restrict__ normals,
             unsigned char *__restrict__ inside_to_outside)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const float3 v0 = vertices[3 * index];
    const float3 v1 = vertices[3 * index + 1];
    const float3 v2 = vertices[3 * index + 2];
    const float3 v01 = v1 - v0;
    const float3 v12 = v2 - v1;
    float3 normal = normalize(cross(v01, v12));
    bool inside = false;
    if (dot(normal, -directions[index]) <= 0.0f) {
        normal = -normal;
        inside = true;
    }
    normals[index] = normal;
    inside_to_outside[index] = inside;
}
