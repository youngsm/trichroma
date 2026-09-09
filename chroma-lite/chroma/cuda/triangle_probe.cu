// Raw-word/PTX probe for Chroma's historical Moller-Trumbore intersection.
// Compile with the same --use_fast_math option as propagate.cu.

#include "intersect.h"

extern "C"
__global__ void
triangle_probe(int count,
               const float3 *__restrict__ origins,
               const float3 *__restrict__ directions,
               const float3 *__restrict__ vertices,
               float *__restrict__ distances,
               unsigned char *__restrict__ hits)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    Triangle triangle;
    triangle.v0 = vertices[3 * index];
    triangle.v1 = vertices[3 * index + 1];
    triangle.v2 = vertices[3 * index + 2];
    float distance = 0.0f;
    const bool hit = intersect_triangle(
        origins[index], directions[index], triangle, distance);
    distances[index] = distance;
    hits[index] = hit;
}
