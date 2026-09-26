#include "cuComplex.h"

/* complex math functions adding to those available in cuComplex.h */

__host__ __device__ static __inline__ cuFloatComplex cuCsinf (cuFloatComplex x)
{
    float u = coshf(x.y) * sinf(x.x);
    float v = sinhf(x.y) * cosf(x.x);
    return make_cuFloatComplex(u, v);
}

__host__ __device__ static __inline__ cuFloatComplex cuCcosf (cuFloatComplex x)
{
    float u = coshf(x.y) * cosf(x.x);
    float v = -sinhf(x.y) * sinf(x.x);
    return make_cuFloatComplex(u, v);
}

__host__ __device__ static __inline__ cuFloatComplex cuCtanf (cuFloatComplex x)
{
    return cuCdivf(cuCsinf(x), cuCcosf(x));
}

__host__ __device__ static __inline__ float cuCargf (cuFloatComplex x)
{
    return atan2f(x.y, x.x);
}

__host__ __device__ static __inline__ cuFloatComplex cuCsqrtf (cuFloatComplex x)
{
    float r = sqrtf(cuCabsf(x));
    float t = cuCargf(x) / 2.0f;
    return make_cuFloatComplex(r * cosf(t), r * sinf(t));
}

