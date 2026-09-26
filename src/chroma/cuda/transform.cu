//-*-c-*-

#include "linalg.h"
#include "rotate.h"

extern "C"
{

/* Translate the points `a` by the vector `v` */
__global__ void
translate(int nthreads, float3 *a, float3 v)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    a[id] += v;
}

/* Rotate the points `a` through an angle `phi` counter-clockwise about the
   axis `axis` (when looking towards +infinity). */
__global__ void
rotate(int nthreads, float3 *a, float phi, float3 axis)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    a[id] = rotate(a[id], phi, axis);
}

/* Rotate the points `a` through an angle `phi` counter-clockwise
   (when looking towards +infinity along `axis`) about the axis defined
   by the point `point` and the vector `axis` . */
__global__ void
rotate_around_point(int nthreads, float3 *a, float phi, float3 axis,
		    float3 point)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    a[id] -= point;
    a[id] = rotate(a[id], phi, axis);
    a[id] += point;
}

} // extern "c"
