//-*-c-*-

#include "linalg.h"

extern "C"
{

__global__ void float3add(float3 *a, float3 *b, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = a[idx] + b[idx];
}

__global__ void float3addequal(float3 *a, float3 *b)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	a[idx] += b[idx];
}

__global__ void float3sub(float3 *a, float3 *b, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = a[idx] - b[idx];
}

__global__ void float3subequal(float3 *a, float3 *b)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	a[idx] -= b[idx];
}

__global__ void float3addfloat(float3 *a, float c, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = a[idx] + c;
}

__global__ void float3addfloatequal(float3 *a, float c)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	a[idx] += c;
}

__global__ void floataddfloat3(float3 *a, float c, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = c + a[idx];
}

__global__ void float3subfloat(float3 *a, float c, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = a[idx] - c;
}

__global__ void float3subfloatequal(float3 *a, float c)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	a[idx] -= c;
}

__global__ void floatsubfloat3(float3 *a, float c, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = c - a[idx];
}

__global__ void float3mulfloat(float3 *a, float c, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = a[idx]*c;
}

__global__ void float3mulfloatequal(float3 *a, float c)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	a[idx] *= c;
}

__global__ void floatmulfloat3(float3 *a, float c, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = c*a[idx];
}

__global__ void float3divfloat(float3 *a, float c, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = a[idx]/c;
}

__global__ void float3divfloatequal(float3 *a, float c)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	a[idx] /= c;
}

__global__ void floatdivfloat3(float3 *a, float c, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = c/a[idx];
}

__global__ void dot(float3 *a, float3 *b, float *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = dot(a[idx],b[idx]);
}

__global__ void cross(float3 *a, float3 *b, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = cross(a[idx],b[idx]);
}

__global__ void norm(float3 *a, float *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = norm(a[idx]);
}

__global__ void minusfloat3(float3 *a, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = -a[idx];
}

} // extern "c"
