//-*-c-*-

#include "rotate.h"

extern "C"
{

__global__ void rotate(float3 *a, float *phi, float3 n, float3 *dest)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx] = rotate(a[idx], phi[idx], n);
}

} // extern "c"
