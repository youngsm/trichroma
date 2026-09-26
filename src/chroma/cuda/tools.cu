//--*-c-*-

extern "C"
{

__global__ void
interleave(int nthreads, unsigned long long *x, unsigned long long *y,
	   unsigned long long *z, int bits, unsigned long long *dest)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    for (int i=0; i < bits; i++)
	dest[id] |= (x[id] & 1 << i) << (2*i) |
	            (y[id] & 1 << i) << (2*i+1) |
	            (z[id] & 1 << i) << (2*i+2);
}

} // extern "C"
