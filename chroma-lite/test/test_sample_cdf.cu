// -*-c++-*-
#include "random.h"

extern "C" {

__global__ void test_sample_cdf(int offset, int ncdf, 
				float *cdf_x, float *cdf_y, float *out)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  curandState s;
  curand_init(0, id, offset, &s);

  out[id] = sample_cdf(&s, ncdf, cdf_x, cdf_y);
}

}
