#ifndef __INTERPOLATE_H__
#define __INTERPOLATE_H__

__device__ float
interp_idx(float x, int n, float *xp)
{
    int lower = 0;
    int upper = n-1;

    if (x <= xp[lower])
	return lower;

    if (x >= xp[upper])
	return upper;

    while (lower < upper-1)
    {
	int half = (lower+upper)/2;

	if (x < xp[half])
	    upper = half;
	else
	    lower = half;
    }

    float dx = xp[upper] - xp[lower];

    return lower + 1.0*(x-xp[lower])/dx;
}


__device__ float
interp(float x, int n, float *xp, float *fp)
{
    int lower = 0;
    int upper = n-1;

    if (x <= xp[lower])
	return fp[lower];

    if (x >= xp[upper])
	return fp[upper];

    while (lower < upper-1)
    {
	int half = (lower+upper)/2;

	if (x < xp[half])
	    upper = half;
	else
	    lower = half;
    }

    float df = fp[upper] - fp[lower];
    float dx = xp[upper] - xp[lower];

    return fp[lower] + df*(x-xp[lower])/dx;
}

__device__ float
interp_uniform(float x, int n, float x0, float dx, float *fp)
{
    if (x <= x0)
	return x0;

    float xmax = x0 + dx*(n-1);

    if (x >= xmax)
	return xmax;

    int lower = (x - x0)/dx;
    int upper = lower + 1;

    float df = fp[upper] - fp[lower];

    return fp[lower] + df*(x-(x0+dx*lower))/dx;
}

#endif
