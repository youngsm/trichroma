#ifndef __ROTATE_H__
#define __ROTATE_H__

#include "linalg.h"
#include "matrix.h"

__device__ const Matrix IDENTITY_MATRIX = {1,0,0,0,1,0,0,0,1};

__device__ Matrix
make_rotation_matrix(float phi, const float3 &n)
{
    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    return IDENTITY_MATRIX*cos_phi + (1-cos_phi)*outer(n,n) +
	sin_phi*make_matrix(0,n.z,-n.y,-n.z,0,n.x,n.y,-n.x,0);
}

/* rotate points counterclockwise, when looking towards +infinity,
   through an angle `phi` about the axis `n`. */
__device__ float3
rotate(const float3 &a, float phi, const float3 &n)
{
    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    return a*cos_phi + n*dot(a,n)*(1.0f-cos_phi) + cross(a,n)*sin_phi;
}

/* rotate points counterclockwise, when looking towards +infinity,
   through an angle `phi` about the axis `n`. */
__device__ float3
rotate_with_matrix(const float3 &a, float phi, const float3 &n)
{
    return make_rotation_matrix(phi,n)*a;
}

#endif
