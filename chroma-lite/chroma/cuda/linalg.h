#ifndef __LINALG_H__
#define __LINALG_H__

__device__ float3
operator- (const float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ float3
operator* (const float3 &a, const float3 &b)
{
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}

__device__ float3
operator/ (const float3 &a, const float3 &b)
{
    return make_float3(a.x/b.x, a.y/b.y, a.z/b.z);
}

__device__ void
operator*= (float3 &a, const float3 &b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

__device__ void
operator/= (float3 &a, const float3 &b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

__device__ float3
operator+ (const float3 &a, const float3 &b)
{
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ void
operator+= (float3 &a, const float3 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

__device__ float3
operator- (const float3 &a, const float3 &b)
{
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ void
operator-= (float3 &a, const float3 &b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

__device__ float3
operator+ (const float3 &a, const float &c)
{
    return make_float3(a.x+c, a.y+c, a.z+c);
}

__device__ void
operator+= (float3 &a, const float &c)
{
    a.x += c;
    a.y += c;
    a.z += c;
}

__device__ float3
operator+ (const float &c, const float3 &a)
{
    return make_float3(c+a.x, c+a.y, c+a.z);
}

__device__ float3
operator- (const float3 &a, const float &c)
{
    return make_float3(a.x-c, a.y-c, a.z-c);
}

__device__ void
operator-= (float3 &a, const float &c)
{
    a.x -= c;
    a.y -= c;
    a.z -= c;
}

__device__ float3
operator- (const float &c, const float3& a)
{
    return make_float3(c-a.x, c-a.y, c-a.z);
}

__device__ float3
operator* (const float3 &a, const float &c)
{
    return make_float3(a.x*c, a.y*c, a.z*c);
}

__device__ void
operator*= (float3 &a, const float &c)
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
}

__device__ float3 
operator* (const float &c, const float3& a)
{
    return make_float3(c*a.x, c*a.y, c*a.z);
}

__device__ float3
operator/ (const float3 &a, const float &c)
{
    return make_float3(a.x/c, a.y/c, a.z/c);
}

__device__ void
operator/= (float3 &a, const float &c)
{
    a.x /= c;
    a.y /= c;
    a.z /= c;
}

__device__ float3
operator/ (const float &c, const float3 &a)
{
    return make_float3(c/a.x, c/a.y, c/a.z);
}

__device__ float
dot(const float3 &a, const float3 &b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ float3
cross(const float3 &a, const float3 &b)
{
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}

__device__ float3
abs(const float3&a)
{
    return make_float3(abs(a.x),abs(a.y),abs(a.z));
}

__device__ float
norm(const float3 &a)
{
    return sqrtf(dot(a,a));
}

__device__ float3
normalize(const float3 &a)
{
    return a/norm(a);
}

// float4 packing helpers for memory coalescing optimization
// pack position.xyz + wavelength into one float4
__device__ __forceinline__ float4
pack_pos_wl(const float3 &pos, float wavelength)
{
    return make_float4(pos.x, pos.y, pos.z, wavelength);
}

// pack direction.xyz + time into one float4
__device__ __forceinline__ float4
pack_dir_t(const float3 &dir, float time)
{
    return make_float4(dir.x, dir.y, dir.z, time);
}

// pack polarization.xyz + weight into one float4
__device__ __forceinline__ float4
pack_pol_w(const float3 &pol, float weight)
{
    return make_float4(pol.x, pol.y, pol.z, weight);
}

// unpack position from float4
__device__ __forceinline__ float3
unpack_pos(const float4 &packed)
{
    return make_float3(packed.x, packed.y, packed.z);
}

// unpack wavelength from float4
__device__ __forceinline__ float
unpack_wl(const float4 &packed)
{
    return packed.w;
}

// unpack direction from float4
__device__ __forceinline__ float3
unpack_dir(const float4 &packed)
{
    return make_float3(packed.x, packed.y, packed.z);
}

// unpack time from float4
__device__ __forceinline__ float
unpack_t(const float4 &packed)
{
    return packed.w;
}

// unpack polarization from float4
__device__ __forceinline__ float3
unpack_pol(const float4 &packed)
{
    return make_float3(packed.x, packed.y, packed.z);
}

// unpack weight from float4
__device__ __forceinline__ float
unpack_w(const float4 &packed)
{
    return packed.w;
}

#endif
