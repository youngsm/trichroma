#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "geometry_types.h"
#include "linalg.h"

__device__ float3
to_float3(const uint3 &a)
{
    return make_float3(a.x, a.y, a.z);
}

__device__ uint4
get_packed_node(const Geometry *geometry, const unsigned int &i)
{
    if (i < geometry->nprimary_nodes)
        return geometry->primary_nodes[i];
    else
        return geometry->extra_nodes[i - geometry->nprimary_nodes];
}
__device__ void
put_packed_node(Geometry *geometry, const unsigned int &i, const uint4 &node)
{
    if (i < geometry->nprimary_nodes)
        geometry->primary_nodes[i] = node;
    else
        geometry->extra_nodes[i - geometry->nprimary_nodes] = node;
}

__device__ Node
get_node(const Geometry *geometry, const unsigned int &i)
{
    uint4 node = get_packed_node(geometry, i);

    Node node_struct;

    uint3 lower_int = make_uint3(node.x & 0xFFFF, node.y & 0xFFFF, node.z & 0xFFFF);
    uint3 upper_int = make_uint3(node.x >> 16, node.y >> 16, node.z >> 16);

    node_struct.lower = geometry->world_origin + to_float3(lower_int) * geometry->world_scale;
    node_struct.upper = geometry->world_origin + to_float3(upper_int) * geometry->world_scale;
    node_struct.child = node.w & ~NCHILD_MASK;
    node_struct.nchild = node.w >> CHILD_BITS;

    return node_struct;
}

__device__ Triangle
get_triangle(const Geometry *geometry, const unsigned int &i)
{
    uint3 triangle_data = geometry->triangles[i];

    Triangle triangle;
    triangle.v0 = geometry->vertices[triangle_data.x];
    triangle.v1 = geometry->vertices[triangle_data.y];
    triangle.v2 = geometry->vertices[triangle_data.z];

    return triangle;
}

template <class T>
__device__ float
interp_property(T *m, const float &x, const float *fp)
{
    if (x < m->wavelength_start)
        return fp[0];

    if (x > (m->wavelength_start + (m->wavelength_n - 1) * m->wavelength_step))
        return fp[m->wavelength_n - 1];

    int jl = (x - m->wavelength_start) / m->wavelength_step;

    return fp[jl] + (x - (m->wavelength_start + jl * m->wavelength_step)) * (fp[jl + 1] - fp[jl]) / m->wavelength_step;
}

#endif
