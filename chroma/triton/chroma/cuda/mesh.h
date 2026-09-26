#ifndef __MESH_H__
#define __MESH_H__

#include "intersect.h"
#include "geometry.h"
#include "profile.h"

#include "stdio.h"

#define STACK_SIZE 1000

/* Tests the intersection between a ray and a node in the bounding volume
   hierarchy. If the ray intersects the bounding volume and `min_distance`
   is less than zero or the distance from `origin` to the intersection is
   less than `min_distance`, return true, else return false. */
__device__ bool
intersect_node(const float3 &neg_origin_inv_dir, const float3 &inv_dir,
	       const Geometry *g, const Node &node, const float min_distance=-1.0f)
{
    CHROMA_PROF_FUNC_START(CHROMA_PROF_INTERSECT_NODE);
    float distance_to_box;
    bool result = false;

    if (intersect_box(neg_origin_inv_dir, inv_dir, node.lower, node.upper,
		      distance_to_box)) {
	if (min_distance < 0.0f)
	    result = true;
	else if (distance_to_box > min_distance)
	    result = false;
	else
	    result = true;
    } else {
	result = false;
    }

    CHROMA_PROF_FUNC_END(CHROMA_PROF_INTERSECT_NODE);
    return result;
}

/* Finds the intersection between a ray and `geometry`. If the ray does
   intersect the mesh and the index of the intersected triangle is not equal
   to `last_hit_triangle`, set `min_distance` to the distance from `origin` to
   the intersection and return the index of the triangle which the ray
   intersected, else return -1. */
__device__ int
intersect_mesh(const float3 &origin, const float3& direction, const Geometry *g,
	       float &min_distance, int last_hit_triangle = -1)
{
    CHROMA_PROF_FUNC_START(CHROMA_PROF_INTERSECT_MESH);
    int triangle_index = -1;

    float distance;
    min_distance = -1.0f;

    Node root = get_node(g, 0);

    float3 neg_origin_inv_dir = -origin / direction;
    float3 inv_dir = 1.0f / direction;

    if (!intersect_node(neg_origin_inv_dir, inv_dir, g, root, min_distance)) {
        CHROMA_PROF_FUNC_END(CHROMA_PROF_INTERSECT_MESH);
	return -1;
    }

    unsigned int child_ptr_stack[STACK_SIZE];
    unsigned int nchild_ptr_stack[STACK_SIZE];
    child_ptr_stack[0] = root.child;
    nchild_ptr_stack[0] = root.nchild;

    int curr = 0;

    unsigned int count = 0;
    unsigned int tri_count = 0;

    while (curr >= 0) {
	unsigned int first_child = child_ptr_stack[curr];
	unsigned int nchild = nchild_ptr_stack[curr];
	curr--;

	for (unsigned int i=first_child; i < first_child + nchild; i++) {
	    Node node = get_node(g, i);
	    count++;

	    if (intersect_node(neg_origin_inv_dir, inv_dir, g, node, min_distance)) {

		if (node.nchild == 0) { /* leaf node */
		    // This node wraps a triangle

		    if (node.child != last_hit_triangle) {
			// Can't hit same triangle twice in a row
			tri_count++;
			Triangle t = get_triangle(g, node.child);			
			if (intersect_triangle(origin, direction, t, distance)) {

			    if (triangle_index == -1 || distance < min_distance) {
				triangle_index = node.child;
				min_distance = distance;
			    } // if hit triangle is closer than previous hits

			} // if hit triangle
			
		    } // if not hitting same triangle as last step

		} else {
		    curr++;
		    child_ptr_stack[curr] = node.child;
		    nchild_ptr_stack[curr] = node.nchild;
		} // leaf or internal node?
	    } // hit node?
	    
	    if (curr >= STACK_SIZE) {
	    	printf("warning: intersect_mesh() aborted; node > tail\n");
	    	break;
	    }
	} // loop over children, starting with first_child

    } // while nodes on stack

    //if (blockIdx.x == 0 && threadIdx.x == 0) {
    //  printf("node count: %d\n", count);
    //  printf("triangle count: %d\n", tri_count);
    //}

    CHROMA_PROF_FUNC_END(CHROMA_PROF_INTERSECT_MESH);
    return triangle_index;
}

extern "C"
{

__global__ void
distance_to_mesh(int nthreads, float3 *_origin, float3 *_direction,
		 const Geometry *g, float *_distance)
{
    __shared__ Geometry sg;

    if (threadIdx.x == 0)
	sg = *g;

    __syncthreads();

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    g = &sg;

    float3 origin = _origin[id];
    float3 direction = _direction[id];
    direction /= norm(direction);

    float distance;

    int triangle_index = intersect_mesh(origin, direction, g, distance);

    if (triangle_index != -1)
	_distance[id] = distance;
}

__global__ void
color_solids(int first_triangle, int nthreads, int *solid_id_map,
	     bool *solid_hit, unsigned int *solid_colors, Geometry *g)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    int triangle_id = first_triangle + id;
    int solid_id = solid_id_map[triangle_id];
    if (solid_hit[solid_id])
	g->colors[triangle_id] = solid_colors[solid_id];
}

} // extern "C"

#endif
