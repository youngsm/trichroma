//-*-c-*-
#include <math_constants.h>
#include <curand_kernel.h>

#include "linalg.h"
#include "matrix.h"
#include "rotate.h"
#include "mesh.h"
#include "geometry.h"
#include "photon.h"

__device__ void
fAtomicAdd(float *addr, float data)
{
    while (data)
	data = atomicExch(addr, data+atomicExch(addr, 0.0f));
}

__device__ void
to_diffuse(Photon &p, State &s, Geometry *g, curandState &rng, int max_steps)
{
    int steps = 0;
    while (steps < max_steps) {
	steps++;

	int command;

	fill_state(s, p, g);

	if (p.last_hit_triangle == -1)
	    break;

	command = propagate_to_boundary(p, s, rng);

	if (command == BREAK)
	    break;

	if (command == CONTINUE)
	    continue;

	if (s.surface_index != -1) {
	    command = propagate_at_surface(p, s, rng, g);

	    if (p.history & REFLECT_DIFFUSE)
		break;

	    if (command == BREAK)
		break;

	    if (command == CONTINUE)
		continue;
	}

	propagate_at_boundary(p, s, rng);

    } // while (steps < max_steps)

} // to_diffuse

extern "C"
{

__global__ void
update_xyz_lookup(int nthreads, int total_threads, int offset, float3 position,
		  curandState *rng_states, float wavelength, float3 xyz,
		  float3 *xyz_lookup1, float3 *xyz_lookup2, int max_steps,
		  Geometry *g)
{
    int kernel_id = blockIdx.x*blockDim.x + threadIdx.x;
    int id = kernel_id + offset;

    if (kernel_id >= nthreads || id >= total_threads)
	return;

    curandState rng = rng_states[kernel_id];

    Triangle t = get_triangle(g, id);

    float a = curand_uniform(&rng);
    float b = uniform(&rng, 0.0f, (1.0f - a));
    float c = 1.0f - a - b;

    float3 direction = a*t.v0 + b*t.v1 + c*t.v2 - position;
    direction /= norm(direction);

    float distance;
    int triangle_index = intersect_mesh(position, direction, g, distance);

    if (triangle_index != id) {
	rng_states[kernel_id] = rng;
	return;
    }

    float3 v01 = t.v1 - t.v0;
    float3 v12 = t.v2 - t.v1;
    
    float3 surface_normal = normalize(cross(v01,v12));

    float cos_theta = dot(surface_normal,-direction);

    if (cos_theta < 0.0f)
	cos_theta = dot(-surface_normal,-direction);

    Photon p;
    p.position = position;
    p.direction = direction;
    p.wavelength = wavelength;
    p.polarization = uniform_sphere(&rng);
    p.last_hit_triangle = -1;
    p.time = 0;
    p.history = 0;

    State s;
    to_diffuse(p, s, g, rng, max_steps);

    if (p.history & REFLECT_DIFFUSE) {
	if (s.inside_to_outside) {
	    fAtomicAdd(&xyz_lookup1[p.last_hit_triangle].x, cos_theta*xyz.x);
	    fAtomicAdd(&xyz_lookup1[p.last_hit_triangle].y, cos_theta*xyz.y);
	    fAtomicAdd(&xyz_lookup1[p.last_hit_triangle].z, cos_theta*xyz.z);
	}
	else {
	    fAtomicAdd(&xyz_lookup2[p.last_hit_triangle].x, cos_theta*xyz.x);
	    fAtomicAdd(&xyz_lookup2[p.last_hit_triangle].y, cos_theta*xyz.y);
	    fAtomicAdd(&xyz_lookup2[p.last_hit_triangle].z, cos_theta*xyz.z);
	}
    }

    rng_states[kernel_id] = rng;

} // update_xyz_lookup

__global__ void
update_xyz_image(int nthreads, curandState *rng_states, float3 *positions,
		 float3 *directions, float wavelength, float3 xyz,
		 float3 *xyz_lookup1, float3 *xyz_lookup2, float3 *image,
		 int nlookup_calls, int max_steps, Geometry *g)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    curandState rng = rng_states[id];

    Photon p;
    p.position = positions[id];
    p.direction = directions[id];
    p.direction /= norm(p.direction);
    p.wavelength = wavelength;
    p.polarization = uniform_sphere(&rng);
    p.last_hit_triangle = -1;
    p.time = 0;
    p.history = 0;

    State s;
    to_diffuse(p, s, g, rng, max_steps);

    if (p.history & REFLECT_DIFFUSE) {
	if (s.inside_to_outside)
	    image[id] += xyz*xyz_lookup1[p.last_hit_triangle]/nlookup_calls;
	else
	    image[id] += xyz*xyz_lookup2[p.last_hit_triangle]/nlookup_calls;
    }

    rng_states[id] = rng;

} // update_xyz_image

__global__ void
process_image(int nthreads, float3 *image, unsigned int *pixels, int nimages)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    float3 rgb = image[id]/nimages;

    if (rgb.x < 0.0f)
	rgb.x = 0.0f;
    if (rgb.y < 0.0f)
	rgb.y = 0.0f;
    if (rgb.z < 0.0f)
	rgb.z = 0.0f;

    if (rgb.x > 1.0f)
	rgb.x = 1.0f;
    if (rgb.y > 1.0f)
	rgb.y = 1.0f;
    if (rgb.z > 1.0f)
	rgb.z = 1.0f;

    unsigned int r = floorf(rgb.x*255.0f);
    unsigned int g = floorf(rgb.y*255.0f);
    unsigned int b = floorf(rgb.z*255.0f);

    pixels[id] = 255 << 24 | r << 16 | g << 8 | b;

} // process_image

} // extern "c"
