//-*-c-*-

#include "linalg.h"
#include "geometry.h"
#include "detector.h"
#include "photon.h"
#include "profile.h"

#include "stdio.h"

#ifdef CHROMA_DEVICE_PROFILE
// Device-global counters for inner-kernel profiling
extern "C" __device__ unsigned long long chroma_prof_calls[CHROMA_PROF_COUNT];
extern "C" __device__ unsigned long long chroma_prof_cycles[CHROMA_PROF_COUNT];

extern "C" __global__ void chroma_prof_reset()
{
    for (int i = threadIdx.x; i < CHROMA_PROF_COUNT; i += blockDim.x) {
        chroma_prof_calls[i] = 0ULL;
        chroma_prof_cycles[i] = 0ULL;
    }
}
#endif

extern "C"
{


__global__ void
photon_duplicate(int first_photon, int nthreads,
		 float3 *__restrict__ positions, float3 *__restrict__ directions,
		 float *__restrict__ wavelengths, float3 *__restrict__ polarizations,
		 float *__restrict__ times, unsigned int *__restrict__ histories,
		 int *__restrict__ last_hit_triangles, float *__restrict__ weights, unsigned int *__restrict__ evidx,
		 int copies, int stride)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    int photon_id = first_photon + id;

    Photon p;
    p.position = positions[photon_id];
    p.direction = directions[photon_id];
    p.polarization = polarizations[photon_id];
    p.wavelength = wavelengths[photon_id];
    p.time = times[photon_id];
    p.last_hit_triangle = last_hit_triangles[photon_id];
    p.history = histories[photon_id];
    p.weight = weights[photon_id];
    p.evidx = evidx[photon_id];

    for (int i=1; i <= copies; i++) {
      int target_photon_id = photon_id + stride * i;

      positions[target_photon_id] = p.position;
      directions[target_photon_id] = p.direction;
      polarizations[target_photon_id] = p.polarization;
      wavelengths[target_photon_id] = p.wavelength;
      times[target_photon_id] = p.time;
      last_hit_triangles[target_photon_id] = p.last_hit_triangle;
      histories[target_photon_id] = p.history;
      weights[target_photon_id] = p.weight;
      evidx[target_photon_id] = p.evidx;
    }
}

__global__ void
count_photons(int first_photon, int nthreads, unsigned int target_flag,
	      unsigned int *__restrict__ index_counter,
	      unsigned int *__restrict__ histories)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ unsigned int counter;

    if (threadIdx.x == 0)
	counter = 0;
    __syncthreads();

    if (id < nthreads) {
	int photon_id = first_photon + id;

	if (histories[photon_id] & target_flag) {
	    atomicAdd(&counter, 1);
	}
	    
    }

    __syncthreads();

    if (threadIdx.x == 0)
	atomicAdd(index_counter, counter);
}

__global__ void
copy_photons(int first_photon, int nthreads, unsigned int target_flag,
	     unsigned int *__restrict__ index_counter,
	     const float3 *__restrict__ positions, const float3 *__restrict__ directions,
	     const float *__restrict__ wavelengths, const float3 *__restrict__ polarizations,
	     const float *__restrict__ times, const unsigned int *__restrict__ histories,
	     const int *__restrict__ last_hit_triangles, const float *__restrict__ weights, const unsigned int *__restrict__ evidx,
	     float3 *__restrict__ new_positions, float3 *__restrict__ new_directions,
	     float *__restrict__ new_wavelengths, float3 *__restrict__ new_polarizations,
	     float *__restrict__ new_times, unsigned int *__restrict__ new_histories,
	     int *__restrict__ new_last_hit_triangles, float *__restrict__ new_weights, unsigned int *__restrict__ new_evidx)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (id >= nthreads)
	return;
    
    int photon_id = first_photon + id;

    unsigned int alive = (histories[photon_id] & target_flag) ? 1u : 0u;
    unsigned int mask = __ballot_sync(0xffffffff, alive);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(mask);
    int warp_prefix = __popc(mask & ((1u << lane) - 1u));
    unsigned int base = 0;
    if (lane == 0 && warp_count > 0) {
        base = atomicAdd(index_counter, (unsigned int)warp_count);
    }
    base = __shfl_sync(0xffffffff, base, 0);

    if (alive) {
        int offset = base + warp_prefix;
        new_positions[offset] = positions[photon_id];
        new_directions[offset] = directions[photon_id];
        new_polarizations[offset] = polarizations[photon_id];
        new_wavelengths[offset] = wavelengths[photon_id];
        new_times[offset] = times[photon_id];
        new_histories[offset] = histories[photon_id];
        new_last_hit_triangles[offset] = last_hit_triangles[photon_id];
        new_weights[offset] = weights[photon_id];
        new_evidx[offset] = evidx[photon_id];
    }
}

__global__ void
copy_photon_queue(int first_photon, int nthreads, const unsigned int *__restrict__ queue,
	     const float3 *__restrict__ positions, const float3 *__restrict__ directions,
	     const float *__restrict__ wavelengths, const float3 *__restrict__ polarizations,
	     const float *__restrict__ times, const unsigned int *__restrict__ histories,
	     const int *__restrict__ last_hit_triangles, const float *__restrict__ weights, const unsigned int *__restrict__ evidx,
	     float3 *__restrict__ new_positions, float3 *__restrict__ new_directions,
	     float *__restrict__ new_wavelengths, float3 *__restrict__ new_polarizations,
	     float *__restrict__ new_times, unsigned int *__restrict__ new_histories,
	     int *__restrict__ new_last_hit_triangles, float *__restrict__ new_weights, unsigned int *__restrict__ new_evidx)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (id >= nthreads)
	return;
    
    int offset = first_photon + id;
    int photon_id = queue[offset];

    new_positions[offset] = positions[photon_id];
    new_directions[offset] = directions[photon_id];
    new_polarizations[offset] = polarizations[photon_id];
    new_wavelengths[offset] = wavelengths[photon_id];
    new_times[offset] = times[photon_id];
    new_histories[offset] = histories[photon_id];
    new_last_hit_triangles[offset] = last_hit_triangles[photon_id];
    new_weights[offset] = weights[photon_id];
    new_evidx[offset] = evidx[photon_id];
}


__global__ void
count_photon_hits(int first_photon, int nphotons, unsigned int detection_state,
            const unsigned int *__restrict__ histories, const int *__restrict__ solid_map, const int *__restrict__ last_hit_triangles,
            const Detector *__restrict__ detector, unsigned int *__restrict__ index_counter)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ unsigned int counter;

    if (threadIdx.x == 0) counter = 0;
    
    __syncthreads();

    if (id < nphotons) {
	    int photon_id = first_photon + id;
	    if (histories[photon_id] & detection_state) {
	        int triangle_id = last_hit_triangles[photon_id];
        	if (triangle_id > -1) {
	            int solid_id = solid_map[triangle_id];
	            int channel_index = detector->solid_id_to_channel_index[solid_id];
	            if (channel_index >= 0) atomicAdd(&counter, 1);
	        }
	    }
    }
    
    __syncthreads();

    if (threadIdx.x == 0) atomicAdd(index_counter, counter);
}

__global__ void
copy_photon_hits(int first_photon, int nphotons, unsigned int detection_state,
            const int *__restrict__ solid_map, const Detector *__restrict__ detector, unsigned int *__restrict__ index_counter,
            const float3 *__restrict__ positions, const float3 *__restrict__ directions,
            const float *__restrict__ wavelengths, const float3 *__restrict__ polarizations,
            const float *__restrict__ times, const unsigned int *__restrict__ histories,
            const int *__restrict__ last_hit_triangles, const float *__restrict__ weights, const unsigned int *__restrict__ evidx,
            float3 *__restrict__ new_positions, float3 *__restrict__ new_directions,
            float *__restrict__ new_wavelengths, float3 *__restrict__ new_polarizations,
            float *__restrict__ new_times, unsigned int *__restrict__ new_histories,
            int *__restrict__ new_last_hit_triangles, float *__restrict__ new_weights, unsigned int *__restrict__ new_evidx,
            int *__restrict__ new_channels)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    
    if (id < nphotons) {
        int photon_id = first_photon + id;
        int triangle_id = last_hit_triangles[photon_id];
        int channel_index = -1;
        unsigned int hit = 0u;
        if ((histories[photon_id] & detection_state) && triangle_id > -1) {
            int solid_id = solid_map[triangle_id];
            channel_index = detector->solid_id_to_channel_index[solid_id];
            hit = (channel_index >= 0) ? 1u : 0u;
        }

        unsigned int mask = __ballot_sync(0xffffffff, hit);
        int lane = threadIdx.x & 31;
        int warp_count = __popc(mask);
        int warp_prefix = __popc(mask & ((1u << lane) - 1u));
        unsigned int base = 0;
        if (lane == 0 && warp_count > 0) {
            base = atomicAdd(index_counter, (unsigned int)warp_count);
        }
        base = __shfl_sync(0xffffffff, base, 0);

        if (hit) {
            int offset = base + warp_prefix;
            new_positions[offset] = positions[photon_id];
            new_directions[offset] = directions[photon_id];
            new_polarizations[offset] = polarizations[photon_id];
            new_wavelengths[offset] = wavelengths[photon_id];
            new_times[offset] = times[photon_id];
            new_histories[offset] = histories[photon_id];
            new_last_hit_triangles[offset] = last_hit_triangles[photon_id];
            new_weights[offset] = weights[photon_id];
            new_evidx[offset] = evidx[photon_id];
            new_channels[offset] = channel_index;
        }
    }
}

	      
__global__ void
propagate(int first_photon, int nthreads, const unsigned int *__restrict__ input_queue,
	  unsigned int *__restrict__ output_queue, curandState *rng_states,
	  float3 *__restrict__ positions, float3 *__restrict__ directions,
	  float *__restrict__ wavelengths, float3 *__restrict__ polarizations,
	  float *__restrict__ times, unsigned int *__restrict__ histories,
	  int *__restrict__ last_hit_triangles, float *__restrict__ weights, unsigned int *__restrict__ evidx,
	  int max_steps, int use_weights, int scatter_first,
	  Geometry *g)
{
    __shared__ Geometry sg;

    if (threadIdx.x == 0)
	sg = *g;

    __syncthreads();

    int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= nthreads)
	return;

    g = &sg;

    curandState rng = rng_states[id];

    int photon_id = input_queue[first_photon + id];

    Photon p;
    p.position = positions[photon_id];
    p.direction = directions[photon_id];
    p.direction /= norm(p.direction);
    p.polarization = polarizations[photon_id];
    p.polarization /= norm(p.polarization);
    p.wavelength = wavelengths[photon_id];
    p.time = times[photon_id];
    p.last_hit_triangle = last_hit_triangles[photon_id];
    p.history = histories[photon_id];
    p.weight = weights[photon_id];
    p.evidx = evidx[photon_id];

    if (p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT))
	return;

    State s;

    int steps = 0;
    while (steps < max_steps) {
	steps++;

	int command;

	// check for NaN and fail
	if (isnan(p.direction.x*p.direction.y*p.direction.z*p.position.x*p.position.y*p.position.z)) {
	    p.history |= NO_HIT | NAN_ABORT;
	    break;
	}

	fill_state(s, p, g);

	if (p.last_hit_triangle == -1)
	    break;

	command = propagate_to_boundary(p, s, rng, use_weights, scatter_first);
	scatter_first = 0; // Only use the scatter_first value once

	if (command == BREAK)
	    break;

	if (command == CONTINUE)
	    continue;

	if (s.surface_index != -1) {
	  command = propagate_at_surface(p, s, rng, g, use_weights);

	    if (command == BREAK)
		break;

	    if (command == CONTINUE)
		continue;
	}

	propagate_at_boundary(p, s, rng);

    } // while (steps < max_steps)

    rng_states[id] = rng;
    positions[photon_id] = p.position;
    directions[photon_id] = p.direction;
    polarizations[photon_id] = p.polarization;
    wavelengths[photon_id] = p.wavelength;
    times[photon_id] = p.time;
    histories[photon_id] = p.history;
    last_hit_triangles[photon_id] = p.last_hit_triangle;
    weights[photon_id] = p.weight;
    evidx[photon_id] = p.evidx;

    // Not done, put photon in output queue using warp-aggregated atomics
    unsigned int still_alive = ((p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT)) == 0) ? 1u : 0u;
    unsigned int mask = __ballot_sync(0xffffffff, still_alive);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(mask);
    int warp_prefix = __popc(mask & ((1u << lane) - 1u));
    unsigned int base = 0;
    if (lane == 0 && warp_count > 0) {
        base = atomicAdd(output_queue, (unsigned int)warp_count);
    }
    base = __shfl_sync(0xffffffff, base, 0);
    if (still_alive) {
        int out_idx = base + warp_prefix;
        output_queue[out_idx] = photon_id;
    }
} // propagate

} // extern "C"
