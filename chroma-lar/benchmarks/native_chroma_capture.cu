// Diagnostic wrapper around the reference installation's UNCHANGED headers.
// The host must compare this recorder against its ordinary propagate kernel;
// including the same source is not itself evidence that instrumentation is inert.
#include <curand_kernel.h>

typedef curandState NativeChromaRng;

struct CaptureRng {
    NativeChromaRng native;
    float *values;
    unsigned int *overflow;
    int row, interaction, draw, interactions, draws;
};

// random.h also declares initialization helpers; they are not launched by the
// recorder, but their redirected RNG type still needs a matching overload.
__device__ void curand_init(unsigned long long seed, unsigned long long sequence,
                           unsigned long long offset, CaptureRng *rng)
{
    curand_init(seed, sequence, offset, &rng->native);
}

__device__ float curand_uniform(CaptureRng *rng)
{
    const float value = curand_uniform(&rng->native);
    if (rng->draw < rng->draws && rng->interaction < rng->interactions) {
        const unsigned long long offset =
            ((unsigned long long)rng->row * rng->interactions + rng->interaction)
            * rng->draws + rng->draw;
        rng->values[offset] = value;
    } else {
        rng->overflow[rng->row] = 1u;
    }
    ++rng->draw;
    return value;
}

// Redirect only the RNG type; the reference's arithmetic and functions are
// included verbatim. curand_kernel.h was included before this macro.
#define curandState CaptureRng
#include "photon.h"
#undef curandState

__device__ void write_photon_words(const Photon &p, unsigned int *words)
{
    words[0] = __float_as_uint(p.position.x);
    words[1] = __float_as_uint(p.position.y);
    words[2] = __float_as_uint(p.position.z);
    words[3] = __float_as_uint(p.direction.x);
    words[4] = __float_as_uint(p.direction.y);
    words[5] = __float_as_uint(p.direction.z);
    words[6] = __float_as_uint(p.polarization.x);
    words[7] = __float_as_uint(p.polarization.y);
    words[8] = __float_as_uint(p.polarization.z);
    words[9] = __float_as_uint(p.wavelength);
    words[10] = __float_as_uint(p.time);
    words[11] = p.history;
    words[12] = (unsigned int)p.last_hit_triangle;
    words[13] = __float_as_uint(p.weight);
    words[14] = p.evidx;
}

__device__ void commit_capture(
    const Photon &p, CaptureRng &rng, unsigned int *state_words, int *draw_counts)
{
    const unsigned long long record =
        (unsigned long long)rng.row * rng.interactions + rng.interaction;
    write_photon_words(p, state_words + record * 15);
    draw_counts[record] = rng.draw;
    ++rng.interaction;
    rng.draw = 0;
}

extern "C" __global__ void capture_original_chroma(
    int count, NativeChromaRng *native_states,
    float3 *positions, float3 *directions, float *wavelengths,
    float3 *polarizations, float *times, unsigned int *histories,
    int *last_hit_triangles, float *weights, unsigned int *evidx,
    int max_steps, int use_weights, int scatter_first, Geometry *geometry,
    float *random_words, unsigned int *initial_words,
    unsigned int *state_words, int *draw_counts, int *interaction_counts,
    unsigned int *overflow, int draws_per_interaction)
{
    __shared__ Geometry shared_geometry;
    if (threadIdx.x == 0)
        shared_geometry = *geometry;
    __syncthreads();
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= count)
        return;
    geometry = &shared_geometry;

    CaptureRng rng;
    rng.native = native_states[id];
    rng.values = random_words;
    rng.overflow = overflow;
    rng.row = id;
    rng.interaction = rng.draw = 0;
    rng.interactions = max_steps;
    rng.draws = draws_per_interaction;

    Photon p;
    p.position = positions[id];
    p.direction = directions[id];
    p.direction /= norm(p.direction);
    p.polarization = polarizations[id];
    p.polarization /= norm(p.polarization);
    p.wavelength = wavelengths[id];
    p.time = times[id];
    p.history = histories[id];
    p.last_hit_triangle = last_hit_triangles[id];
    p.weight = weights[id];
    p.evidx = evidx[id];
    write_photon_words(p, initial_words + (unsigned long long)id * 15);

    if (p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT))
        return;
    State state;
    int step = 0;
    while (step < max_steps) {
        ++step;
        if (isnan(p.direction.x*p.direction.y*p.direction.z*
                  p.position.x*p.position.y*p.position.z)) {
            p.history |= NO_HIT | NAN_ABORT;
            commit_capture(p, rng, state_words, draw_counts);
            break;
        }
        fill_state(state, p, geometry);
        if (p.last_hit_triangle == -1) {
            commit_capture(p, rng, state_words, draw_counts);
            break;
        }
        int command = propagate_to_boundary(p, state, rng, use_weights, scatter_first);
        scatter_first = 0;
        if (command == BREAK || command == CONTINUE) {
            commit_capture(p, rng, state_words, draw_counts);
            if (command == BREAK) break;
            continue;
        }
        if (state.surface_index != -1) {
            command = propagate_at_surface(p, state, rng, geometry, use_weights);
            if (command == BREAK || command == CONTINUE) {
                commit_capture(p, rng, state_words, draw_counts);
                if (command == BREAK) break;
                continue;
            }
        }
        propagate_at_boundary(p, state, rng);
        commit_capture(p, rng, state_words, draw_counts);
    }
    native_states[id] = rng.native;
    positions[id] = p.position;
    directions[id] = p.direction;
    polarizations[id] = p.polarization;
    wavelengths[id] = p.wavelength;
    times[id] = p.time;
    histories[id] = p.history;
    last_hit_triangles[id] = p.last_hit_triangle;
    weights[id] = p.weight;
    evidx[id] = p.evidx;
    interaction_counts[id] = rng.interaction;
}
