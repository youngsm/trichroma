//-*-c-*-

// Opt-in lockstep version of propagate().  This is deliberately a separate
// module and symbol: production Chroma continues to compile propagate.cu with
// curandState/XORWOW and its Python launch signature is unchanged.
//
// One tape interaction is one completed legacy transport-loop iteration.
// Draws retain Chroma's call order: bulk absorption/scattering distances are
// slots 0/1; a Rayleigh winner uses 2/3; an explicit surface starts at 2; a
// surface PASS is followed by dielectric polarization/reflection at 3/4,
// while a surface-free dielectric uses 2/3.  Diffuse reflection consumes
// (sphere theta, sphere z, acceptance) triples followed by two polarization
// sphere draws.  All variable-length paths fail closed on bounded overflow.
#define CHROMA_USE_RANDOM_TAPE 1

#include "linalg.h"
#include "geometry.h"
#include "detector.h"
#include "photon.h"
#include "rng_alignment.h"

namespace
{

__device__ __forceinline__ void
record_first_tape_error(int photon_id,
                        unsigned int flags,
                        int interaction,
                        int draw,
                        int stage,
                        int *first_error_photon_id,
                        unsigned int *first_error_flags,
                        int *first_error_interaction,
                        int *first_error_draw,
                        int *first_error_stage)
{
    if (flags == 0u || first_error_photon_id == 0 || first_error_flags == 0)
        return;
    if (atomicCAS(first_error_photon_id, -1, photon_id) == -1) {
        *first_error_flags = flags;
        *first_error_interaction = interaction;
        *first_error_draw = draw;
        *first_error_stage = stage;
    }
}

__device__ __forceinline__ void
record_committed_interaction(const ChromaRandomTapeState &state,
                             unsigned int *certificate,
                             int enabled)
{
    if (!enabled || !state.mapping_valid || state.overflow != 0u ||
        state.interaction < 0 ||
        state.interaction >= state.tape.max_interactions)
        return;
    // Keep this layout synchronized with triton/rng_alignment.py.  Process
    // code 15 is reserved for the host-initialized 0xffffffff empty word.
    const unsigned int process = (unsigned int)state.process;
    const unsigned int draws = (unsigned int)state.draw;
    const unsigned int word = (process << 28) | (draws & 0x0fffffffu);
    const unsigned long long offset =
        (unsigned long long)state.row * state.tape.max_interactions +
        (unsigned long long)state.interaction;
    certificate[offset] = word;
}

// Intrinsic raw Photon words, deliberately excluding detector-derived
// channel/boundary labels.  Keep the order synchronized with
// triton/rng_alignment.py:STATE_CERTIFICATE_FIELDS.
enum { CHROMA_TAPE_STATE_FIELD_COUNT = 15 };

__device__ __forceinline__ void
record_committed_state(const ChromaRandomTapeState &state,
                       const Photon &p,
                       unsigned int *certificate,
                       int enabled)
{
    if (!enabled || !state.mapping_valid || state.overflow != 0u ||
        state.interaction < 0 ||
        state.interaction >= state.tape.max_interactions)
        return;
    const unsigned long long record =
        (unsigned long long)state.row * state.tape.max_interactions +
        (unsigned long long)state.interaction;
    unsigned int *words =
        certificate + record * CHROMA_TAPE_STATE_FIELD_COUNT;
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
    words[11] = (unsigned int)p.history;
    words[12] = (unsigned int)p.last_hit_triangle;
    words[13] = __float_as_uint(p.weight);
    words[14] = p.evidx;
}

} // anonymous namespace

extern "C"
{

// Export the exact input normalization performed at the head of
// propagate_tape().  The cross-runtime lockstep harness feeds these words to
// Triton so the comparison begins *after* Chroma's launch-time normalization,
// rather than attributing a setup convention to transport physics.
__global__ void
propagate_tape_normalize_inputs(
    int count,
    const float3 *__restrict__ directions,
    const float3 *__restrict__ polarizations,
    float3 *__restrict__ normalized_directions,
    float3 *__restrict__ normalized_polarizations)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    float3 direction = directions[index];
    float3 polarization = polarizations[index];
    direction /= norm(direction);
    polarization /= norm(polarization);
    normalized_directions[index] = direction;
    normalized_polarizations[index] = polarization;
}

__global__ void
propagate_tape_debug_geometry_size(unsigned long long *size)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *size = (unsigned long long)sizeof(Geometry);
}

// Focused raw-word oracle for host tests.  This proves that the state writer
// preserves signed zero and non-integral IEEE-754 payloads rather than using
// numeric float-to-uint conversion.
__global__ void
propagate_tape_debug_write_state_certificate(
    const unsigned int *__restrict__ input_words,
    unsigned int *__restrict__ output_words)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    Photon p;
    p.position = make_float3(
        __uint_as_float(input_words[0]),
        __uint_as_float(input_words[1]),
        __uint_as_float(input_words[2]));
    p.direction = make_float3(
        __uint_as_float(input_words[3]),
        __uint_as_float(input_words[4]),
        __uint_as_float(input_words[5]));
    p.polarization = make_float3(
        __uint_as_float(input_words[6]),
        __uint_as_float(input_words[7]),
        __uint_as_float(input_words[8]));
    p.wavelength = __uint_as_float(input_words[9]);
    p.time = __uint_as_float(input_words[10]);
    p.history = (unsigned short)input_words[11];
    p.last_hit_triangle = (int)input_words[12];
    p.weight = __uint_as_float(input_words[13]);
    p.evidx = input_words[14];

    ChromaRandomTapeState state;
    state.tape.max_interactions = 1;
    state.row = 0;
    state.interaction = 0;
    state.overflow = 0u;
    state.mapping_valid = true;
    record_committed_state(state, p, output_words, 1);
}

__global__ void
propagate_tape_init_no_hit_geometry(Geometry *g, uint4 *root)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
        return;
    root[0] = make_uint4(1u << 16, 1u << 16, 1u << 16, 0u);
    g->vertices = 0;
    g->triangles = 0;
    g->material_codes = 0;
    g->colors = 0;
    g->primary_nodes = root;
    g->extra_nodes = 0;
    g->materials = 0;
    g->surfaces = 0;
    g->wireplanes = 0;
    g->world_origin = make_float3(0.0f, 0.0f, 0.0f);
    g->world_scale = 1.0f;
    g->nprimary_nodes = 1;
    g->nwireplanes = 0;
}

// Queue-resolution probe used by the Python lockstep adapter before any
// geometry/physics work.  It proves that worker reordering cannot change the
// random row selected for a photon and exercises the same sticky error path
// as propagate_tape().
__global__ void
propagate_tape_mapping_probe(
    int first_photon,
    int nthreads,
    const unsigned int *__restrict__ input_queue,
    const int *__restrict__ photon_tape_rows,
    const long long *__restrict__ photon_global_ids,
    const long long *__restrict__ tape_global_photon_ids,
    int tape_photon_count,
    int tape_max_interactions,
    int tape_draws_per_interaction,
    int *__restrict__ interaction_cursor,
    int *__restrict__ draw_cursor,
    unsigned int *__restrict__ tape_overflow,
    int *__restrict__ trace_stage,
    int *__restrict__ first_error_photon_id,
    unsigned int *__restrict__ first_error_flags,
    int *__restrict__ first_error_interaction,
    int *__restrict__ first_error_draw,
    int *__restrict__ first_error_stage)
{
    int worker = blockIdx.x*blockDim.x + threadIdx.x;
    if (worker >= nthreads)
        return;
    int photon_id = input_queue[first_photon + worker];
    int row = photon_tape_rows[photon_id];
    bool row_in_range = row >= 0 && row < tape_photon_count;

    ChromaRandomTapeView tape;
    tape.values = 0; // This probe audits mapping only and performs no load.
    tape.global_photon_ids = tape_global_photon_ids;
    tape.photon_count = tape_photon_count;
    tape.max_interactions = tape_max_interactions;
    tape.draws_per_interaction = tape_draws_per_interaction;

    int interaction = row_in_range ? interaction_cursor[row] : 0;
    int draw = row_in_range ? draw_cursor[row] : 0;
    unsigned int flags = row_in_range ? tape_overflow[row] : 0u;
    ChromaRandomTapeState state = chroma_random_tape_init(
        tape, row, photon_global_ids[photon_id], interaction, draw, flags);
    chroma_random_tape_commit(
        state, interaction_cursor, draw_cursor, tape_overflow);
    trace_stage[photon_id] = CHROMA_TAPE_STAGE_MAPPING;
    record_first_tape_error(
        photon_id, state.overflow, state.interaction, state.draw,
        CHROMA_TAPE_STAGE_MAPPING, first_error_photon_id, first_error_flags,
        first_error_interaction, first_error_draw, first_error_stage);
}

__global__ void
propagate_tape(
    int first_photon,
    int nthreads,
    const unsigned int *__restrict__ input_queue,
    unsigned int *__restrict__ output_queue,
    float3 *__restrict__ positions,
    float3 *__restrict__ directions,
    float *__restrict__ wavelengths,
    float3 *__restrict__ polarizations,
    float *__restrict__ times,
    unsigned int *__restrict__ histories,
    int *__restrict__ last_hit_triangles,
    float *__restrict__ weights,
    unsigned int *__restrict__ evidx,
    int max_steps,
    int use_weights,
    int scatter_first,
    Geometry *g,
    const int *__restrict__ photon_tape_rows,
    const long long *__restrict__ photon_global_ids,
    const float *__restrict__ tape_values,
    const long long *__restrict__ tape_global_photon_ids,
    int tape_photon_count,
    int tape_max_interactions,
    int tape_draws_per_interaction,
    int *__restrict__ interaction_cursor,
    int *__restrict__ draw_cursor,
    unsigned int *__restrict__ tape_overflow,
    unsigned int *__restrict__ interaction_certificate,
    int certify_interactions,
    unsigned int *__restrict__ state_certificate,
    int certify_states,
    int *__restrict__ trace_process,
    int *__restrict__ trace_interaction,
    int *__restrict__ trace_draw_count,
    int *__restrict__ trace_stage,
    int *__restrict__ first_error_photon_id,
    unsigned int *__restrict__ first_error_flags,
    int *__restrict__ first_error_interaction,
    int *__restrict__ first_error_draw,
    int *__restrict__ first_error_stage)
{
    __shared__ Geometry sg;

    if (threadIdx.x == 0)
        sg = *g;
    __syncthreads();

    int worker = blockIdx.x*blockDim.x + threadIdx.x;
    if (worker >= nthreads)
        return;

    g = &sg;
    int photon_id = input_queue[first_photon + worker];
    int tape_row = photon_tape_rows[photon_id];
    long long global_photon_id = photon_global_ids[photon_id];

    ChromaRandomTapeView tape;
    tape.values = tape_values;
    tape.global_photon_ids = tape_global_photon_ids;
    tape.photon_count = tape_photon_count;
    tape.max_interactions = tape_max_interactions;
    tape.draws_per_interaction = tape_draws_per_interaction;

    bool row_in_range = tape_row >= 0 && tape_row < tape_photon_count;
    int interaction = row_in_range ? interaction_cursor[tape_row] : 0;
    int draw = row_in_range ? draw_cursor[tape_row] : 0;
    unsigned int prior_overflow = row_in_range ? tape_overflow[tape_row] : 0u;
    ChromaRandomTapeState rng = chroma_random_tape_init(
        tape, tape_row, global_photon_id, interaction, draw, prior_overflow);

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

    bool runnable = rng.mapping_valid && rng.overflow == 0u &&
        ((p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT |
                       SURFACE_ABSORB | NAN_ABORT)) == 0);
    bool started_runnable = runnable;

    int last_process = CHROMA_TAPE_PROCESS_UNSET;
    int last_interaction = rng.interaction;
    int last_draw_count = rng.draw;
    int last_stage = CHROMA_TAPE_STAGE_MAPPING;

    int steps = 0;
    while (runnable && steps < max_steps) {
        steps++;

        if (isnan(p.direction.x*p.direction.y*p.direction.z*
                  p.position.x*p.position.y*p.position.z)) {
            p.history |= NO_HIT | NAN_ABORT;
            break;
        }

        State s;
        fill_state(s, p, g);
        if (p.last_hit_triangle == -1)
            break;

        last_stage = CHROMA_TAPE_STAGE_BULK;
        int command = propagate_to_boundary(
            p, s, rng, use_weights, scatter_first);
        scatter_first = 0;

        last_process = rng.process;
        last_interaction = rng.interaction;
        last_draw_count = rng.draw;
        if (rng.overflow != 0u) {
            p.history |= NAN_ABORT;
            break;
        }

        // A tape interaction is exactly one iteration of Chroma's transport
        // loop.  Collision, surface and dielectric draws therefore share one
        // bounded row segment in their original call order.
        if (command == BREAK) {
            record_committed_interaction(
                rng, interaction_certificate, certify_interactions);
            record_committed_state(
                rng, p, state_certificate, certify_states);
            chroma_random_tape_next_interaction(&rng);
            break;
        }
        if (command == CONTINUE) {
            record_committed_interaction(
                rng, interaction_certificate, certify_interactions);
            record_committed_state(
                rng, p, state_certificate, certify_states);
            chroma_random_tape_next_interaction(&rng);
            if (rng.overflow != 0u) {
                p.history |= NAN_ABORT;
                break;
            }
            continue;
        }

        if (s.surface_index != -1) {
            last_stage = CHROMA_TAPE_STAGE_SURFACE;
            command = propagate_at_surface(p, s, rng, g, use_weights);
            last_process = rng.process;
            last_interaction = rng.interaction;
            last_draw_count = rng.draw;
            if (rng.overflow != 0u) {
                p.history |= NAN_ABORT;
                break;
            }
            if (command == BREAK) {
                record_committed_interaction(
                    rng, interaction_certificate, certify_interactions);
                record_committed_state(
                    rng, p, state_certificate, certify_states);
                chroma_random_tape_next_interaction(&rng);
                break;
            }
            if (command == CONTINUE) {
                record_committed_interaction(
                    rng, interaction_certificate, certify_interactions);
                record_committed_state(
                    rng, p, state_certificate, certify_states);
                chroma_random_tape_next_interaction(&rng);
                if (rng.overflow != 0u) {
                    p.history |= NAN_ABORT;
                    break;
                }
                continue;
            }
        }

        last_stage = CHROMA_TAPE_STAGE_DIELECTRIC;
        propagate_at_boundary(p, s, rng);
        last_process = rng.process;
        last_interaction = rng.interaction;
        last_draw_count = rng.draw;
        if (rng.overflow != 0u) {
            p.history |= NAN_ABORT;
            break;
        }
        record_committed_interaction(
            rng, interaction_certificate, certify_interactions);
        record_committed_state(
            rng, p, state_certificate, certify_states);
        chroma_random_tape_next_interaction(&rng);
        if (rng.overflow != 0u) {
            p.history |= NAN_ABORT;
            break;
        }
    }

    chroma_random_tape_commit(
        rng, interaction_cursor, draw_cursor, tape_overflow);
    trace_process[photon_id] = last_process;
    trace_interaction[photon_id] = last_interaction;
    trace_draw_count[photon_id] = last_draw_count;
    trace_stage[photon_id] = last_stage;
    record_first_tape_error(
        photon_id, rng.overflow, rng.interaction, rng.draw, last_stage,
        first_error_photon_id, first_error_flags, first_error_interaction,
        first_error_draw, first_error_stage);

    if (rng.mapping_valid && started_runnable) {
        positions[photon_id] = p.position;
        directions[photon_id] = p.direction;
        polarizations[photon_id] = p.polarization;
        wavelengths[photon_id] = p.wavelength;
        times[photon_id] = p.time;
        histories[photon_id] = p.history;
        last_hit_triangles[photon_id] = p.last_hit_triangle;
        weights[photon_id] = p.weight;
        evidx[photon_id] = p.evidx;
    }

    unsigned int still_alive = rng.mapping_valid && rng.overflow == 0u &&
        ((p.history & (NO_HIT | BULK_ABSORB | SURFACE_DETECT |
                       SURFACE_ABSORB | NAN_ABORT)) == 0) ? 1u : 0u;
    unsigned int active_mask = __activemask();
    unsigned int mask = __ballot_sync(active_mask, still_alive);
    int lane = threadIdx.x & 31;
    int warp_count = __popc(mask);
    int warp_prefix = __popc(mask & ((1u << lane) - 1u));
    unsigned int base = 0;
    if (lane == 0 && warp_count > 0)
        base = atomicAdd(output_queue, (unsigned int)warp_count);
    base = __shfl_sync(active_mask, base, 0);
    if (still_alive)
        output_queue[base + warp_prefix] = photon_id;
}

} // extern "C"
