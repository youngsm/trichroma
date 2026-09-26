// Opt-in probe for the photon-stable validation tape.  Production propagation
// does not compile this source unless a validation caller explicitly requests
// it through get_cu_module("rng_alignment.cu").

#include "rng_alignment.h"

extern "C"
{

__global__ void
rng_alignment_probe(int nwork,
                    int max_requests,
                    const float *__restrict__ tape_values,
                    const long long *__restrict__ tape_global_photon_ids,
                    int photon_count,
                    int max_interactions,
                    int draws_per_interaction,
                    const int *__restrict__ row_indices,
                    const long long *__restrict__ requested_global_photon_ids,
                    const int *__restrict__ requested_draws,
                    int *__restrict__ interaction_cursor,
                    int *__restrict__ draw_cursor,
                    unsigned int *__restrict__ overflow,
                    float *__restrict__ output,
                    unsigned int *__restrict__ work_overflow)
{
    int worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= nwork)
        return;

    ChromaRandomTapeView tape;
    tape.values = tape_values;
    tape.global_photon_ids = tape_global_photon_ids;
    tape.photon_count = photon_count;
    tape.max_interactions = max_interactions;
    tape.draws_per_interaction = draws_per_interaction;

    int row = row_indices[worker];
    bool row_valid = row >= 0 && row < photon_count;
    int interaction = row_valid ? interaction_cursor[row] : 0;
    int draw = row_valid ? draw_cursor[row] : 0;
    unsigned int prior_overflow = row_valid ? overflow[row] : 0u;
    ChromaRandomTapeState state = chroma_random_tape_init(
        tape, row, requested_global_photon_ids[worker], interaction, draw,
        prior_overflow);

    int count = requested_draws[worker];
    for (int request = 0; request < max_requests; ++request) {
        output[worker * max_requests + request] = request < count
            ? chroma_random_tape_uniform(&state)
            : chroma_random_tape_nan();
    }
    chroma_random_tape_commit(
        state, interaction_cursor, draw_cursor, overflow);
    work_overflow[worker] = state.overflow;
}

__global__ void
rng_alignment_advance_interactions(int nrows,
                                   const int *__restrict__ rows,
                                   int photon_count,
                                   int max_interactions,
                                   int *__restrict__ interaction_cursor,
                                   int *__restrict__ draw_cursor,
                                   unsigned int *__restrict__ overflow)
{
    int worker = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker >= nrows)
        return;
    int row = rows[worker];
    if (row < 0 || row >= photon_count)
        return;
    int interaction = interaction_cursor[row] + 1;
    interaction_cursor[row] = interaction;
    draw_cursor[row] = 0;
    if (interaction >= max_interactions)
        overflow[row] |= CHROMA_TAPE_INTERACTION_OVERFLOW;
}

} // extern "C"
