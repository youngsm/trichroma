#ifndef __CHROMA_RNG_ALIGNMENT_H__
#define __CHROMA_RNG_ALIGNMENT_H__

// Bounded, photon-stable random tape used only for CUDA/Triton lockstep
// validation.  Unlike Chroma's ordinary XORWOW state, this state is created
// from a photon row after the compacted queue has resolved the photon ID.

#define CHROMA_TAPE_DRAW_OVERFLOW        (1u << 0)
#define CHROMA_TAPE_INTERACTION_OVERFLOW (1u << 1)
#define CHROMA_TAPE_GLOBAL_ID_MISMATCH   (1u << 2)
#define CHROMA_TAPE_ROW_OUT_OF_RANGE     (1u << 3)

// Shared process/stage vocabulary for CUDA/Triton lockstep traces.  Values
// are intentionally independent of Chroma's cumulative history bit mask.
enum ChromaTapeProcess
{
    CHROMA_TAPE_PROCESS_UNSET = 0,
    CHROMA_TAPE_PROCESS_BULK_ABSORB = 1,
    CHROMA_TAPE_PROCESS_BULK_SCATTER = 2,
    CHROMA_TAPE_PROCESS_SURFACE_ABSORB = 3,
    CHROMA_TAPE_PROCESS_SURFACE_DETECT = 4,
    CHROMA_TAPE_PROCESS_SURFACE_DIFFUSE = 5,
    CHROMA_TAPE_PROCESS_SURFACE_SPECULAR = 6,
    CHROMA_TAPE_PROCESS_DIELECTRIC_REFLECT = 7,
    CHROMA_TAPE_PROCESS_DIELECTRIC_TRANSMIT = 8,
    CHROMA_TAPE_PROCESS_BULK_REEMIT = 9,
    CHROMA_TAPE_PROCESS_SURFACE_REEMIT = 10
};

enum ChromaTapeStage
{
    CHROMA_TAPE_STAGE_MAPPING = 0,
    CHROMA_TAPE_STAGE_BULK = 1,
    CHROMA_TAPE_STAGE_SURFACE = 2,
    CHROMA_TAPE_STAGE_DIELECTRIC = 3
};

struct ChromaRandomTapeView
{
    const float *values;
    const long long *global_photon_ids;
    int photon_count;
    int max_interactions;
    int draws_per_interaction;
};

struct ChromaRandomTapeState
{
    ChromaRandomTapeView tape;
    int row;
    int interaction;
    int draw;
    unsigned int overflow;
    int process;
    bool row_valid;
    bool mapping_valid;
};

__device__ __forceinline__ float
chroma_random_tape_nan()
{
    return __int_as_float(0x7fc00000);
}

__device__ __forceinline__ ChromaRandomTapeState
chroma_random_tape_init(const ChromaRandomTapeView &tape,
                        int row,
                        long long expected_global_photon_id,
                        int interaction,
                        int draw,
                        unsigned int previous_overflow)
{
    ChromaRandomTapeState state;
    state.tape = tape;
    state.row = row;
    state.interaction = interaction;
    state.draw = draw;
    state.overflow = previous_overflow;
    state.process = CHROMA_TAPE_PROCESS_UNSET;
    state.row_valid = row >= 0 && row < tape.photon_count;
    state.mapping_valid = state.row_valid &&
        tape.global_photon_ids[row] == expected_global_photon_id;
    if (!state.row_valid)
        state.overflow |= CHROMA_TAPE_ROW_OUT_OF_RANGE;
    else if (!state.mapping_valid)
        state.overflow |= CHROMA_TAPE_GLOBAL_ID_MISMATCH;
    if (state.mapping_valid &&
        (interaction < 0 || interaction >= tape.max_interactions))
        state.overflow |= CHROMA_TAPE_INTERACTION_OVERFLOW;
    return state;
}

__device__ __forceinline__ float
chroma_random_tape_uniform(ChromaRandomTapeState *state)
{
    if (!state->mapping_valid)
        return chroma_random_tape_nan();
    if (state->interaction < 0 ||
        state->interaction >= state->tape.max_interactions) {
        state->overflow |= CHROMA_TAPE_INTERACTION_OVERFLOW;
        return chroma_random_tape_nan();
    }
    if (state->draw < 0 ||
        state->draw >= state->tape.draws_per_interaction) {
        state->overflow |= CHROMA_TAPE_DRAW_OVERFLOW;
        state->draw = state->tape.draws_per_interaction;
        return chroma_random_tape_nan();
    }
    unsigned long long offset =
        ((unsigned long long)state->row * state->tape.max_interactions +
         (unsigned long long)state->interaction) *
        state->tape.draws_per_interaction + (unsigned long long)state->draw;
    float value = state->tape.values[offset];
    state->draw += 1;
    return value;
}

__device__ __forceinline__ void
chroma_random_tape_next_interaction(ChromaRandomTapeState *state)
{
    state->interaction += 1;
    state->draw = 0;
    state->process = CHROMA_TAPE_PROCESS_UNSET;
    if (state->interaction >= state->tape.max_interactions)
        state->overflow |= CHROMA_TAPE_INTERACTION_OVERFLOW;
}

__device__ __forceinline__ void
chroma_random_tape_commit(const ChromaRandomTapeState &state,
                          int *interaction_cursor,
                          int *draw_cursor,
                          unsigned int *overflow)
{
    if (!state.row_valid)
        return;
    interaction_cursor[state.row] = state.interaction;
    draw_cursor[state.row] = state.draw;
    overflow[state.row] = state.overflow;
}

#endif
