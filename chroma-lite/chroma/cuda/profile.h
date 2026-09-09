// Device-side lightweight profiling for CUDA kernels.
// Enabled when compiled with -DCHROMA_DEVICE_PROFILE=1

#ifndef CHROMA_CUDA_PROFILE_H
#define CHROMA_CUDA_PROFILE_H

#include <stdint.h>

enum ChromaProfileId {
    CHROMA_PROF_INTERSECT_MESH = 0,
    CHROMA_PROF_INTERSECT_NODE = 1,
    CHROMA_PROF_INTERSECT_TRIANGLE = 2,
    CHROMA_PROF_INTERSECT_BOX = 3,
    CHROMA_PROF_FILL_MATERIAL = 4,
    CHROMA_PROF_FILL_ANALYTIC = 5,
    CHROMA_PROF_COUNT = 64
};

#ifdef CHROMA_DEVICE_PROFILE

// device-global counters are defined in one CU (propagate.cu)
extern "C" __device__ unsigned long long chroma_prof_calls[CHROMA_PROF_COUNT];
extern "C" __device__ unsigned long long chroma_prof_cycles[CHROMA_PROF_COUNT];

#define CHROMA_PROF_FUNC_START(ID) \
    unsigned long long __chroma_prof_t0 = clock64(); \
    atomicAdd(&chroma_prof_calls[(ID)], 1ULL);

#define CHROMA_PROF_FUNC_END(ID) \
    atomicAdd(&chroma_prof_cycles[(ID)], (unsigned long long)(clock64() - __chroma_prof_t0));

#else

#define CHROMA_PROF_FUNC_START(ID) do {} while(0)
#define CHROMA_PROF_FUNC_END(ID)   do {} while(0)

#endif // CHROMA_DEVICE_PROFILE

#endif // CHROMA_CUDA_PROFILE_H

