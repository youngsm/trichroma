"""Independent host initialization of CUDA-compatible XORWOW uniform streams.

The 160-bit linear recurrence is exponentiated over GF(2); no CUDA state dump
or precomputed vendor transition table is needed. A subsequence is 2**67 draws.
Only the six words used by uniform draws are represented (Weyl d, then v[0:5]).
"""

from functools import lru_cache
import numpy as np


def _transition(v):
    t = v[..., 0] ^ (v[..., 0] >> np.uint32(2))
    last = (v[..., 4] ^ (v[..., 4] << np.uint32(4))) ^ (t ^ (t << np.uint32(1)))
    return np.concatenate((v[..., 1:], last[..., None]), axis=-1)


def _apply(columns, vectors):
    result = np.zeros_like(vectors)
    for word in range(5):
        for bit in range(32):
            present = ((vectors[..., word] >> np.uint32(bit)) & np.uint32(1)) != 0
            result ^= np.where(present[..., None], columns[word * 32 + bit], np.uint32(0))
    return result


@lru_cache(maxsize=132)
def _power(exponent):
    if exponent == 0:
        basis = np.zeros((160, 5), np.uint32)
        index = np.arange(160)
        basis[index, index // 32] = np.uint32(1) << (index % 32).astype(np.uint32)
        result = _transition(basis)
    else:
        half = _power(exponent - 1)
        result = _apply(half, half)
    result.flags.writeable = False
    return result


def initialize_xorwow(seed, subsequences, offset=0):
    """Return shape(N,6) uint32 state for explicit native worker subsequences."""
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned64-bit integer")
    if not isinstance(offset, (int, np.integer)) or not 0 <= offset < 2**64:
        raise ValueError("offset must be an unsigned64-bit integer")
    seq = np.asarray(subsequences)
    if seq.ndim != 1 or not np.issubdtype(seq.dtype, np.integer) or np.any(seq < 0):
        raise ValueError("subsequences must be a vector of nonnegative integers")
    seq = seq.astype(np.uint64)
    t0 = (1099087573 * ((int(seed) & 0xFFFFFFFF) ^ 0xAAD26B49)) & 0xFFFFFFFF
    t1 = (2591861531 * ((int(seed) >> 32) ^ 0xF7DCEFDD)) & 0xFFFFFFFF
    initial = np.asarray(
        [
            (123456789 + t0) & 0xFFFFFFFF,
            362436069 ^ t0,
            (521288629 + t1) & 0xFFFFFFFF,
            88675123 ^ t1,
            (5783321 + t0) & 0xFFFFFFFF,
        ],
        np.uint32,
    )
    v = np.broadcast_to(initial, (len(seq), 5)).copy()
    for bit in range(int(seq.max(initial=0)).bit_length()):
        selected = ((seq >> np.uint64(bit)) & np.uint64(1)) != 0
        v[selected] = _apply(_power(67 + bit), v[selected])
    for bit in range(int(offset).bit_length()):
        if (int(offset) >> bit) & 1:
            v = _apply(_power(bit), v)
    d = (6615241 + t1 + t0 + 362437 * int(offset)) & 0xFFFFFFFF
    return np.column_stack((np.full(len(seq), d, np.uint32), v))


def xorwow_uint32(states):
    """Advance uniform-generator words in place and return each next uint32."""
    states[:, 1:] = _transition(states[:, 1:])
    with np.errstate(over="ignore"):
        states[:, 0] += np.uint32(362437)
        return states[:, 0] + states[:, 5]
