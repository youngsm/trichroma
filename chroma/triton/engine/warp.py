"""Warp-level helpers for the one-warp programs of the production engine.

Every kernel that uses them runs one warp per program (``num_warps=1``) on
tensors of BLOCK = 32 elements, one per lane with element i in lane i (the
only layout Triton gives such tensors), and executes them with all 32 lanes
converged (Triton's control flow is warp-uniform), as Triton's own warp
reductions assume. Each helper returns exactly the integer that the
corresponding Triton reduction or scan would, with fewer instructions.
"""

import triton
import triton.language as tl


@triton.jit
def _lane_rank(m):
    """Number of lower lanes of the warp whose mask ``m`` is set:
    popc(ballot(m) & %lanemask_lt), i.e. ``tl.cumsum(m) - m``. For the
    one-warp programs here (BLOCK = 32 elements, one per lane, element i in
    lane i: the kernels' only layout) this is the same integer as Triton's
    shuffle scan, in three instructions instead of ~20 (and without the
    loop-invariant lane predicates the scan keeps live)."""
    return tl.inline_asm_elementwise(
        "{ .reg .pred %lrp; .reg .b32 %lrb, %lrl; setp.ne.b32 %lrp, $1, 0; "
        "vote.sync.ballot.b32 %lrb, %lrp, 0xffffffff; mov.u32 %lrl, %lanemask_lt; "
        "and.b32 %lrb, %lrb, %lrl; popc.b32 $0, %lrb; }",
        "=r,r", [m.to(tl.int32)], dtype=tl.int32, is_pure=True, pack=1)


# Warp-uniform decisions (``if any lane ...``) as warp votes. Triton makes a
# scalar out of a per-lane mask only by a reduction (redux.sync on sm_80, five
# shuffles before), so the vote is done by two volatile asm statements that
# share a PTX register: the first (per-lane input) opens a PTX scope, declares
# the register and votes into it; the second (no input: a scalar output)
# reads it and closes the scope. They are emitted back to back (volatile asm
# keeps its order and is never duplicated apart), so the braces pair up; the
# result is the exact any/count of the mask. One warp, element i in lane i,
# every lane converged (as for Triton's own warp reductions).


@triton.jit
def _warp_any(m):
    """1 if the mask ``m`` is set in some lane of the warp, else 0 (a scalar)."""
    tl.inline_asm_elementwise(
        "{ .reg .pred %wap, %wat; setp.ne.b32 %wat, $1, 0; vote.sync.any.pred %wap, %wat, 0xffffffff; mov.u32 $0, 0;",
        "=r,r", [m.to(tl.int32)], dtype=tl.int32, is_pure=False, pack=1)
    return tl.inline_asm_elementwise("selp.u32 $0, 1, 0, %wap; }", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _warp_count(m):
    """Number of lanes of the warp whose mask ``m`` is set (a scalar)."""
    tl.inline_asm_elementwise(
        "{ .reg .pred %wct; .reg .b32 %wcb; setp.ne.b32 %wct, $1, 0; vote.sync.ballot.b32 %wcb, %wct, 0xffffffff; "
        "mov.u32 $0, 0;",
        "=r,r", [m.to(tl.int32)], dtype=tl.int32, is_pure=False, pack=1)
    return tl.inline_asm_elementwise("popc.b32 $0, %wcb; }", "=r", [], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _ring(v, CAP: tl.constexpr):
    """Ring slot ``v mod CAP`` of a non-negative counter (a mask when CAP is a
    power of two: the same value, one instruction instead of four)."""
    if (CAP & (CAP - 1)) == 0:
        return v & (CAP - 1)
    else:
        return v % CAP
