"""Triton scheduling of the fingerprinted original FP32 analytic-wire primitive."""

import triton
import triton.language as tl
from ._legacy_wire_ptx import ASM as _ASM


@triton.jit
def merge_original_wires(
    positions,
    directions,
    planes,
    distance,
    normal,
    material_from,
    material_to,
    surface,
    triangle,
    count,
    NPLANES: tl.constexpr,
    BLOCK: tl.constexpr,
    ASM: tl.constexpr = _ASM,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = (row < count).to(tl.int32)
    tl.inline_asm_elementwise(
        asm=ASM,
        constraints="=r,l,l,l,l,l,l,l,l,l,r,r,r",
        args=[
            positions.to(tl.uint64),
            directions.to(tl.uint64),
            planes.to(tl.uint64),
            distance.to(tl.uint64),
            normal.to(tl.uint64),
            material_from.to(tl.uint64),
            material_to.to(tl.uint64),
            surface.to(tl.uint64),
            triangle.to(tl.uint64),
            tl.full((BLOCK,), NPLANES, tl.int32),
            row,
            valid,
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
