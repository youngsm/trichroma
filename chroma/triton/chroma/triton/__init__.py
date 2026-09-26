"""Triton optical-transport building blocks.

The package deliberately keeps CPU reference implementations separate from
GPU kernels.  Import :mod:`chroma.triton.physics` in validation code and
:mod:`chroma.triton.physics_kernels` only in environments with Triton.
"""

