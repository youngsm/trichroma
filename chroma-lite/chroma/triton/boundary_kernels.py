"""Triton counterpart of the CPU boundary-origin construction."""

import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import nextafter


@triton.jit
def offset_boundary_point(vertices, tri, x, y, z, dx, dy, dz, mask):
    ax = tl.load(vertices + tri * 9, mask=mask, other=0.0).to(tl.float64)
    ay = tl.load(vertices + tri * 9 + 1, mask=mask, other=0.0).to(tl.float64)
    az = tl.load(vertices + tri * 9 + 2, mask=mask, other=0.0).to(tl.float64)
    ux = tl.load(vertices + tri * 9 + 3, mask=mask, other=0.0).to(tl.float64) - ax
    uy = tl.load(vertices + tri * 9 + 4, mask=mask, other=0.0).to(tl.float64) - ay
    uz = tl.load(vertices + tri * 9 + 5, mask=mask, other=0.0).to(tl.float64) - az
    vx = tl.load(vertices + tri * 9 + 6, mask=mask, other=0.0).to(tl.float64) - ax
    vy = tl.load(vertices + tri * 9 + 7, mask=mask, other=0.0).to(tl.float64) - ay
    vz = tl.load(vertices + tri * 9 + 8, mask=mask, other=0.0).to(tl.float64) - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    norm2 = nx * nx + ny * ny + nz * nz
    extent = tl.maximum(
        tl.maximum(
            tl.abs(ux) + tl.abs(vx) + tl.abs(tl.abs(ux) - tl.abs(vx)),
            tl.abs(uy) + tl.abs(vy) + tl.abs(tl.abs(uy) - tl.abs(vy)),
        ),
        tl.abs(uz) + tl.abs(vz) + tl.abs(tl.abs(uz) - tl.abs(vz)),
    )
    epsilon = 1.1920928955078125e-7
    ex, ey, ez = (
        epsilon * tl.abs(ax) + 3 * epsilon * extent,
        epsilon * tl.abs(ay) + 3 * epsilon * extent,
        epsilon * tl.abs(az) + 3 * epsilon * extent,
    )
    clearance = tl.abs(nx) * ex + tl.abs(ny) * ey + tl.abs(nz) * ez
    residual = (
        (x.to(tl.float64) - ax) * nx + (y.to(tl.float64) - ay) * ny + (z.to(tl.float64) - az) * nz
    )
    side = tl.where(
        dx.to(tl.float64) * nx + dy.to(tl.float64) * ny + dz.to(tl.float64) * nz >= 0, 1.0, -1.0
    )
    move = (side * clearance - residual) / tl.where(norm2 > 0, norm2, 1.0)
    a = (x.to(tl.float64) + move * nx).to(tl.float32)
    b = (y.to(tl.float64) + move * ny).to(tl.float32)
    c = (z.to(tl.float64) + move * nz).to(tl.float32)
    a = tl.where(nx != 0, nextafter(a, tl.where(side * nx > 0, float("inf"), -float("inf"))), a)
    b = tl.where(ny != 0, nextafter(b, tl.where(side * ny > 0, float("inf"), -float("inf"))), b)
    c = tl.where(nz != 0, nextafter(c, tl.where(side * nz > 0, float("inf"), -float("inf"))), c)
    return tl.where(mask, a, x), tl.where(mask, b, y), tl.where(mask, c, z)
