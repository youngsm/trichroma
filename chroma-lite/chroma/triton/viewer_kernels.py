"""Small rendering kernels; transport and BVH traversal remain shared services."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["seed"])
def camera_rays(
    origins,
    directions,
    distance,
    shade,
    camera,
    count,
    width,
    height,
    seed,
    jitter: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ray = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = ray < count
    pixel = ray % (width * height)
    if jitter:
        jx, jy = tl.rand(seed, ray * 2), tl.rand(seed, ray * 2 + 1)
    else:
        jx, jy = 0.5, 0.5
    u = (2 * (pixel % width + jx) / width - 1) * (width / height)
    v = 1 - 2 * (pixel // width + jy) / height
    tan_half_fov = tl.load(camera + 12)
    dx = tl.load(camera + 3) + tan_half_fov * (u * tl.load(camera + 6) + v * tl.load(camera + 9))
    dy = tl.load(camera + 4) + tan_half_fov * (u * tl.load(camera + 7) + v * tl.load(camera + 10))
    dz = tl.load(camera + 5) + tan_half_fov * (u * tl.load(camera + 8) + v * tl.load(camera + 11))
    norm = tl.rsqrt(dx * dx + dy * dy + dz * dz)
    for axis in tl.static_range(3):
        tl.store(origins + 3 * ray + axis, tl.load(camera + axis), valid)
    tl.store(directions + 3 * ray, dx * norm, valid)
    tl.store(directions + 3 * ray + 1, dy * norm, valid)
    tl.store(directions + 3 * ray + 2, dz * norm, valid)
    tl.store(distance + ray, float("inf"), valid)
    # Background deliberately distinct from the neutral detector walls.
    tl.store(shade + 3 * ray, 0.035, valid)
    tl.store(shade + 3 * ray + 1, 0.045, valid)
    tl.store(shade + 3 * ray + 2, 0.065, valid)


@triton.jit
def shade_hits(
    triangles,
    distances,
    normals,
    colors,
    directions,
    nearest,
    shade,
    count,
    PER_RAY_NORMALS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    ray = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = ray < count
    triangle = tl.load(triangles + ray, valid, other=-1)
    distance = tl.load(distances + ray, valid, other=float("inf"))
    previous = tl.load(nearest + ray, valid, other=0.0)
    hit = valid & (triangle >= 0) & (distance < previous)
    normal_index = ray if PER_RAY_NORMALS else triangle
    nx = tl.load(normals + 3 * normal_index, hit, other=0.0)
    ny = tl.load(normals + 3 * normal_index + 1, hit, other=0.0)
    nz = tl.load(normals + 3 * normal_index + 2, hit, other=0.0)
    dx = tl.load(directions + 3 * ray, valid, other=0.0)
    dy = tl.load(directions + 3 * ray + 1, valid, other=0.0)
    dz = tl.load(directions + 3 * ray + 2, valid, other=0.0)
    intensity = 0.22 + 0.78 * tl.abs(nx * dx + ny * dy + nz * dz)
    packed = tl.load(colors + triangle, hit, other=0).to(tl.uint32)
    red = ((packed >> 16) & 255).to(tl.float32) / 255.0
    green = ((packed >> 8) & 255).to(tl.float32) / 255.0
    blue = (packed & 255).to(tl.float32) / 255.0
    tl.store(shade + 3 * ray, red * intensity, hit)
    tl.store(shade + 3 * ray + 1, green * intensity, hit)
    tl.store(shade + 3 * ray + 2, blue * intensity, hit)
    tl.store(nearest + ray, distance, hit)


@triton.jit
def resolve_image(shade, image, count, pixels, SAMPLES: tl.constexpr, BLOCK: tl.constexpr):
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pixel < pixels
    red = tl.full((BLOCK,), 0.0, tl.float32)
    green = tl.full((BLOCK,), 0.0, tl.float32)
    blue = tl.full((BLOCK,), 0.0, tl.float32)
    samples = tl.full((BLOCK,), 0.0, tl.float32)
    for sample in range(SAMPLES):
        ray = pixel + sample * pixels
        active = valid & (ray < count)
        red += tl.load(shade + 3 * ray, active, other=0.0)
        green += tl.load(shade + 3 * ray + 1, active, other=0.0)
        blue += tl.load(shade + 3 * ray + 2, active, other=0.0)
        samples += active.to(tl.float32)
    divisor = tl.maximum(samples, 1.0)
    tl.store(image + 3 * pixel, (255 * red / divisor + 0.5).to(tl.uint8), valid)
    tl.store(image + 3 * pixel + 1, (255 * green / divisor + 0.5).to(tl.uint8), valid)
    tl.store(image + 3 * pixel + 2, (255 * blue / divisor + 0.5).to(tl.uint8), valid)
