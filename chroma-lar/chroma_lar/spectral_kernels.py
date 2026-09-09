"""Spectral source and collision-first kernels using the calibrated physics laws."""
import triton
import triton.language as tl

from chroma.triton.spectral_kernels import (
    random_uniform, interpolate, sample_time_cdf, sample_regular_cdf,
    hemisphere, polarization,
)
from chroma.triton.physics_kernels import rayleigh_scatter, fresnel_step


@triton.jit
def append_rows(rows, mask, output, output_count):
    selected = mask.to(tl.int32)
    total = tl.sum(selected, 0)
    offset = tl.cumsum(selected, 0) - selected
    start = tl.atomic_add(output_count, total)
    tl.store(output+start+offset, rows, mask=mask)


@triton.jit
def round_to_material_side(coordinate, normal):
    """Directed float32 rounding of a reconstructed FP64 boundary point."""
    value = coordinate.to(tl.float32)
    bits = value.to(tl.int32, bitcast=True)
    increment = tl.where((value >= 0) == (normal > 0), 1, -1)
    stepped = (bits+increment).to(tl.float32, bitcast=True)
    zero_bits = tl.where(normal > 0, 1, -2147483647)
    stepped = tl.where(value == 0, zero_bits.to(tl.float32, bitcast=True), stepped)
    wrong_side = ((normal > 0) & (value.to(tl.float64) <= coordinate)) | ((normal < 0) & (value.to(tl.float64) >= coordinate))
    return tl.where(wrong_side, stepped, value)


@triton.jit
def source_kernel(pos, direction, pol, times, flags, ids, last_instance, last_triangle,
                  channels, steps, wavelengths, events,
                  spectrum_offsets, spectrum_x, spectrum_cdf, spectrum_pdf,
                  lifetimes, fractions, count, seed, id_base, event_id,
                  cx, cy, cz, voxel, birth, rise,
                  NC: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = row < count
    photon = row.to(tl.int64)+id_base
    choice = random_uniform(photon, seed, 0x10000000)
    cumulative = tl.full((BLOCK,), 0., tl.float64)
    tau = tl.full((BLOCK,), 0., tl.float32)
    for component in tl.static_range(NC):
        weight = tl.load(fractions+component)
        lifetime = tl.load(lifetimes+component)
        tau = tl.where((choice >= cumulative) & (choice < cumulative+weight), lifetime, tau)
        cumulative += weight
    t = birth-tau*tl.log(random_uniform(photon, seed, 0x10000001))
    t -= rise*tl.log(random_uniform(photon, seed, 0x10000002))
    dz = 2*random_uniform(photon, seed, 0x10000003)-1
    phi = 6.283185307179586*random_uniform(photon, seed, 0x10000004)
    r = tl.sqrt(tl.maximum(0., 1-dz*dz))
    sn, cs = tl.sin(phi), tl.cos(phi)
    dx, dy = r*cs, r*sn
    alpha = 6.283185307179586*random_uniform(photon, seed, 0x10000005)
    a, b = tl.cos(alpha), tl.sin(alpha)
    px, py, pz = -a*sn-b*dz*cs, a*cs-b*dz*sn, b*r
    wl = sample_time_cdf(spectrum_offsets, spectrum_x, spectrum_cdf, spectrum_pdf,
                         tl.full((BLOCK,), 0, tl.int32),
                         random_uniform(photon, seed, 0x10000006), valid)
    for axis in tl.static_range(3):
        center = cx if axis == 0 else cy if axis == 1 else cz
        coordinate = center+voxel*(random_uniform(photon, seed, 0x10000020+axis)-.5)
        tl.store(pos+row*3+axis, coordinate, valid)
    tl.store(direction+row*3, dx, valid)
    tl.store(direction+row*3+1, dy, valid)
    tl.store(direction+row*3+2, dz, valid)
    tl.store(pol+row*3, px, valid)
    tl.store(pol+row*3+1, py, valid)
    tl.store(pol+row*3+2, pz, valid)
    tl.store(times+row, t, valid)
    tl.store(flags+row, 1 << 11, valid)
    tl.store(ids+row, photon, valid)
    tl.store(last_instance+row, -1, valid)
    tl.store(last_triangle+row, -1, valid)
    tl.store(channels+row, -1, valid)
    tl.store(steps+row, 0, valid)
    tl.store(wavelengths+row, wl, valid)
    tl.store(events+row, event_id, valid)


@triton.jit
def bulk_epoch(pos, direction, pol, times, flags, ids, last_instance, last_triangle,
               channels, steps, wavelengths, events,
               input_rows, input_count, capacity, output_rows, output_count,
               boundary_rows, boundary_count, absorption, scattering, velocities,
               seed, max_steps, lx, ly, lz, ux, uy, uz, wl_start, wl_step,
               LAR: tl.constexpr, NW: tl.constexpr, HISTORY: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    count = tl.load(input_count)
    if tl.program_id(0)*BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row = tl.load(input_rows+lane, valid, 0)
    photon = tl.load(ids+row, valid, 0)
    x = tl.load(pos+row*3, valid, 0.)
    y = tl.load(pos+row*3+1, valid, 0.)
    z = tl.load(pos+row*3+2, valid, 0.)
    dx = tl.load(direction+row*3, valid, 1.)
    dy = tl.load(direction+row*3+1, valid, 0.)
    dz = tl.load(direction+row*3+2, valid, 0.)
    px = tl.load(pol+row*3, valid, 0.)
    py = tl.load(pol+row*3+1, valid, 1.)
    pz = tl.load(pol+row*3+2, valid, 0.)
    t = tl.load(times+row, valid, 0.)
    history = tl.load(flags+row, valid, 0)
    step = tl.load(steps+row, valid, 0)
    wl = tl.load(wavelengths+row, valid, wl_start)
    alen = interpolate(absorption, LAR, wl, wl_start, wl_step, NW, valid)
    slen = interpolate(scattering, LAR, wl, wl_start, wl_step, NW, valid)
    speed = interpolate(velocities, LAR, wl, wl_start, wl_step, NW, valid)
    live = valid
    boundary = tl.full((BLOCK,), False, tl.int1)
    moved = tl.full((BLOCK,), False, tl.int1)
    iteration = 0
    while (iteration < HISTORY) & (tl.sum(live.to(tl.int32), 0) > 0):
        inside = (x > lx) & (x < ux) & (y > ly) & (y < uy) & (z > lz) & (z < uz)
        tx = tl.where(dx > 0, (ux-x)/dx, tl.where(dx < 0, (lx-x)/dx, float("inf")))
        ty = tl.where(dy > 0, (uy-y)/dy, tl.where(dy < 0, (ly-y)/dy, float("inf")))
        tz = tl.where(dz > 0, (uz-z)/dz, tl.where(dz < 0, (lz-z)/dz, float("inf")))
        exit_distance = tl.minimum(tx, tl.minimum(ty, tz))
        safe_distance = tl.maximum(0., exit_distance-tl.maximum(.01, 2.e-6*tl.abs(exit_distance)))
        da = -alen*tl.log(random_uniform(photon, seed, step*32))
        ds = -slen*tl.log(random_uniform(photon, seed, step*32+1))
        distance = tl.minimum(da, ds)
        at_limit = live & (step >= max_steps)
        history |= tl.where(at_limit, 1 << 30, 0)
        live &= ~at_limit
        collide = live & inside & (distance < safe_distance)
        # A handoff commits no move and consumes no stream. The exact geometry
        # boundary step reuses these same free paths, preserving conditioning.
        boundary |= live & ~collide
        absorbed = collide & (da <= ds)
        scattered = collide & ~absorbed
        travel = tl.where(collide, distance, 0.)
        x += travel*dx
        y += travel*dy
        z += travel*dz
        t += travel/speed
        if tl.sum(scattered.to(tl.int32), 0) > 0:
            sx, sy, sz, qx, qy, qz = rayleigh_scatter(dx, dy, dz, px, py, pz,
                random_uniform(photon, seed, step*32+2), random_uniform(photon, seed, step*32+3),
                random_uniform(photon, seed, step*32+4), random_uniform(photon, seed, step*32+5))
            dx, dy, dz = tl.where(scattered, sx, dx), tl.where(scattered, sy, dy), tl.where(scattered, sz, dz)
            px, py, pz = tl.where(scattered, qx, px), tl.where(scattered, qy, py), tl.where(scattered, qz, pz)
        history |= tl.where(absorbed, 2, tl.where(scattered, 16, 0))
        moved |= collide
        step += collide.to(tl.int32)
        live = scattered
        iteration += 1
    tl.store(pos+row*3, x, valid)
    tl.store(pos+row*3+1, y, valid)
    tl.store(pos+row*3+2, z, valid)
    tl.store(direction+row*3, dx, valid)
    tl.store(direction+row*3+1, dy, valid)
    tl.store(direction+row*3+2, dz, valid)
    tl.store(pol+row*3, px, valid)
    tl.store(pol+row*3+1, py, valid)
    tl.store(pol+row*3+2, pz, valid)
    tl.store(times+row, t, valid)
    tl.store(flags+row, history, valid)
    tl.store(steps+row, step, valid)
    tl.store(last_instance+row, -1, moved)
    tl.store(last_triangle+row, -1, moved)
    append_rows(row, live, output_rows, output_count)
    append_rows(row, boundary, boundary_rows, boundary_count)


@triton.jit
def boundary_step(pos, direction, pol, times, flags, ids, last_instance, last_triangle,
                  channels, steps, wavelengths, events,
                  rows, active_count, capacity, output_rows, output_count,
                  hit_distances, hit_normals, material_from, material_to, surfaces, instances, triangles, hit_channels,
                  triangle_material1,
                  rindex, absorption, scattering, velocities, surface_model,
                  surface_detect, surface_absorb, surface_diffuse, surface_specular, surface_reemit, surface_cdf,
                  time_offsets, time_x, time_cdf, time_pdf, reemit_side,
                  seed, max_steps, wl_start, wl_step, NW: tl.constexpr, BLOCK: tl.constexpr,
                  analytic_index=None, analytic_primitive=None, analytic_outward=None,
                  wire_geometry=None, box_bounds=None, PROJECT_ANALYTIC: tl.constexpr=False):
    lane = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    count = tl.load(active_count)
    if tl.program_id(0)*BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row = tl.load(rows+lane, valid, 0)
    step = tl.load(steps+row, valid, 0)
    photon = tl.load(ids+row, valid, 0)
    h = tl.load(flags+row, valid, 0)
    at_limit = valid & (step >= max_steps)
    h |= tl.where(at_limit, 1 << 30, 0)
    active = valid & ~at_limit
    distance = tl.load(hit_distances+lane, active, float("inf"))
    incident = tl.load(material_from+lane, active, -1)
    other = tl.load(material_to+lane, active, -1)
    hit = active & (incident >= 0)
    h |= tl.where(active & ~hit, 1, 0)
    wl = tl.load(wavelengths+row, valid, wl_start)
    t = tl.load(times+row, valid, 0.)
    dx = tl.load(direction+row*3, valid, 1.)
    dy = tl.load(direction+row*3+1, valid, 0.)
    dz = tl.load(direction+row*3+2, valid, 0.)
    px = tl.load(pol+row*3, valid, 0.)
    py = tl.load(pol+row*3+1, valid, 1.)
    pz = tl.load(pol+row*3+2, valid, 0.)
    nx = tl.load(hit_normals+lane*3, hit, 0.)
    ny = tl.load(hit_normals+lane*3+1, hit, 0.)
    nz = tl.load(hit_normals+lane*3+2, hit, 1.)
    alen = interpolate(absorption, incident, wl, wl_start, wl_step, NW, hit)
    slen = interpolate(scattering, incident, wl, wl_start, wl_step, NW, hit)
    da = -alen*tl.log(random_uniform(photon, seed, step*32))
    ds = -slen*tl.log(random_uniform(photon, seed, step*32+1))
    absorbed = hit & (da <= ds) & (da <= distance)
    scattered = hit & (ds < da) & (ds <= distance)
    boundary = hit & ~(absorbed | scattered)
    travel = tl.where(hit, tl.minimum(distance, tl.minimum(da, ds)), 0.)
    speed = interpolate(velocities, incident, wl, wl_start, wl_step, NW, hit)
    t += tl.where(hit, travel/tl.maximum(speed, 1.e-30), 0.)
    x = tl.load(pos+row*3, valid, 0.)+travel*dx
    y = tl.load(pos+row*3+1, valid, 0.)+travel*dy
    z = tl.load(pos+row*3+2, valid, 0.)+travel*dz
    h |= tl.where(absorbed, 2, 0)
    if tl.sum(scattered.to(tl.int32), 0) > 0:
        sx, sy, sz, qx, qy, qz = rayleigh_scatter(dx, dy, dz, px, py, pz,
            random_uniform(photon, seed, step*32+2), random_uniform(photon, seed, step*32+3),
            random_uniform(photon, seed, step*32+4), random_uniform(photon, seed, step*32+5))
        dx, dy, dz = tl.where(scattered, sx, dx), tl.where(scattered, sy, dy), tl.where(scattered, sz, dz)
        px, py, pz = tl.where(scattered, qx, px), tl.where(scattered, qy, py), tl.where(scattered, qz, pz)
    h |= tl.where(scattered, 16, 0)
    sid = tl.load(surfaces+lane, boundary, -1)
    surface = boundary & (sid >= 0)
    safe_sid = tl.maximum(sid, 0)
    model = tl.load(surface_model+safe_sid, surface, 0)
    absorb = interpolate(surface_absorb, safe_sid, wl, wl_start, wl_step, NW, surface)
    detect = interpolate(surface_detect, safe_sid, wl, wl_start, wl_step, NW, surface & (model == 0))
    diffuse = interpolate(surface_diffuse, safe_sid, wl, wl_start, wl_step, NW, surface)
    specular = interpolate(surface_specular, safe_sid, wl, wl_start, wl_step, NW, surface)
    u = random_uniform(photon, seed, step*32+6)
    killed = surface & (u < absorb)
    detected = surface & (u >= absorb) & (u < absorb+detect)
    diff = surface & (u >= absorb+detect) & (u < absorb+detect+diffuse)
    spec = surface & (u >= absorb+detect+diffuse) & (u < absorb+detect+diffuse+specular)
    passed = boundary & ~(killed | detected | diff | spec)
    rp = interpolate(surface_reemit, safe_sid, wl, wl_start, wl_step, NW, surface & (model == 2))
    reemit = killed & (model == 2) & (random_uniform(photon, seed, step*32+7) < rp)
    if tl.sum(reemit.to(tl.int32), 0) > 0:
        wl = tl.where(reemit, sample_regular_cdf(surface_cdf, safe_sid, random_uniform(photon, seed, step*32+8),
                                                wl_start, wl_step, NW, reemit), wl)
        t += tl.where(reemit, sample_time_cdf(time_offsets, time_x, time_cdf, time_pdf, safe_sid,
                             random_uniform(photon, seed, step*32+9), reemit).to(tl.float32), 0.)
        tri = tl.load(triangles+lane, reemit, 0)
        m1 = tl.load(triangle_material1+tl.maximum(tri, 0), reemit, 0)
        outward_sign = tl.where(incident == m1, -1., 1.)
        side_prob = tl.load(reemit_side+safe_sid, reemit, .5)
        side = tl.where(random_uniform(photon, seed, step*32+10) < side_prob, -1., 1.)*outward_sign
        sx, sy, sz = hemisphere(side*nx, side*ny, side*nz, random_uniform(photon, seed, step*32+11),
                                 random_uniform(photon, seed, step*32+12), False)
        qx, qy, qz = polarization(sx, sy, sz, random_uniform(photon, seed, step*32+13))
        dx, dy, dz = tl.where(reemit, sx, dx), tl.where(reemit, sy, dy), tl.where(reemit, sz, dz)
        px, py, pz = tl.where(reemit, qx, px), tl.where(reemit, qy, py), tl.where(reemit, qz, pz)
    h |= tl.where(reemit, 128, 0) | tl.where(killed & ~reemit, 8, 0) | tl.where(detected, 4, 0)
    if tl.sum(diff.to(tl.int32), 0) > 0:
        sx, sy, sz = hemisphere(nx, ny, nz, random_uniform(photon, seed, step*32+14),
                                 random_uniform(photon, seed, step*32+15), True)
        qx, qy, qz = polarization(sx, sy, sz, random_uniform(photon, seed, step*32+16))
        dx, dy, dz = tl.where(diff, sx, dx), tl.where(diff, sy, dy), tl.where(diff, sz, dz)
        px, py, pz = tl.where(diff, qx, px), tl.where(diff, qy, py), tl.where(diff, qz, pz)
    h |= tl.where(diff, 32, 0)
    dn, pn = dx*nx+dy*ny+dz*nz, px*nx+py*ny+pz*nz
    dx, dy, dz = tl.where(spec, dx-2*dn*nx, dx), tl.where(spec, dy-2*dn*ny, dy), tl.where(spec, dz-2*dn*nz, dz)
    px, py, pz = tl.where(spec, px-2*pn*nx, px), tl.where(spec, py-2*pn*ny, py), tl.where(spec, pz-2*pn*nz, pz)
    h |= tl.where(spec, 64, 0)
    if tl.sum(passed.to(tl.int32), 0) > 0:
        n1 = interpolate(rindex, incident, wl, wl_start, wl_step, NW, passed)
        n2 = interpolate(rindex, other, wl, wl_start, wl_step, NW, passed)
        n1, n2 = tl.where(passed, n1, 1.), tl.where(passed, n2, 1.)
        sx, sy, sz, qx, qy, qz, reflected, _, _, _ = fresnel_step(dx, dy, dz, px, py, pz, nx, ny, nz, n1, n2,
            random_uniform(photon, seed, step*32+17), random_uniform(photon, seed, step*32+18))
        dx, dy, dz = tl.where(passed, sx, dx), tl.where(passed, sy, dy), tl.where(passed, sz, dz)
        px, py, pz = tl.where(passed, qx, px), tl.where(passed, qy, py), tl.where(passed, qz, pz)
        h |= tl.where(passed, tl.where(reflected, 64, 256), 0)
    inst = tl.load(instances+lane, boundary, -1)
    tri = tl.load(triangles+lane, boundary, -1)
    if PROJECT_ANALYTIC:
        survives_boundary = boundary & ~(detected | (killed & ~reemit))
        wire = survives_boundary & (inst == -1)
        box = survives_boundary & (inst < -1)
        # A float32 flight at detector coordinates (~2 m) can round a point
        # inside a 75 um radius wire after reflection. Reconstruct the actual
        # analytic surface in FP64, then round into the outgoing material.
        # This changes only the representable boundary position, not flight
        # distance/time or any interaction probability.
        bx, by, bz = x.to(tl.float64), y.to(tl.float64), z.to(tl.float64)
        if tl.sum(wire.to(tl.int32), 0) > 0:
            wi = tl.load(analytic_index+lane, wire, 0)
            k = tl.load(analytic_primitive+lane, wire, 0).to(tl.float64)
            ox = tl.load(wire_geometry+wi*14, wire, 0.)
            oy = tl.load(wire_geometry+wi*14+1, wire, 0.)
            oz = tl.load(wire_geometry+wi*14+2, wire, 0.)
            ux = tl.load(wire_geometry+wi*14+3, wire, 0.)
            uy = tl.load(wire_geometry+wi*14+4, wire, 0.)
            uz = tl.load(wire_geometry+wi*14+5, wire, 0.)
            vx = tl.load(wire_geometry+wi*14+6, wire, 0.)
            vy = tl.load(wire_geometry+wi*14+7, wire, 0.)
            vz = tl.load(wire_geometry+wi*14+8, wire, 0.)
            pitch = tl.load(wire_geometry+wi*14+9, wire, 0.)
            v0 = tl.load(wire_geometry+wi*14+10, wire, 0.)
            radius = tl.load(wire_geometry+wi*14+11, wire, 0.)
            umin = tl.load(wire_geometry+wi*14+12, wire, 0.)
            umax = tl.load(wire_geometry+wi*14+13, wire, 0.)
            along = tl.minimum(umax, tl.maximum(umin, (bx-ox)*ux+(by-oy)*uy+(bz-oz)*uz))
            wx = tl.load(analytic_outward+lane*3, wire, 1.).to(tl.float64)
            wy = tl.load(analytic_outward+lane*3+1, wire, 0.).to(tl.float64)
            wz = tl.load(analytic_outward+lane*3+2, wire, 0.).to(tl.float64)
            scale = radius/tl.sqrt(wx*wx+wy*wy+wz*wz)
            bx = tl.where(wire, ox+along*ux+(v0+k*pitch)*vx+scale*wx, bx)
            by = tl.where(wire, oy+along*uy+(v0+k*pitch)*vy+scale*wy, by)
            bz = tl.where(wire, oz+along*uz+(v0+k*pitch)*vz+scale*wz, bz)
        if tl.sum(box.to(tl.int32), 0) > 0:
            bi = -inst-2
            axis = tri//2
            coordinate = tl.load(box_bounds+bi*6+(tri%2)*3+axis, box, 0.).to(tl.float64)
            bx = tl.where(box & (axis == 0), coordinate, bx)
            by = tl.where(box & (axis == 1), coordinate, by)
            bz = tl.where(box & (axis == 2), coordinate, bz)
        side = tl.where(dx*nx+dy*ny+dz*nz >= 0, 1., -1.)
        x = tl.where(wire | box, round_to_material_side(bx, side*nx), x)
        y = tl.where(wire | box, round_to_material_side(by, side*ny), y)
        z = tl.where(wire | box, round_to_material_side(bz, side*nz), z)
    channel = tl.load(hit_channels+lane, detected, -1)
    tl.store(last_instance+row, inst, hit)
    tl.store(last_triangle+row, tri, hit)
    tl.store(channels+row, channel, detected)
    tl.store(pos+row*3, x, valid)
    tl.store(pos+row*3+1, y, valid)
    tl.store(pos+row*3+2, z, valid)
    tl.store(direction+row*3, dx, valid)
    tl.store(direction+row*3+1, dy, valid)
    tl.store(direction+row*3+2, dz, valid)
    tl.store(pol+row*3, px, valid)
    tl.store(pol+row*3+1, py, valid)
    tl.store(pol+row*3+2, pz, valid)
    tl.store(times+row, t, valid)
    tl.store(wavelengths+row, wl, valid)
    tl.store(flags+row, h, valid)
    tl.store(steps+row, step+hit.to(tl.int32), valid)
    append_rows(row, hit & ~(absorbed | detected | (killed & ~reemit)), output_rows, output_count)
