// Geometry camera rays only. Bounds and shared triangles come from Chroma's
// host BVHs. Threaded escape links remove per-ray stacks. No optical transport.
struct Config {
    eye: vec4<f32>, forward: vec4<f32>, right: vec4<f32>, up: vec4<f32>,
    dimensions: vec4<u32>, options: vec4<u32>,
}
@group(0) @binding(0) var<storage, read> scene: array<u32>;
@group(0) @binding(1) var<uniform> cfg: Config;
@group(0) @binding(2) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read_write> hit_records: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> failures: atomic<u32>;
const END: u32 = 0xffffffffu;
const FAR: f32 = 1e30;
struct MeshHit { normal: vec3<f32>, distance: f32, triangle: u32, }
struct Hit {
    normal: vec3<f32>, distance: f32,
    triangle: u32, instance: u32, group: u32, color: u32,
}
fn float_at(i: u32) -> f32 { return bitcast<f32>(scene[i]); }
fn vector_at(i: u32) -> vec3<f32> {
    return vec3<f32>(float_at(i), float_at(i+1u), float_at(i+2u));
}
fn box_hit(lo: vec3<f32>, hi: vec3<f32>, origin: vec3<f32>, direction: vec3<f32>, limit: f32) -> bool {
    var near = 0.; var far = limit;
    for (var axis = 0u; axis < 3u; axis++) {
        if (direction[axis] == 0.) {
            if (origin[axis] < lo[axis] || origin[axis] > hi[axis]) { return false; }
        } else {
            let a = (lo[axis]-origin[axis])/direction[axis];
            let b = (hi[axis]-origin[axis])/direction[axis];
            near = max(near, min(a, b)); far = min(far, max(a, b));
        }
    }
    return near <= far;
}
fn box_entry(lo: vec3<f32>, hi: vec3<f32>, origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    var near = 0.;
    for (var axis = 0u; axis < 3u; axis++) {
        if (direction[axis] != 0.) {
            near = max(near, min((lo[axis]-origin[axis])/direction[axis], (hi[axis]-origin[axis])/direction[axis]));
        }
    }
    return near;
}
fn trace_mesh(header: u32, origin: vec3<f32>, direction: vec3<f32>, limit: f32) -> MeshHit {
    var best = MeshHit(vec3<f32>(0.), limit, END);
    let node_base = scene[header]; let tri_base = scene[header+1u];
    let escapes = scene[header+5u]; let offset = vector_at(header+6u);
    let scale = float_at(header+9u); let count = scene[header+10u];
    var index = 0u; var visits = 0u;
    loop {
        if (index == END) { break; }
        if (index >= count || visits >= count) { atomicAdd(&failures, 1u); break; }
        visits++;
        let base = node_base+index*4u;
        let packed = vec3<u32>(scene[base], scene[base+1u], scene[base+2u]);
        // One extra quantization unit makes f32 slabs conservative. Leaf
        // positions and returned intersections are unchanged.
        let lo = offset+(vec3<f32>(packed & vec3<u32>(65535u))-vec3<f32>(1.))*scale;
        let hi = offset+(vec3<f32>(packed >> vec3<u32>(16u))+vec3<f32>(1.))*scale;
        let word = scene[base+3u]; let child = word & 0x0fffffffu;
        let successor = scene[escapes+index];
        if (!box_hit(lo, hi, origin, direction, best.distance)) { index = successor; continue; }
        if ((word >> 28u) != 0u) { index = child; continue; }
        let v0 = vector_at(tri_base+child*9u);
        let e1 = vector_at(tri_base+child*9u+3u)-v0;
        let e2 = vector_at(tri_base+child*9u+6u)-v0;
        let h = cross(direction, e2); let determinant = dot(e1, h);
        if (abs(determinant) > 1.1920928955078125e-7) {
            let reciprocal = 1./determinant;
            let s = origin-v0; let u = reciprocal*dot(s, h);
            let q = cross(s, e1); let v = reciprocal*dot(direction, q);
            let distance = reciprocal*dot(e2, q);
            if (u >= -1e-6 && u <= 1.000001 && v >= -1e-6 && u+v <= 1.000001 && distance > 1e-6 && distance < best.distance) {
                best = MeshHit(normalize(cross(e1, e2)), distance, child);
            }
        }
        index = successor;
    }
    return best;
}
fn trace(origin: vec3<f32>, direction: vec3<f32>) -> Hit {
    var best = Hit(vec3<f32>(0.), FAR, END, END, END, 0u);
    for (var group = 0u; group < scene[2]; group++) {
        let header = 4u+group*16u; let tlas = scene[header+3u];
        let count = scene[header+12u]; var index = 0u; var visits = 0u;
        loop {
            if (index == END) { break; }
            if (index >= count || visits >= count) { atomicAdd(&failures, 1u); break; }
            visits++;
            let base = tlas+index*8u; let successor = scene[base+7u];
            if (!box_hit(vector_at(base), vector_at(base+4u), origin, direction, best.distance)) {
                index = successor; continue;
            }
            let child = scene[base+3u];
            if ((child & 0x80000000u) == 0u) { index = child; continue; }
            let instance = child & 0x7fffffffu;
            let transform = scene[header+4u]+instance*12u;
            let r0 = vector_at(transform); let r1 = vector_at(transform+3u); let r2 = vector_at(transform+6u);
            let translation = vector_at(transform+9u);
            let translated = origin-translation;
            // Preserve low bits of the large world-coordinate subtraction,
            // then advance the ray close to the instance before rotating it.
            // This avoids a 40 m origin in a centimetre-scale PMT leaf query.
            let v = translated-origin;
            let subtraction_error = (origin-(translated-v))+(-translation-v);
            let entry = box_entry(vector_at(base), vector_at(base+4u), origin, direction);
            let advance = max(0., entry-max(.001, abs(entry)*4e-6));
            let nearby = fma(direction, vec3<f32>(advance), translated)+subtraction_error;
            let local_origin = nearby.x*r0+nearby.y*r1+nearby.z*r2;
            let local_direction = direction.x*r0+direction.y*r1+direction.z*r2;
            let candidate = trace_mesh(header, local_origin, local_direction, best.distance-advance);
            if (candidate.triangle != END) {
                let normal = normalize(vec3<f32>(dot(r0, candidate.normal), dot(r1, candidate.normal), dot(r2, candidate.normal)));
                best = Hit(normal, candidate.distance+advance, candidate.triangle, instance, group, scene[scene[header+2u]+candidate.triangle]);
            }
            index = successor;
        }
    }
    return best;
}
fn hash(input: u32) -> u32 {
    var x = input; x = (x ^ (x >> 16u))*0x7feb352du;
    x = (x ^ (x >> 15u))*0x846ca68bu; return x ^ (x >> 16u);
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = cfg.dimensions.x; let height = cfg.dimensions.y;
    if (id.x >= width || id.y >= height) { return; }
    let pixel = id.y*width+id.x; let pixels = width*height;
    var color = vec3<f32>(0.); var samples = 0u;
    let sample_count = (cfg.dimensions.z-1u-pixel)/pixels+1u;
    for (var sample = 0u; sample < sample_count; sample++) {
        let ray = pixel+sample*pixels;
        var jitter = vec2<f32>(.5);
        if (cfg.options.y != 0u) {
            jitter = vec2<f32>(f32(hash(ray ^ cfg.dimensions.w) & 0xffffffu), f32(hash(ray ^ cfg.dimensions.w ^ 0x9e3779b9u) & 0xffffffu))/16777216.;
        }
        let uv = (vec2<f32>(id.xy)+jitter)/vec2<f32>(f32(width), f32(height));
        let direction = normalize(cfg.forward.xyz+cfg.right.xyz*((2.*uv.x-1.)*f32(width)/f32(height)*cfg.forward.w)+cfg.up.xyz*((1.-2.*uv.y)*cfg.forward.w));
        let hit = trace(cfg.eye.xyz, direction);
        if (hit.triangle == END) {
            color += mix(vec3<f32>(.025, .035, .06), vec3<f32>(.10, .14, .20), uv.y);
        } else {
            let rgb = vec3<f32>(f32((hit.color >> 16u)&255u), f32((hit.color >> 8u)&255u), f32(hit.color&255u))/255.;
            color += rgb*(.22+.78*abs(dot(hit.normal, direction)));
        }
        if (cfg.options.x != 0u && samples == 0u) {
            if (hit.triangle == END) {
                hit_records[pixel*3u] = vec4<f32>(-1.);
                hit_records[pixel*3u+1u] = vec4<f32>(0.);
            } else {
                hit_records[pixel*3u] = vec4<f32>(hit.distance, f32(hit.group), f32(hit.instance), f32(hit.triangle));
                hit_records[pixel*3u+1u] = vec4<f32>(hit.normal, 1.);
            }
            hit_records[pixel*3u+2u] = vec4<f32>(direction, 0.);
        }
        samples++;
    }
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(color/f32(max(samples, 1u)), 1.));
}
