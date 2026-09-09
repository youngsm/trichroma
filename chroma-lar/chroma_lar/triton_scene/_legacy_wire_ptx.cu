
#include "photon.h"
extern "C" __global__ void wire_probe(
    const float3 *positions, const float3 *directions, const WirePlane *planes,
    float *distance, float3 *normal, int *material_from, int *material_to,
    int *surface, int *triangle, int plane_count, int row, int valid)
{
    if (!valid) return;
    Photon p;
    p.position = positions[row];
    p.direction = directions[row];
    float best_distance = triangle[row] == -1 ? 1e30f : distance[row];
    int nplanes = plane_count;
    int analytic_surface = -1;
    int analytic_mat_inner = -1;
    int analytic_mat_outer = -1;
    float3 analytic_normal = make_float3(0.0f,0.0f,0.0f);
    int analytic_plane_idx = -1;
    float analytic_distance = 1e30f; // do not prune analytic by mesh incumbent
    // extra analytic bookkeeping for robust classification
    float3 analytic_normal_raw = make_float3(0.0f,0.0f,0.0f);
    float analytic_dot_raw = 0.0f;

    if (nplanes > 0 && planes != 0) {
        CHROMA_PROF_FUNC_START(CHROMA_PROF_FILL_ANALYTIC);
        for (int ip=0; ip<nplanes; ++ip) {
            const WirePlane *wp = planes + ip;
            if (wp == 0) continue;

            // use precomputed normalized orthonormal frame (FP32)
            const float ux1 = wp->u_norm.x, uy1 = wp->u_norm.y, uz1 = wp->u_norm.z;
            const float vx = wp->v_norm.x, vy = wp->v_norm.y, vz = wp->v_norm.z;
            const float nx = wp->n_norm.x, ny = wp->n_norm.y, nz = wp->n_norm.z;

            float3 w = p.position - wp->origin;
            float dn = p.direction.x*nx  + p.direction.y*ny  + p.direction.z*nz;
            float wn0 = w.x*nx  + w.y*ny  + w.z*nz;

            // EARLY CULL: quick plane distance check
            // if photon is far from plane and moving away, skip this plane
            float plane_dist = fabsf(wn0);
            if (plane_dist > wp->radius + 0.01f) {
                // photon is outside wire envelope
                if (dn * wn0 > 0.0f) continue;  // moving away from plane
                // estimate minimum distance to plane
                float t_plane = -wn0 / dn;
                if (t_plane > best_distance + wp->radius) continue;  // too far
            }

            float du = p.direction.x*ux1 + p.direction.y*uy1 + p.direction.z*uz1;
            float dv = p.direction.x*vx  + p.direction.y*vy  + p.direction.z*vz;
            float wu = w.x*ux1 + w.y*uy1 + w.z*uz1;
            float wv0 = w.x*vx  + w.y*vy  + w.z*vz - wp->v0;

            // u-extent cull (FP32)
            float t_in = -1.0e30f, t_out = 1.0e30f;
            if (fabsf(du) < 1e-7f) {
                if (wu < wp->umin || wu > wp->umax) continue;
            } else {
                float t1 = (wp->umin - wu) / du;
                float t2 = (wp->umax - wu) / du;
                if (t1 > t2) { float tmp=t1; t1=t2; t2=tmp; }
                if (t1 > t_in) t_in = t1;
                if (t2 < t_out) t_out = t2;
                if (t_in > t_out) continue;
            }

            const float pitch = wp->pitch;
            const float inv_pitch = (pitch != 0.0f) ? (1.0f / pitch) : 0.0f;
            const float wire_radius = wp->radius;
            const float wire_thickness = 2.0f * wire_radius;
            const float pad_v = 0.5f * wire_thickness + 1e-5f;
            const float pad_n = 0.5f * wire_thickness + 1e-5f;

            // use precomputed wire index bounds
            int kmin = wp->k_min;
            int kmax = wp->k_max;
            float A = __fmaf_rn(dn, dn, __fmul_rn(dv, dv));

            int k_start = kmin;
            int k_stop = kmax;

            if (kmin <= kmax) {
                const float t_eps = 1.0e-4f;
                float t_lo = fmaxf(t_in, t_eps);
                float t_hi = t_out;
                float best_cap = best_distance;
                if (best_cap < t_hi)
                    t_hi = best_cap;

                if (fabsf(dn) > 1e-7f) {
                    float tn1 = (-pad_n - wn0) / dn;
                    float tn2 = ( pad_n - wn0) / dn;
                    if (tn1 > tn2) { float tmp = tn1; tn1 = tn2; tn2 = tmp; }
                    t_lo = fmaxf(t_lo, tn1);
                    t_hi = fminf(t_hi, tn2);
                } else {
                    if (fabsf(wn0) > pad_n)
                        continue;
                }

                if (t_hi < t_lo)
                    continue;

                if (fabsf(dn) <= 1e-7f && fabsf(dv) > 1e-7f) {
                    float t_span = (pitch + wire_thickness) / fabsf(dv);
                    t_hi = fminf(t_hi, t_lo + t_span);
                }

                float v_entry = wv0 + dv * t_lo;
                float v_exit = wv0 + dv * t_hi;
                float v_lo = fminf(v_entry, v_exit) - pad_v;
                float v_hi = fmaxf(v_entry, v_exit) + pad_v;

                if (wv0 - pad_v < v_lo)
                    v_lo = wv0 - pad_v;
                if (wv0 + pad_v > v_hi)
                    v_hi = wv0 + pad_v;

                int k_lo = (int)floorf(v_lo * inv_pitch);
                int k_hi = (int)ceilf(v_hi * inv_pitch);

                if (k_lo < kmin)
                    k_lo = kmin;
                if (k_hi > kmax)
                    k_hi = kmax;
                if (k_lo > k_hi)
                    continue;

                k_start = k_lo;
                k_stop = k_hi;
            }

            const float r2_wire = wire_radius*wire_radius;
            for (int k=k_start; k<=k_stop; ++k) {
                float wv = wv0 - (float)k * pitch;
                float B = wv*dv + wn0*dn;
                float C = wv*wv + wn0*wn0 - r2_wire;
                float disc = B*B - A*C;
                if (disc < 0.0f) continue;
                float sqrt_disc = sqrtf(disc);
                float t_small = (-B - sqrt_disc) / A;
                float t_large = (-B + sqrt_disc) / A;
                // robust epsilon to avoid immediate self-hit at boundary
                const float t_min = 1.0e-4f; // mm
                const float r2_0 = wv*wv + wn0*wn0; // squared radius at ray start for this k
                const float eps0 = fmaxf(1e-12f, 1e-6f * r2_wire);

                float t;
                if (r2_0 > r2_wire + eps0) {
                    // origin outside: require a valid forward entry root; otherwise skip
                    if (t_small <= t_min) continue;
                    t = t_small;
                } else if (r2_0 < r2_wire - eps0) {
                    // origin inside: must use forward exit root
                    if (t_large <= t_min) continue;
                    t = t_large;
                } else {
                    // origin numerically on boundary: take a small step forward
                    t = t_min;
                }
                float uc = wu + du * t;
                if (uc < wp->umin || uc > wp->umax) continue;
                if (t >= analytic_distance) continue;
                // enforce u-slab window
                if (t < t_in || t > t_out) continue;

                float vn_hit = wv + dv * t;
                float nn_hit = wn0 + dn * t;
                float len = sqrtf(vn_hit*vn_hit + nn_hit*nn_hit);
                if (len <= 0.0f) continue;
                float inv_len = 1.0f / len;
                float3 n_local = make_float3((vn_hit*inv_len)*vx + (nn_hit*inv_len)*nx,
                                             (vn_hit*inv_len)*vy + (nn_hit*inv_len)*ny,
                                             (vn_hit*inv_len)*vz + (nn_hit*inv_len)*nz);
                float3 n_world_raw = n_local; // outward cylinder normal (unoriented)
                float dot_raw_local = dot(n_world_raw, -p.direction);

                analytic_distance = t;
                analytic_surface = wp->surface_index;
                analytic_mat_inner = wp->material_inner_index;
                analytic_mat_outer = wp->material_outer_index;
                analytic_normal_raw = n_world_raw;
                analytic_dot_raw = dot_raw_local;
                analytic_plane_idx = ip;
            }
        }

        CHROMA_PROF_FUNC_END(CHROMA_PROF_FILL_ANALYTIC);
    }

    bool use_analytic = false;
    if (analytic_surface >= 0) {
        // use FP32 comparison with small epsilon
        use_analytic = (analytic_distance + 1e-6f < best_distance);
    }


    if (use_analytic) {
        distance[row] = analytic_distance;
        bool outside = analytic_dot_raw > 0.0f;
        normal[row] = outside ? analytic_normal_raw : -analytic_normal_raw;
        material_from[row] = outside ? analytic_mat_outer : analytic_mat_inner;
        material_to[row] = outside ? analytic_mat_inner : analytic_mat_outer;
        surface[row] = analytic_surface;
        triangle[row] = -2;
    }
}
