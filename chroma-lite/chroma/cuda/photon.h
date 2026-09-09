#ifndef __PHOTON_H__
#define __PHOTON_H__

#include "stdio.h"
#include "linalg.h"
#include "rotate.h"
#include "random.h"
#include "physical_constants.h"
#include "mesh.h"
#include "geometry.h"
#include "cx.h"

#define WEIGHT_LOWER_THRESHOLD 0.0001f

#ifndef CHROMA_FORCE_SCATTER_AT_PASS
#define CHROMA_FORCE_SCATTER_AT_PASS 0
#endif

struct Photon
{
    float3 position;
    float3 direction;
    float3 polarization;
    float wavelength;
    float time;
  
    float weight;
  
    unsigned short history;

    int last_hit_triangle;
    
    unsigned int evidx;
};

struct State
{
    bool inside_to_outside;

    float3 surface_normal;

    float refractive_index1, refractive_index2;
    float absorption_length;
    float scattering_length;
    float reemission_prob;
    Material *material1;

    int surface_index;

    float distance_to_boundary;
};

enum
{
    NO_HIT           = 0x1 << 0,
    BULK_ABSORB      = 0x1 << 1,
    SURFACE_DETECT   = 0x1 << 2,
    SURFACE_ABSORB   = 0x1 << 3,
    RAYLEIGH_SCATTER = 0x1 << 4,
    REFLECT_DIFFUSE  = 0x1 << 5,
    REFLECT_SPECULAR = 0x1 << 6,
    SURFACE_REEMIT   = 0x1 << 7,
    SURFACE_TRANSMIT = 0x1 << 8,
    BULK_REEMIT      = 0x1 << 9,
    CHERENKOV        = 0x1 << 10,
    SCINTILLATION    = 0x1 << 11,
    NAN_ABORT        = 0x1 << 15
}; // processes

enum { BREAK, CONTINUE, PASS }; // return value from propagate_to_boundary

__device__ int
convert(int c)
{
    if (c & 0x80)
        return (0xFFFFFF00 | c);
    else
        return c;
}

__device__ float
get_theta(const float3 &a, const float3 &b)
{
    return acosf(fmaxf(-1.0f,fminf(1.0f,dot(a,b))));
}

__device__ void
fill_state(State &s, Photon &p, const Geometry *g)
{
    // 1) query mesh BVH for nearest triangle boundary
    int mesh_triangle = intersect_mesh(p.position, p.direction, g,
                                       s.distance_to_boundary,
                                       p.last_hit_triangle);
    float best_distance = (mesh_triangle == -1) ? 1e30f : s.distance_to_boundary;

    // 2) analytic wire-plane candidate (if present)
    int nplanes = g->nwireplanes;
    int analytic_surface = -1;
    int analytic_mat_inner = -1;
    int analytic_mat_outer = -1;
    float3 analytic_normal = make_float3(0.0f,0.0f,0.0f);
    int analytic_plane_idx = -1;
    float analytic_distance = 1e30f; // do not prune analytic by mesh incumbent
    // extra analytic bookkeeping for robust classification
    float3 analytic_normal_raw = make_float3(0.0f,0.0f,0.0f);
    float analytic_dot_raw = 0.0f;

    if (nplanes > 0 && g->wireplanes != 0) {
        CHROMA_PROF_FUNC_START(CHROMA_PROF_FILL_ANALYTIC);
        for (int ip=0; ip<nplanes; ++ip) {
            const WirePlane *wp = g->wireplanes[ip];
            if (wp == 0) continue;

            // orthonormal frame in FP64
            const double ux = (double)wp->u.x, uy = (double)wp->u.y, uz = (double)wp->u.z;
            const double vx0 = (double)wp->v.x, vy0 = (double)wp->v.y, vz0 = (double)wp->v.z;
            const double un = 1.0 / sqrt(ux*ux + uy*uy + uz*uz);
            const double ux1 = ux*un, uy1 = uy*un, uz1 = uz*un;
            const double vdotu = vx0*ux1 + vy0*uy1 + vz0*uz1;
            const double vx1 = vx0 - vdotu*ux1;
            const double vy1 = vy0 - vdotu*uy1;
            const double vz1 = vz0 - vdotu*uz1;
            const double vn = 1.0 / sqrt(vx1*vx1 + vy1*vy1 + vz1*vz1);
            const double vx = vx1*vn, vy = vy1*vn, vz = vz1*vn;
            const double nx = uy1*vz - uz1*vy;
            const double ny = uz1*vx - ux1*vz;
            const double nz = ux1*vy - uy1*vx;

            float3 w = p.position - wp->origin;
            double du = (double)p.direction.x*ux1 + (double)p.direction.y*uy1 + (double)p.direction.z*uz1;
            double dv = (double)p.direction.x*vx  + (double)p.direction.y*vy  + (double)p.direction.z*vz;
            double dn = (double)p.direction.x*nx  + (double)p.direction.y*ny  + (double)p.direction.z*nz;
            double wu = (double)w.x*ux1 + (double)w.y*uy1 + (double)w.z*uz1;
            double wv0 = (double)w.x*vx  + (double)w.y*vy  + (double)w.z*vz - (double)wp->v0;
            double wn0 = (double)w.x*nx  + (double)w.y*ny  + (double)w.z*nz;

            // u-extent cull (FP64)
            double t_in = -1.0e300, t_out = 1.0e300;
            if (fabs(du) < 1e-15) {
                if (wu < (double)wp->umin || wu > (double)wp->umax) continue;
            } else {
                double t1 = ((double)wp->umin - wu) / du;
                double t2 = ((double)wp->umax - wu) / du;
                if (t1 > t2) { double tmp=t1; t1=t2; t2=tmp; }
                if (t1 > t_in) t_in = t1;
                if (t2 < t_out) t_out = t2;
                if (t_in > t_out) continue;
            }

            const double pitch = (double)wp->pitch;
            const double inv_pitch = (pitch != 0.0) ? (1.0 / pitch) : 0.0;
            const double wire_radius = (double)wp->radius;
            const double wire_thickness = 2.0 * wire_radius;
            const double pad_v = 0.5 * wire_thickness + 1e-6;
            const double pad_n = 0.5 * wire_thickness + 1e-6;

            int kmin = (int)ceil(((double)wp->vmin - (double)wp->v0) / pitch);
            int kmax = (int)floor(((double)wp->vmax - (double)wp->v0) / pitch);
            double A = dv*dv + dn*dn;

            int k_start = kmin;
            int k_stop = kmax;

            if (kmin <= kmax) {
                const double t_eps = 1.0e-4;
                double t_lo = fmax(t_in, t_eps);
                double t_hi = t_out;
                double best_cap = (double)best_distance;
                if (best_cap < t_hi)
                    t_hi = best_cap;

                if (fabs(dn) > 1e-12) {
                    double tn1 = (-pad_n - wn0) / dn;
                    double tn2 = ( pad_n - wn0) / dn;
                    if (tn1 > tn2) { double tmp = tn1; tn1 = tn2; tn2 = tmp; }
                    t_lo = fmax(t_lo, tn1);
                    t_hi = fmin(t_hi, tn2);
                } else {
                    if (fabs(wn0) > pad_n)
                        continue;
                }

                if (t_hi < t_lo)
                    continue;

                if (fabs(dn) <= 1e-12 && fabs(dv) > 1e-12) {
                    double t_span = (pitch + wire_thickness) / fabs(dv);
                    t_hi = fmin(t_hi, t_lo + t_span);
                }

                double v_entry = wv0 + dv * t_lo;
                double v_exit = wv0 + dv * t_hi;
                double v_lo = fmin(v_entry, v_exit) - pad_v;
                double v_hi = fmax(v_entry, v_exit) + pad_v;

                if (wv0 - pad_v < v_lo)
                    v_lo = wv0 - pad_v;
                if (wv0 + pad_v > v_hi)
                    v_hi = wv0 + pad_v;

                long long k_lo = (long long)floor(v_lo * inv_pitch);
                long long k_hi = (long long)ceil(v_hi * inv_pitch);

                if (k_lo < kmin)
                    k_lo = kmin;
                if (k_hi > kmax)
                    k_hi = kmax;
                if (k_lo > k_hi)
                    continue;

                k_start = (int)k_lo;
                k_stop = (int)k_hi;
            }

            for (int k=k_start; k<=k_stop; ++k) {
                double wv = wv0 - (double)k * pitch;
                double B = wv*dv + wn0*dn;
                double C = wv*wv + wn0*wn0 - wire_radius*wire_radius;
                double disc = B*B - A*C;
                if (disc < 0.0) continue;
                double sqrt_disc = sqrt(disc);
                double t_small = (-B - sqrt_disc) / A;
                double t_large = (-B + sqrt_disc) / A;
                // robust epsilon to avoid immediate self-hit at boundary
                const double t_min = 1.0e-4; // mm
                const double r2_wire = wire_radius*wire_radius;
                const double r2_0 = wv*wv + wn0*wn0; // squared radius at ray start for this k
                const double eps0 = fmax(1e-18, 1e-12 * r2_wire);

                double t;
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
                double uc = wu + du * t;
                if (uc < wp->umin || uc > wp->umax) continue;
                if ((float)t >= analytic_distance) continue;
                // enforce u-slab window
                if (t < t_in || t > t_out) continue;

                double vn_hit = wv + dv * t;
                double nn_hit = wn0 + dn * t;
                double len = sqrt(vn_hit*vn_hit + nn_hit*nn_hit);
                if (len <= 0.0) continue;
                float3 n_local = make_float3((float)((vn_hit/len)*vx + (nn_hit/len)*nx),
                                             (float)((vn_hit/len)*vy + (nn_hit/len)*ny),
                                             (float)((vn_hit/len)*vz + (nn_hit/len)*nz));
                float3 n_world_raw = n_local; // outward cylinder normal (unoriented)
                float dot_raw_local = dot(n_world_raw, -p.direction);

                analytic_distance = (float)t;
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
        double da = (double)analytic_distance;
        double dm = (double)best_distance;
        use_analytic = (da + 1e-12 < dm);
    }

    Material *material1 = 0;
    Material *material2 = 0;
    CHROMA_PROF_FUNC_START(CHROMA_PROF_FILL_MATERIAL);
    if (use_analytic) {
        // printf("Analytic hit: distance=%.6f mesh_distance=%.6f plane_idx=%d wavelength=%.1f\n", 
        //        analytic_distance, best_distance, analytic_plane_idx, p.wavelength);

        // choose analytic hit
        s.distance_to_boundary = analytic_distance;
        s.surface_index = analytic_surface;
        p.last_hit_triangle = -2;
        // classification using raw outward normal from chosen wire and incident ray
        if (analytic_plane_idx >= 0) {
            const WirePlane *wp = g->wireplanes[analytic_plane_idx];
            // // boundary point in FP64 (for logging)
            // const double xb_x = (double)p.position.x + (double)s.distance_to_boundary * (double)p.direction.x;
            // const double xb_y = (double)p.position.y + (double)s.distance_to_boundary * (double)p.direction.y;
            // const double xb_z = (double)p.position.z + (double)s.distance_to_boundary * (double)p.direction.z;

            // decide outside/inside purely from raw normal and ray (physically robust)
            const float dot_raw = analytic_dot_raw;
            bool outside_now = (dot_raw > 0.0f); // outside->inside if true
            if (outside_now) {
                material1 = g->materials[analytic_mat_outer];
                material2 = g->materials[analytic_mat_inner];
                s.surface_normal = analytic_normal_raw; // already outward; faces incoming when outside
                s.inside_to_outside = false;
            } else {
                material1 = g->materials[analytic_mat_inner];
                material2 = g->materials[analytic_mat_outer];
                s.surface_normal = -analytic_normal_raw; // flip to face incoming when inside
                s.inside_to_outside = true;
            }

            // const double ux = (double)wp->u.x, uy = (double)wp->u.y, uz = (double)wp->u.z;
            // const double vx0 = (double)wp->v.x, vy0 = (double)wp->v.y, vz0 = (double)wp->v.z;
            // const double un = 1.0 / sqrt(ux*ux + uy*uy + uz*uz);
            // const double ux1 = ux*un, uy1 = uy*un, uz1 = uz*un;
            // const double vdotu = vx0*ux1 + vy0*uy1 + vz0*uz1;
            // const double vx1 = vx0 - vdotu*ux1;
            // const double vy1 = vy0 - vdotu*uy1;
            // const double vz1 = vz0 - vdotu*uz1;
            // const double vn = 1.0 / sqrt(vx1*vx1 + vy1*vy1 + vz1*vz1);
            // const double vx = vx1*vn, vy = vy1*vn, vz = vz1*vn;
            // const double nx = uy1*vz - uz1*vy;
            // const double ny = uz1*vx - ux1*vz;
            // const double nz = ux1*vy - uy1*vx;
            // const double wbx = xb_x - (double)wp->origin.x;
            // const double wby = xb_y - (double)wp->origin.y;
            // const double wbz = xb_z - (double)wp->origin.z;
            // const double wv_b = wbx*vx + wby*vy + wbz*vz;
            // const double wn_b = wbx*nx + wby*ny + wbz*nz;
            // const double dv_b = wv_b - (double)wp->v0;
            // const double pitch = (double)wp->pitch;
            // const double kf = floor(dv_b / pitch + 0.5);
            // const double yb = dv_b - kf * pitch;
            // const double r2_b = yb*yb + wn_b*wn_b;
            // const double r2_wire = (double)wp->radius * (double)wp->radius;

            // printf("Boundary classification: r2_b=%.6e r2_wire=%.6e diff=%.6e dot=%.6f outside=%d\n",
            //     r2_b, r2_wire, fabs(r2_b - r2_wire), dot_raw, outside_now);
            // printf("Boundary Xb=(%.6f,%.6f,%.6f)\n", xb_x, xb_y, xb_z);

        } else {
            // fallback to dot-based if plane index missing
            if (dot(s.surface_normal,-p.direction) > 0.0f) {
                material1 = g->materials[analytic_mat_outer];
                material2 = g->materials[analytic_mat_inner];
                s.inside_to_outside = false;
            } else {
                material1 = g->materials[analytic_mat_inner];
                material2 = g->materials[analytic_mat_outer];
                s.surface_normal = -s.surface_normal;
                s.inside_to_outside = true;
            }
        }
    } else if (mesh_triangle != -1) {
        // use mesh hit
        p.last_hit_triangle = mesh_triangle;
        Triangle t = get_triangle(g, p.last_hit_triangle);

        unsigned int material_code = g->material_codes[p.last_hit_triangle];
        int inner_material_index = convert(0xFF & (material_code >> 24));
        int outer_material_index = convert(0xFF & (material_code >> 16));
        s.surface_index = convert(0xFF & (material_code >> 8));

        float3 v01 = t.v1 - t.v0;
        float3 v12 = t.v2 - t.v1;
        s.surface_normal = normalize(cross(v01, v12));

        if (dot(s.surface_normal,-p.direction) > 0.0f) {
            material1 = g->materials[outer_material_index];
            material2 = g->materials[inner_material_index];
            s.inside_to_outside = false;
        } else {
            material1 = g->materials[inner_material_index];
            material2 = g->materials[outer_material_index];
            s.surface_normal = -s.surface_normal;
            s.inside_to_outside = true;
        }
    } else {
        CHROMA_PROF_FUNC_END(CHROMA_PROF_FILL_MATERIAL);
        // no hit at all
        p.last_hit_triangle = -1;
        p.history |= NO_HIT;
        return;
    }

    s.refractive_index1 = interp_property(material1, p.wavelength,
                                          material1->refractive_index);
    s.refractive_index2 = interp_property(material2, p.wavelength,
                                          material2->refractive_index);
    s.absorption_length = interp_property(material1, p.wavelength,
                                          material1->absorption_length);
    s.scattering_length = interp_property(material1, p.wavelength,
                                          material1->scattering_length);
    s.material1 = material1;
    CHROMA_PROF_FUNC_END(CHROMA_PROF_FILL_MATERIAL);
} // fill_state

__device__ float3
pick_new_direction(float3 axis, float theta, float phi)
{
    // Taken from SNOMAN rayscatter.for
    float cos_theta, sin_theta;
    sincosf(theta, &sin_theta, &cos_theta);
    float cos_phi, sin_phi;
    sincosf(phi, &sin_phi, &cos_phi);
        
    float sin_axis_theta = sqrt(1.0f - axis.z*axis.z);
    float cos_axis_phi, sin_axis_phi;
        
    if (isnan(sin_axis_theta) || sin_axis_theta < 0.00001f) {
        cos_axis_phi = 1.0f;
        sin_axis_phi = 0.0f;
    }
    else {
        cos_axis_phi = axis.x / sin_axis_theta;
        sin_axis_phi = axis.y / sin_axis_theta;
    }

    float dirx = cos_theta*axis.x +
        sin_theta*(axis.z*cos_phi*cos_axis_phi - sin_phi*sin_axis_phi);
    float diry = cos_theta*axis.y +
        sin_theta*(cos_phi*axis.z*sin_axis_phi + sin_phi*cos_axis_phi);
    float dirz = cos_theta*axis.z - sin_theta*cos_phi*sin_axis_theta;

    return make_float3(dirx, diry, dirz);
}

__device__ void
rayleigh_scatter(Photon &p, ChromaPhotonRng &rng)
{
    float cos_theta = 2.0f*cosf((acosf(1.0f - 2.0f*chroma_photon_uniform(&rng))-2*PI)/3.0f);
    if (cos_theta > 1.0f)
        cos_theta = 1.0f;
    else if (cos_theta < -1.0f)
        cos_theta = -1.0f;

    float theta = acosf(cos_theta);
    float phi = uniform(&rng, 0.0f, 2.0f * PI);

    p.direction = pick_new_direction(p.polarization, theta, phi);

    if (1.0f - fabsf(cos_theta) < 1e-6f) {
        p.polarization = pick_new_direction(p.polarization, PI/2.0f, phi);
    }
    else {
        // linear combination of old polarization and new direction
        p.polarization = p.polarization - cos_theta * p.direction;
    }

    p.direction /= norm(p.direction);
    p.polarization /= norm(p.polarization);
} // scatter

__device__
int propagate_to_boundary(Photon &p, State &s, ChromaPhotonRng &rng,
                          bool use_weights=false, int scatter_first=0)
{
    float absorption_distance = -s.absorption_length*logf(chroma_photon_uniform(&rng));
    float scattering_distance = -s.scattering_length*logf(chroma_photon_uniform(&rng));

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD) // Prevent absorption
        absorption_distance = 1e30;
    else
        use_weights = false;

    if (scatter_first == 1) {
        // Force scatter
        float scatter_prob = 1.0f - expf(-s.distance_to_boundary/s.scattering_length);

        if (scatter_prob > WEIGHT_LOWER_THRESHOLD) {
            int i=0;
            const int max_i = 1000;
            while (i < max_i && scattering_distance > s.distance_to_boundary) {
                scattering_distance = -s.scattering_length*logf(chroma_photon_uniform(&rng));
                i++;
            }
            p.weight *= scatter_prob;
        }

    } else if (scatter_first == -1) {
        // Prevent scatter
        float no_scatter_prob = expf(-s.distance_to_boundary/s.scattering_length);

        if (no_scatter_prob > WEIGHT_LOWER_THRESHOLD) {
            int i=0;
            const int max_i = 1000;
            while (i < max_i && scattering_distance <= s.distance_to_boundary) {
                scattering_distance = -s.scattering_length*logf(chroma_photon_uniform(&rng));
                i++;
            }
            p.weight *= no_scatter_prob;
        }
    }

    if (absorption_distance <= scattering_distance) {
        if (absorption_distance <= s.distance_to_boundary) {
            p.time += absorption_distance/(SPEED_OF_LIGHT/s.refractive_index1);
            p.position += absorption_distance*p.direction;
            
            if (s.material1->num_comp == 0) {
                p.last_hit_triangle = -1;
                p.history |= BULK_ABSORB;
                chroma_photon_mark_process(
                    &rng, CHROMA_TAPE_PROCESS_BULK_ABSORB);
                return BREAK;
            }
            
            float uniform_sample_comp = chroma_photon_uniform(&rng);
            float prob = 0.0f;
            int comp;
            for (comp = 0; ; comp++) {
                float comp_abs = interp_property(s.material1, p.wavelength, s.material1->comp_absorption_length[comp]);
                prob += s.absorption_length/comp_abs;
                if (uniform_sample_comp < prob or comp+1 == s.material1->num_comp) break;
            }
            
            float uniform_sample_reemit = chroma_photon_uniform(&rng);
            float comp_reemit_prob = interp_property(s.material1, p.wavelength, s.material1->comp_reemission_prob[comp]);
            if (uniform_sample_reemit < comp_reemit_prob) {
                p.wavelength = sample_cdf(&rng, s.material1->wavelength_n, 
                                          s.material1->wavelength_start,
                                          s.material1->wavelength_step,
                                          s.material1->comp_reemission_wvl_cdf[comp]);
                p.time += sample_cdf(&rng, s.material1->time_n, 
                                          s.material1->time_start,
                                          s.material1->time_step,
                                          s.material1->comp_reemission_time_cdf[comp]);
                p.direction = uniform_sphere(&rng);
                p.polarization = cross(uniform_sphere(&rng), p.direction);
                p.polarization /= norm(p.polarization);
                p.last_hit_triangle = -1;
                p.history |= BULK_REEMIT;
                chroma_photon_mark_process(
                    &rng, CHROMA_TAPE_PROCESS_BULK_REEMIT);
                return CONTINUE;
            } // photon is reemitted isotropically
            else {
                p.last_hit_triangle = -1;
                p.history |= BULK_ABSORB;
                chroma_photon_mark_process(
                    &rng, CHROMA_TAPE_PROCESS_BULK_ABSORB);
                return BREAK;
            } // photon is absorbed in material1
        }
    }
    else {
        if (scattering_distance <= s.distance_to_boundary) {

            // Scale weight by absorption probability along this distance
            if (use_weights)
                p.weight *= expf(-scattering_distance/s.absorption_length);

            p.time += scattering_distance/(SPEED_OF_LIGHT/s.refractive_index1);
            p.position += scattering_distance*p.direction;

            rayleigh_scatter(p, rng);

            p.history |= RAYLEIGH_SCATTER;
            chroma_photon_mark_process(
                &rng, CHROMA_TAPE_PROCESS_BULK_SCATTER);

            p.last_hit_triangle = -1;

            return CONTINUE;
        } // photon is scattered in material1
    } // if scattering_distance < absorption_distance

    // Scale weight by absorption probability along this distance
    if (use_weights)
        p.weight *= expf(-s.distance_to_boundary/s.absorption_length);

    p.position += s.distance_to_boundary*p.direction;
    p.time += s.distance_to_boundary/(SPEED_OF_LIGHT/s.refractive_index1);

    return PASS;

} // propagate_to_boundary

__noinline__ __device__ void
propagate_at_boundary(Photon &p, State &s, ChromaPhotonRng &rng)
{
    float incident_angle = get_theta(s.surface_normal,-p.direction);
    float refracted_angle = asinf(sinf(incident_angle)*s.refractive_index1/s.refractive_index2);

    float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    float incident_plane_normal_length = norm(incident_plane_normal);

    // printf("Fresnel: inc=%.6f n1=%.6f n2=%.6f last=%d\n", incident_angle, s.refractive_index1, s.refractive_index2, p.last_hit_triangle);

    // Photons at normal incidence do not have a unique plane of incidence,
    // so we have to pick the plane normal to be the polarization vector
    // to get the correct logic below
    if (incident_plane_normal_length < 1e-6f)
        incident_plane_normal = p.polarization;
    else
        incident_plane_normal /= incident_plane_normal_length;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);
    float normal_probability = normal_coefficient*normal_coefficient;

    float reflection_coefficient;
    if (chroma_photon_uniform(&rng) < normal_probability) {
        // photon polarization normal to plane of incidence
        reflection_coefficient = -sinf(incident_angle-refracted_angle)/sinf(incident_angle+refracted_angle);

        if ((chroma_photon_uniform(&rng) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle)) {
            p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
                        
            p.history |= REFLECT_SPECULAR;
            chroma_photon_mark_process(
                &rng, CHROMA_TAPE_PROCESS_DIELECTRIC_REFLECT);
            // printf("Fresnel REFLECT last=%d\n", p.last_hit_triangle);
        }
        else {
            p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
            chroma_photon_mark_process(
                &rng, CHROMA_TAPE_PROCESS_DIELECTRIC_TRANSMIT);
            // printf("Fresnel TRANSMIT last=%d\n", p.last_hit_triangle);
        }

        p.polarization = incident_plane_normal;
    }
    else {
        // photon polarization parallel to plane of incidence
        reflection_coefficient = tanf(incident_angle-refracted_angle)/tanf(incident_angle+refracted_angle);

        if ((chroma_photon_uniform(&rng) < reflection_coefficient*reflection_coefficient) || isnan(refracted_angle)) {
            p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);
                        
            p.history |= REFLECT_SPECULAR;
            chroma_photon_mark_process(
                &rng, CHROMA_TAPE_PROCESS_DIELECTRIC_REFLECT);
            // printf("Fresnel REFLECT last=%d\n", p.last_hit_triangle);

        }
        else {
            p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
            chroma_photon_mark_process(
                &rng, CHROMA_TAPE_PROCESS_DIELECTRIC_TRANSMIT);
            // printf("Fresnel TRANSMIT last=%d\n", p.last_hit_triangle);
        }

        p.polarization = cross(incident_plane_normal, p.direction);
        p.polarization /= norm(p.polarization);
    }

} // propagate_at_boundary

__device__ int
propagate_at_specular_reflector(Photon &p, State &s)
{
    float incident_angle = get_theta(s.surface_normal, -p.direction);
    float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    incident_plane_normal /= norm(incident_plane_normal);

    p.direction = rotate(s.surface_normal, incident_angle, incident_plane_normal);

    p.history |= REFLECT_SPECULAR;

    return CONTINUE;
} // propagate_at_specular_reflector

#ifdef CHROMA_USE_RANDOM_TAPE
__device__ int
propagate_at_specular_reflector(Photon &p, State &s, ChromaPhotonRng *rng)
{
    int command = propagate_at_specular_reflector(p, s);
    chroma_photon_mark_process(
        rng, CHROMA_TAPE_PROCESS_SURFACE_SPECULAR);
    return command;
}
#define CHROMA_PROPAGATE_SPECULAR(p, s, rng) \
    propagate_at_specular_reflector(p, s, &(rng))
#else
#define CHROMA_PROPAGATE_SPECULAR(p, s, rng) \
    propagate_at_specular_reflector(p, s)
#endif

__device__ int
propagate_at_diffuse_reflector(Photon &p, State &s, ChromaPhotonRng &rng)
{
    float ndotv;
    do {
        p.direction = uniform_sphere(&rng);
        ndotv = dot(p.direction, s.surface_normal);
        if (ndotv < 0.0f) {
            p.direction = -p.direction;
            ndotv = -ndotv;
        }
#ifdef CHROMA_USE_RANDOM_TAPE
    } while (! (chroma_photon_uniform(&rng) < ndotv) &&
             !chroma_photon_rng_failed(&rng));
#else
    } while (! (chroma_photon_uniform(&rng) < ndotv) );
#endif

    p.polarization = cross(uniform_sphere(&rng), p.direction);
    p.polarization /= norm(p.polarization);

    p.history |= REFLECT_DIFFUSE;
    chroma_photon_mark_process(
        &rng, CHROMA_TAPE_PROCESS_SURFACE_DIFFUSE);

    return CONTINUE;
} // propagate_at_diffuse_reflector

__device__ int
propagate_complex(Photon &p, State &s, ChromaPhotonRng &rng, Surface* surface, bool use_weights=false)
{
    float detect = interp_property(surface, p.wavelength, surface->detect);
    float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    float n2_eta = interp_property(surface, p.wavelength, surface->eta);
    float n2_k = interp_property(surface, p.wavelength, surface->k);

    // thin film optical model, adapted from RAT PMT optical model by P. Jones
    cuFloatComplex n1 = make_cuFloatComplex(s.refractive_index1, 0.0f);
    cuFloatComplex n2 = make_cuFloatComplex(n2_eta, n2_k);
    cuFloatComplex n3 = make_cuFloatComplex(s.refractive_index2, 0.0f);

    float cos_t1 = dot(p.direction, s.surface_normal);
    if (cos_t1 < 0.0f)
        cos_t1 = -cos_t1;
    float theta = acosf(cos_t1);

    cuFloatComplex cos1 = make_cuFloatComplex(cosf(theta), 0.0f);
    cuFloatComplex sin1 = make_cuFloatComplex(sinf(theta), 0.0f);

    float e = 2.0f * PI * surface->thickness / p.wavelength;
    cuFloatComplex ratio13sin = cuCmulf(cuCmulf(cuCdivf(n1, n3), cuCdivf(n1, n3)), cuCmulf(sin1, sin1));
    cuFloatComplex cos3 = cuCsqrtf(cuCsubf(make_cuFloatComplex(1.0f,0.0f), ratio13sin));
    cuFloatComplex ratio12sin = cuCmulf(cuCmulf(cuCdivf(n1, n2), cuCdivf(n1, n2)), cuCmulf(sin1, sin1));
    cuFloatComplex cos2 = cuCsqrtf(cuCsubf(make_cuFloatComplex(1.0f,0.0f), ratio12sin));
    float u = cuCrealf(cuCmulf(n2, cos2));
    float v = cuCimagf(cuCmulf(n2, cos2));

    // s polarization
    cuFloatComplex s_n1c1 = cuCmulf(n1, cos1);
    cuFloatComplex s_n2c2 = cuCmulf(n2, cos2);
    cuFloatComplex s_n3c3 = cuCmulf(n3, cos3);
    cuFloatComplex s_r12 = cuCdivf(cuCsubf(s_n1c1, s_n2c2), cuCaddf(s_n1c1, s_n2c2));
    cuFloatComplex s_r23 = cuCdivf(cuCsubf(s_n2c2, s_n3c3), cuCaddf(s_n2c2, s_n3c3));
    cuFloatComplex s_t12 = cuCdivf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), s_n1c1), cuCaddf(s_n1c1, s_n2c2));
    cuFloatComplex s_t23 = cuCdivf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), s_n2c2), cuCaddf(s_n2c2, s_n3c3));
    cuFloatComplex s_g = cuCdivf(s_n3c3, s_n1c1);

    float s_abs_r12 = cuCabsf(s_r12);
    float s_abs_r23 = cuCabsf(s_r23);
    float s_abs_t12 = cuCabsf(s_t12);
    float s_abs_t23 = cuCabsf(s_t23);
    float s_arg_r12 = cuCargf(s_r12);
    float s_arg_r23 = cuCargf(s_r23);
    float s_exp1 = exp(2.0f * v * e);

    float s_exp2 = 1.0f / s_exp1;
    float s_denom = s_exp1 +
                    s_abs_r12 * s_abs_r12 * s_abs_r23 * s_abs_r23 * s_exp2 +
                    2.0f * s_abs_r12 * s_abs_r23 * cosf(s_arg_r23 + s_arg_r12 + 2.0f * u * e);

    float s_r = s_abs_r12 * s_abs_r12 * s_exp1 + s_abs_r23 * s_abs_r23 * s_exp2 +
                2.0f * s_abs_r12 * s_abs_r23 * cosf(s_arg_r23 - s_arg_r12 + 2.0f * u * e);
    s_r /= s_denom;

    float s_t = cuCrealf(s_g) * s_abs_t12 * s_abs_t12 * s_abs_t23 * s_abs_t23;
    s_t /= s_denom;

    // p polarization
    cuFloatComplex p_n2c1 = cuCmulf(n2, cos1);
    cuFloatComplex p_n3c2 = cuCmulf(n3, cos2);
    cuFloatComplex p_n2c3 = cuCmulf(n2, cos3);
    cuFloatComplex p_n1c2 = cuCmulf(n1, cos2);
    cuFloatComplex p_r12 = cuCdivf(cuCsubf(p_n2c1, p_n1c2), cuCaddf(p_n2c1, p_n1c2));
    cuFloatComplex p_r23 = cuCdivf(cuCsubf(p_n3c2, p_n2c3), cuCaddf(p_n3c2, p_n2c3));
    cuFloatComplex p_t12 = cuCdivf(cuCmulf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), n1), cos1), cuCaddf(p_n2c1, p_n1c2));
    cuFloatComplex p_t23 = cuCdivf(cuCmulf(cuCmulf(make_cuFloatComplex(2.0f,0.0f), n2), cos2), cuCaddf(p_n3c2, p_n2c3));
    cuFloatComplex p_g = cuCdivf(cuCmulf(n3, cos3), cuCmulf(n1, cos1));

    float p_abs_r12 = cuCabsf(p_r12);
    float p_abs_r23 = cuCabsf(p_r23);
    float p_abs_t12 = cuCabsf(p_t12);
    float p_abs_t23 = cuCabsf(p_t23);
    float p_arg_r12 = cuCargf(p_r12);
    float p_arg_r23 = cuCargf(p_r23);
    float p_exp1 = exp(2.0f * v * e);

    float p_exp2 = 1.0f / p_exp1;
    float p_denom = p_exp1 +
                    p_abs_r12 * p_abs_r12 * p_abs_r23 * p_abs_r23 * p_exp2 +
                    2.0f * p_abs_r12 * p_abs_r23 * cosf(p_arg_r23 + p_arg_r12 + 2.0f * u * e);

    float p_r = p_abs_r12 * p_abs_r12 * p_exp1 + p_abs_r23 * p_abs_r23 * p_exp2 +
                2.0f * p_abs_r12 * p_abs_r23 * cosf(p_arg_r23 - p_arg_r12 + 2.0f * u * e);
    p_r /= p_denom;

    float p_t = cuCrealf(p_g) * p_abs_t12 * p_abs_t12 * p_abs_t23 * p_abs_t23;
    p_t /= p_denom;

    // calculate s polarization fraction, identical to propagate_at_boundary
    float incident_angle = get_theta(s.surface_normal,-p.direction);
    float refracted_angle = asinf(sinf(incident_angle)*s.refractive_index1/s.refractive_index2);

    float3 incident_plane_normal = cross(p.direction, s.surface_normal);
    float incident_plane_normal_length = norm(incident_plane_normal);

    if (incident_plane_normal_length < 1e-6f)
        incident_plane_normal = p.polarization;
    else
        incident_plane_normal /= incident_plane_normal_length;

    float normal_coefficient = dot(p.polarization, incident_plane_normal);
    float normal_probability = normal_coefficient * normal_coefficient; // i.e. s polarization fraction

    float transmit = normal_probability * s_t + (1.0f - normal_probability) * p_t;
    if (!surface->transmissive)
        transmit = 0.0f;

    float reflect = normal_probability * s_r + (1.0f - normal_probability) * p_r;
    float absorb = 1.0f - transmit - reflect;

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
        // Prevent absorption and reweight accordingly
        float survive = 1.0f - absorb;
        absorb = 0.0f;
        p.weight *= survive;

        detect /= survive;
        reflect /= survive;
        transmit /= survive;
    }

    if (use_weights && detect > 0.0f) {
        p.history |= SURFACE_DETECT;
        chroma_photon_mark_process(
            &rng, CHROMA_TAPE_PROCESS_SURFACE_DETECT);
        p.weight *= detect;
        return BREAK;
    }

    float uniform_sample = chroma_photon_uniform(&rng);

    if (uniform_sample < absorb) {
        // detection probability is conditional on absorption here
        float uniform_sample_detect = chroma_photon_uniform(&rng);
        if (uniform_sample_detect < detect)
            p.history |= SURFACE_DETECT;
        else
            p.history |= SURFACE_ABSORB;
        chroma_photon_mark_process(
            &rng, uniform_sample_detect < detect
                ? CHROMA_TAPE_PROCESS_SURFACE_DETECT
                : CHROMA_TAPE_PROCESS_SURFACE_ABSORB);

        return BREAK;
    }
    else if (uniform_sample < absorb + reflect || !surface->transmissive) {
        // reflect, specularly (default) or diffusely
        float uniform_sample_reflect = chroma_photon_uniform(&rng);
        if (uniform_sample_reflect < reflect_diffuse)
            return propagate_at_diffuse_reflector(p, s, rng);
        else
            return CHROMA_PROPAGATE_SPECULAR(p, s, rng);
    }
    else {
        // refract and transmit
        p.direction = rotate(s.surface_normal, PI-refracted_angle, incident_plane_normal);
        p.polarization = cross(incident_plane_normal, p.direction);
        p.polarization /= norm(p.polarization);
        p.history |= SURFACE_TRANSMIT;
        chroma_photon_mark_process(
            &rng, CHROMA_TAPE_PROCESS_DIELECTRIC_TRANSMIT);
        return CONTINUE;
    }
} // propagate_complex

__device__ int
propagate_at_wls(Photon &p, State &s, ChromaPhotonRng &rng, Surface *surface, bool use_weights=false)
{
    float absorb = interp_property(surface, p.wavelength, surface->absorb);
    float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);
    float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
    float reemit = interp_property(surface, p.wavelength, surface->reemit);

    float uniform_sample = chroma_photon_uniform(&rng);

    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
        // Prevent absorption and reweight accordingly
        float survive = 1.0f - absorb;
        absorb = 0.0f;
        p.weight *= survive;
        reflect_diffuse /= survive;
        reflect_specular /= survive;
    }

    if (uniform_sample < absorb) {
        float uniform_sample_reemit = chroma_photon_uniform(&rng);
        if (uniform_sample_reemit < reemit) {
            p.history |= SURFACE_REEMIT;
            chroma_photon_mark_process(
                &rng, CHROMA_TAPE_PROCESS_SURFACE_REEMIT);
            p.wavelength = sample_cdf(&rng, surface->wavelength_n, surface->wavelength_start, surface->wavelength_step, surface->reemission_cdf);
            p.direction = uniform_sphere(&rng);
            p.polarization = cross(uniform_sphere(&rng), p.direction);
            p.polarization /= norm(p.polarization);
            return CONTINUE;
        } else {
          p.history |= SURFACE_ABSORB;
          chroma_photon_mark_process(
              &rng, CHROMA_TAPE_PROCESS_SURFACE_ABSORB);
          return BREAK;
        }
    }
    else if (uniform_sample < absorb + reflect_specular + reflect_diffuse) {
        // choose how to reflect, defaulting to diffuse
        float uniform_sample_reflect = chroma_photon_uniform(&rng) * (reflect_specular + reflect_diffuse);
        if (uniform_sample_reflect < reflect_specular)
            return CHROMA_PROPAGATE_SPECULAR(p, s, rng);
        else
            return propagate_at_diffuse_reflector(p, s, rng);
    }
    else {
        p.history |= SURFACE_TRANSMIT;
        chroma_photon_mark_process(
            &rng, CHROMA_TAPE_PROCESS_DIELECTRIC_TRANSMIT);
        return PASS;
    }
} // propagate_at_wls


__noinline__ __device__ int
propagate_at_dichroic(Photon &p, State &s, ChromaPhotonRng &rng, Surface *surface, bool use_weights=false)
{
    float incident_angle = get_theta(s.surface_normal, -p.direction);
    
    const DichroicProps *props = surface->dichroic_props;
    float idx = interp_idx(incident_angle,props->nangles,props->angles);
    unsigned int iidx = (int)idx;
    
    float reflect_prob_low = interp_property(surface, p.wavelength, props->dichroic_reflect[iidx]);
    float reflect_prob_high = interp_property(surface, p.wavelength, props->dichroic_reflect[iidx+1]);
    float transmit_prob_low = interp_property(surface, p.wavelength, props->dichroic_transmit[iidx]);
    float transmit_prob_high = interp_property(surface, p.wavelength, props->dichroic_transmit[iidx+1]);
    
    float reflect_prob = reflect_prob_low + (reflect_prob_high-reflect_prob_low)*(idx-iidx);
    float transmit_prob = transmit_prob_low + (transmit_prob_high-transmit_prob_low)*(idx-iidx);
    
    float uniform_sample = chroma_photon_uniform(&rng);
    if ((uniform_sample < reflect_prob)) {
        return CHROMA_PROPAGATE_SPECULAR(p, s, rng);
    }
    else if (uniform_sample < transmit_prob+reflect_prob) {
        p.history |= SURFACE_TRANSMIT;
        chroma_photon_mark_process(
            &rng, CHROMA_TAPE_PROCESS_DIELECTRIC_TRANSMIT);
        return PASS;
    }
    else {
        p.history |= SURFACE_ABSORB;
        chroma_photon_mark_process(
            &rng, CHROMA_TAPE_PROCESS_SURFACE_ABSORB);
        return BREAK;
    }

} // propagate_at_dichroic

__device__ int
propagate_at_angular(Photon &p, State &s, ChromaPhotonRng &rng, Surface *surface, bool use_weights=false)
{
    float incident_angle = get_theta(s.surface_normal, -p.direction);
    
    const AngularProps *props = surface->angular_props;
    float idx = interp_idx(incident_angle, props->nangles, props->angles);
    unsigned int iidx = (int)idx;
    
    float t = idx - iidx;
    float transmit_prob = props->transmit[iidx] + t * (props->transmit[iidx+1] - props->transmit[iidx]);
    float reflect_spec_prob = props->reflect_specular[iidx] + t * (props->reflect_specular[iidx+1] - props->reflect_specular[iidx]);
    float reflect_diff_prob = props->reflect_diffuse[iidx] + t * (props->reflect_diffuse[iidx+1] - props->reflect_diffuse[iidx]);
    
    float absorb_prob = 1.0f - transmit_prob - reflect_spec_prob - reflect_diff_prob;
    
    if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD && absorb_prob < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
        float survive = 1.0f - absorb_prob;
        absorb_prob = 0.0f;
        p.weight *= survive;
        
        transmit_prob /= survive;
        reflect_spec_prob /= survive;
        reflect_diff_prob /= survive;
    }
    
    float uniform_sample = chroma_photon_uniform(&rng);
    
    if (uniform_sample < absorb_prob) {
        p.history |= SURFACE_ABSORB;
        chroma_photon_mark_process(
            &rng, CHROMA_TAPE_PROCESS_SURFACE_ABSORB);
        return BREAK;
    }
    else if (uniform_sample < absorb_prob + transmit_prob) {
        p.history |= SURFACE_TRANSMIT;
        chroma_photon_mark_process(
            &rng, CHROMA_TAPE_PROCESS_DIELECTRIC_TRANSMIT);
        return PASS;
    }
    else if (uniform_sample < absorb_prob + transmit_prob + reflect_spec_prob) {
        return CHROMA_PROPAGATE_SPECULAR(p, s, rng);
    }
    else {
        return propagate_at_diffuse_reflector(p, s, rng);
    }
} // propagate_at_angular

__device__ int
propagate_at_surface(Photon &p, State &s, ChromaPhotonRng &rng, const Geometry *geometry,
                     bool use_weights=false)
{
    Surface *surface = geometry->surfaces[s.surface_index];

    if (surface->model == SURFACE_COMPLEX)
        return propagate_complex(p, s, rng, surface, use_weights);
    else if (surface->model == SURFACE_WLS)
        return propagate_at_wls(p, s, rng, surface, use_weights);
    else if (surface->model == SURFACE_DICHROIC)
        return propagate_at_dichroic(p, s, rng, surface, use_weights);
    else if (surface->model == SURFACE_ANGULAR)
        return propagate_at_angular(p, s, rng, surface, use_weights);
    else {
        // use default surface model: do a combination of specular and
        // diffuse reflection, detection, and absorption based on relative
        // probabilties

        // since the surface properties are interpolated linearly, we are
        // guaranteed that they still sum to 1.0.
        float detect = interp_property(surface, p.wavelength, surface->detect);
        float absorb = interp_property(surface, p.wavelength, surface->absorb);
        float reflect_diffuse = interp_property(surface, p.wavelength, surface->reflect_diffuse);
        float reflect_specular = interp_property(surface, p.wavelength, surface->reflect_specular);

        // numerically enforce probabilities to sum to 1.0 to avoid unintended PASS
#if CHROMA_FORCE_SCATTER_AT_PASS
        float sum_probs = absorb + detect + reflect_diffuse + reflect_specular;
        if (sum_probs > 0.0f) {
            float inv_sum = 1.0f / sum_probs;
            absorb *= inv_sum;
            detect *= inv_sum;
            reflect_diffuse *= inv_sum;
            reflect_specular *= inv_sum;
        }
        // allocate any tiny residual due to rounding to specular so total is exactly 1
        float sum2 = absorb + detect + reflect_diffuse + reflect_specular;
        if (sum2 != 1.0f) {
            reflect_specular += (1.0f - sum2);
        }
#endif
        float uniform_sample = chroma_photon_uniform(&rng);

        if (use_weights && p.weight > WEIGHT_LOWER_THRESHOLD 
            && absorb < (1.0f - WEIGHT_LOWER_THRESHOLD)) {
            // Prevent absorption and reweight accordingly
            float survive = 1.0f - absorb;
            absorb = 0.0f;
            p.weight *= survive;

            // Renormalize remaining probabilities
            detect /= survive;
            reflect_diffuse /= survive;
            reflect_specular /= survive;
        }

        if (use_weights && detect > 0.0f) {
            p.history |= SURFACE_DETECT;
            chroma_photon_mark_process(
                &rng, CHROMA_TAPE_PROCESS_SURFACE_DETECT);
            p.weight *= detect;
            return BREAK;
        }

        if (uniform_sample < absorb) {
            p.history |= SURFACE_ABSORB;
            chroma_photon_mark_process(
                &rng, CHROMA_TAPE_PROCESS_SURFACE_ABSORB);
            return BREAK;
        }
        else if (uniform_sample < absorb + detect) {
            p.history |= SURFACE_DETECT;
            chroma_photon_mark_process(
                &rng, CHROMA_TAPE_PROCESS_SURFACE_DETECT);
            return BREAK;
        }
        else if (uniform_sample < absorb + detect + reflect_diffuse)
            return propagate_at_diffuse_reflector(p, s, rng);
        else if (uniform_sample < absorb + detect + reflect_diffuse + reflect_specular)
            return CHROMA_PROPAGATE_SPECULAR(p, s, rng);
        else
#if CHROMA_FORCE_SCATTER_AT_PASS
            /* include any residual in specular to avoid PASS */
            return CHROMA_PROPAGATE_SPECULAR(p, s, rng);
#else
            return PASS;
#endif
    }

} // propagate_at_surface

#endif
