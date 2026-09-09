#ifndef __GEOMETRY_TYPES_H__
#define __GEOMETRY_TYPES_H__

struct Material
{
    float *refractive_index;
    float *absorption_length; 
    float *scattering_length; 
    float **comp_reemission_prob; 
    float **comp_reemission_wvl_cdf;
    float **comp_reemission_time_cdf;
    float **comp_absorption_length;
    unsigned int num_comp;
    unsigned int wavelength_n;
    float wavelength_step;
    float wavelength_start;
    unsigned int time_n;
    float time_step;
    float time_start;
};

enum { SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS, SURFACE_DICHROIC, SURFACE_ANGULAR};

struct DichroicProps
{
    float *angles;
    float **dichroic_reflect;
    float **dichroic_transmit;
    unsigned int nangles;
};

struct AngularProps
{
    float *angles;          
    float *transmit;        
    float *reflect_specular;
    float *reflect_diffuse; 
    unsigned int nangles;   
};

// analytic wire-plane descriptor
struct WirePlane
{
    float3 origin;   // a point on the plane
    float3 u;        // unit vector along wire axes
    float3 v;        // unit vector in-plane, perpendicular to wires
    float  pitch;    // center-to-center spacing along v
    float  radius;   // wire radius
    float  umin;     // finite extent along u (min)
    float  umax;     // finite extent along u (max)
    float  vmin;     // finite extent along v (min)
    float  vmax;     // finite extent along v (max)
    float  v0;       // offset of wire centers along v
    int    surface_index;        // surface index (-1 for none)
    int    material_outer_index; // medium outside wire
    int    material_inner_index; // wire bulk medium
    unsigned int color;          // optional display color
};

struct Surface
{
    float *detect;
    float *absorb;
    float *reemit;
    float *reflect_diffuse;
    float *reflect_specular;
    float *eta;
    float *k;
    float *reemission_cdf;
    DichroicProps *dichroic_props;
    AngularProps *angular_props;
    
    unsigned int model;
    unsigned int wavelength_n;
    unsigned int transmissive;
    float wavelength_step;
    float wavelength_start;
    float thickness;
};

struct Triangle
{
    float3 v0, v1, v2;
};

enum { INTERNAL_NODE, LEAF_NODE, PADDING_NODE };
const unsigned int CHILD_BITS = 28;
const unsigned int NCHILD_MASK = (0xFFFFu << CHILD_BITS);

struct Node
{
    float3 lower;
    float3 upper;
    unsigned int child;
    unsigned int nchild;
};

// Analytic, procedural wire-plane primitive: periodic array of parallel cylinders
// struct WirePlane
// {
//     // A point on the plane (typically plane center)
//     float3 origin;
//     // Unit vector along wire axis (u) and in-plane perpendicular to wires (v)
//     float3 u;
//     float3 v;
//     // Period (pitch) along v and cylinder radius
//     float pitch;
//     float radius;
//     // Finite bounds along u and v in local coordinates (relative to origin)
//     float umin;
//     float umax;
//     float vmin;
//     float vmax;
//     // Optional offset of wire centers along v relative to origin
//     float v0;
//     // Indices into global materials/surfaces tables
//     int surface_index;
//     int material_outer_index; // material outside the wire (e.g., LAr)
//     int material_inner_index; // material inside the wire (e.g., metal)
//     // Optional display color
//     unsigned int color;
// };

struct Geometry
{
    float3 *vertices;
    uint3 *triangles;
    unsigned int *material_codes;
    unsigned int *colors;
    uint4 *primary_nodes;
    uint4 *extra_nodes;
    Material **materials;
    Surface **surfaces;
    WirePlane **wireplanes; // array of pointers to WirePlane structs
    float3 world_origin;
    float world_scale;
    int nprimary_nodes;
    int nwireplanes;
};

#endif

