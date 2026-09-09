//-*-c-*-

#ifndef __INTERSECT_H__
#define __INTERSECT_H__

#include "linalg.h"
#include "matrix.h"
#include "geometry.h"
#include "profile.h"
#include <float.h>


// INFINITY is already defined elsewhere
#define CHROMA_INFINITY __int_as_float(0x7f800000)
#define CHROMA_EPSILON 1e-6

/* Tests the intersection between a ray and a triangle.
   If the ray intersects the triangle, set `distance` to the distance from
   `origin` to the intersection and return true, else return false.
   `direction` must be normalized to one. 
   
   old impl commented out, new stolen from https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
   likely same algorithm, need to check.
   
   */
__device__ bool
intersect_triangle(const float3 &origin, const float3 &direction,
		   const Triangle &triangle, float &distance)
{
        CHROMA_PROF_FUNC_START(CHROMA_PROF_INTERSECT_TRIANGLE);
	/*
	float3 m1 = triangle.v1-triangle.v0;
	float3 m2 = triangle.v2-triangle.v0;
	float3 m3 = -direction;

	Matrix m = make_matrix(m1, m2, m3);
	
	float determinant = det(m);

	if (determinant == 0.0f)
		return false;

	float3 b = origin-triangle.v0;

	float u1 = ((m.a11*m.a22 - m.a12*m.a21)*b.x +
		    (m.a02*m.a21 - m.a01*m.a22)*b.y +
		    (m.a01*m.a12 - m.a02*m.a11)*b.z)/determinant;

	if (u1 < 0.0f || u1 > 1.0f)
		return false;

	float u2 = ((m.a12*m.a20 - m.a10*m.a22)*b.x +
		    (m.a00*m.a22 - m.a02*m.a20)*b.y +
		    (m.a02*m.a10 - m.a00*m.a12)*b.z)/determinant;

	if (u2 < 0.0f || u2 > 1.0f)
		return false;

	float u3 = ((m.a10*m.a21 - m.a11*m.a20)*b.x +
		    (m.a01*m.a20 - m.a00*m.a21)*b.y +
		    (m.a00*m.a11 - m.a01*m.a10)*b.z)/determinant;

	if (u3 <= 0.0f || (1.0f-u1-u2) < -EPSILON)
		return false;

	distance = u3;

	return true;
	*/
        float3 edge1, edge2, h, s, q;
        float a,f,u,v;
        edge1 = triangle.v1 - triangle.v0;
        edge2 = triangle.v2 - triangle.v0;
        h = cross(direction,edge2);
        a = dot(edge1,h);
        if (a > -FLT_EPSILON && a < FLT_EPSILON)
            return false;    // This ray is parallel to this triangle.
        f = 1.0/a;
        s = origin - triangle.v0;
        u = f * dot(s,h);
        if (u < -CHROMA_EPSILON || u > 1.0+CHROMA_EPSILON)
            return false;
        q = cross(s,edge1);
        v = f * dot(direction,q);
        if (v < -CHROMA_EPSILON || u + v > 1.0+CHROMA_EPSILON)
            return false;
        // At this stage we can compute t to find out where the intersection point is on the line.
        float t = f * dot(edge2,q);
        if (t > CHROMA_EPSILON && t < CHROMA_INFINITY) // ray intersection
        {
            //outIntersectionPoint = origin + direction * t;
            distance = t;
            CHROMA_PROF_FUNC_END(CHROMA_PROF_INTERSECT_TRIANGLE);
            return true;
        }
        else // This means that there is a line intersection but not a ray intersection.
        {
            CHROMA_PROF_FUNC_END(CHROMA_PROF_INTERSECT_TRIANGLE);
            return false;
        }
}

/* Tests the intersection between a ray and an axis-aligned box defined by
   an upper and lower bound. If the ray intersects the box, set
   `distance_to_box` to the distance from `origin` to the intersection and
   return true, else return false. `direction` must be normalized to one.

    Source: Optimizing ray tracing for CUDA by Hannu Saransaari
    https://wiki.aalto.fi/download/attachments/40023967/gpgpu.pdf
*/

__device__ bool
intersect_box(const float3 &neg_origin_inv_dir, const float3 &inv_dir,
	      const float3 &lower_bound, const float3 &upper_bound,
	      float& distance_to_box)
{
	CHROMA_PROF_FUNC_START(CHROMA_PROF_INTERSECT_BOX);
	float tmin = 0.0f, tmax = CHROMA_INFINITY;
	float t0, t1;

	// X
	if (isfinite(inv_dir.x)) {
	  t0 = lower_bound.x * inv_dir.x + neg_origin_inv_dir.x;
	  t1 = upper_bound.x * inv_dir.x + neg_origin_inv_dir.x;
	
	  tmin = max(tmin, min(t0, t1));
	  tmax = min(tmax, max(t0, t1));
	}

	// Y
	if (isfinite(inv_dir.y)) {
	  t0 = lower_bound.y * inv_dir.y + neg_origin_inv_dir.y;
	  t1 = upper_bound.y * inv_dir.y + neg_origin_inv_dir.y;
	
	  tmin = max(tmin, min(t0, t1));
	  tmax = min(tmax, max(t0, t1));
	}

	// Z
	if (isfinite(inv_dir.z)) {
	  t0 = lower_bound.z * inv_dir.z + neg_origin_inv_dir.z;
	  t1 = upper_bound.z * inv_dir.z + neg_origin_inv_dir.z;
	
	  tmin = max(tmin, min(t0, t1));
	  tmax = min(tmax, max(t0, t1));
	}

	if (tmin > tmax) {
		CHROMA_PROF_FUNC_END(CHROMA_PROF_INTERSECT_BOX);
		return false;
	}

	distance_to_box = tmin;

	CHROMA_PROF_FUNC_END(CHROMA_PROF_INTERSECT_BOX);
	return true;
}

#endif
