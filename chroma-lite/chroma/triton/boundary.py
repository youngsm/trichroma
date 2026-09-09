"""Robust outgoing ray origins for float32 triangle transport.

Project onto the actual triangle plane in float64, then move beyond the
float32 reconstruction/intersection uncertainty along its geometric normal.
This avoids a second interaction with an adjacent, numerically coplanar face.
The offset is numerical, consumes no RNG and adds no propagation time.
"""

import numpy as np


def offset_boundary_points(points, triangles, outgoing):
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    outgoing = np.asarray(outgoing, dtype=np.float64)
    v0 = triangles[:, 0]
    edge1, edge2 = triangles[:, 1] - v0, triangles[:, 2] - v0
    normal = np.cross(edge1, edge2)
    norm2 = np.sum(normal * normal, axis=1)
    if np.any(norm2 == 0):
        raise ValueError("cannot offset a degenerate boundary triangle")
    residual = np.sum((points - v0) * normal, axis=1)
    extent = np.max(np.abs(edge1) + np.abs(edge2) + np.abs(np.abs(edge1) - np.abs(edge2)), axis=1)
    epsilon = np.finfo(np.float32).eps
    error = epsilon * np.abs(v0) + 3 * epsilon * extent[:, None]
    clearance = np.sum(np.abs(normal) * error, axis=1)
    sign = np.where(np.sum(outgoing * normal, axis=1) >= 0, 1.0, -1.0)
    shifted = (points + normal * ((sign * clearance - residual) / norm2)[:, None]).astype(
        np.float32
    )
    outward = sign[:, None] * normal
    target = np.where(outward > 0, np.float32(np.inf), np.float32(-np.inf))
    return np.where(outward != 0, np.nextafter(shifted, target), shifted)
