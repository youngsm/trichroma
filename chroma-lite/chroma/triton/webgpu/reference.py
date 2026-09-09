"""Independent float64 triangle oracle for the exported browser geometry.

This deliberately ignores the exported BVH nodes. Instance bounds select
candidates; every triangle of a selected shared mesh is tested in float64.
It is for small validation ray sets, not interactive rendering.
"""

import numpy as np

from chroma.triton.viewer import Camera


def camera_rays(camera, width, height):
    packed = Camera(**camera).packed()
    y, x = np.indices((height, width), dtype=np.float32)
    horizontal = (2 * (x + 0.5) / width - 1) * (width / height) * packed[12]
    vertical = (1 - 2 * (y + 0.5) / height) * packed[12]
    directions = (
        packed[3:6] + horizontal[..., None] * packed[6:9] + vertical[..., None] * packed[9:12]
    ).reshape(-1, 3)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return np.repeat(packed[None, :3], len(directions), axis=0), directions


def nearest_reference(groups, origins, directions):
    """Return distance/group/instance/triangle, normal.xyz/hit in eight columns."""
    result = np.zeros((len(origins), 8), np.float64)
    result[:, :4] = -1
    for ray, (origin, direction) in enumerate(
        zip(np.asarray(origins, float), np.asarray(directions, float))
    ):
        best = np.inf
        for group_id, group in enumerate(groups):
            lower = np.asarray(group.scene_view.instances.bounds_min, float)
            upper = np.asarray(group.scene_view.instances.bounds_max, float)
            parallel = direction == 0
            divisor = np.where(parallel, 1.0, direction)
            first, second = (lower - origin) / divisor, (upper - origin) / divisor
            near = np.max(np.where(parallel, -np.inf, np.minimum(first, second)), axis=1)
            far = np.min(np.where(parallel, np.inf, np.maximum(first, second)), axis=1)
            candidates = np.flatnonzero(
                (far >= np.maximum(near, 0.0))
                & (near <= best)
                & ~np.any(parallel & ((origin < lower) | (origin > upper)), axis=1)
            )
            triangles = np.asarray(group.bvh.triangle_vertices, float)
            edge1, edge2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            for instance in candidates:
                rotation = group.transforms[instance, :9].reshape(3, 3).astype(float)
                translation = group.transforms[instance, 9:].astype(float)
                local_origin = (origin - translation) @ rotation
                local_direction = direction @ rotation
                h = np.cross(local_direction, edge2)
                determinant = np.sum(edge1 * h, axis=1)
                valid = np.abs(determinant) > np.finfo(np.float32).eps
                reciprocal = 1.0 / np.where(valid, determinant, 1.0)
                offset = local_origin - triangles[:, 0]
                u = reciprocal * np.sum(offset * h, axis=1)
                q = np.cross(offset, edge1)
                v = reciprocal * (q @ local_direction)
                distance = reciprocal * np.sum(edge2 * q, axis=1)
                valid &= (
                    (u >= -1e-6)
                    & (u <= 1.000001)
                    & (v >= -1e-6)
                    & (u + v <= 1.000001)
                    & (distance > 1e-6)
                    & (distance < best)
                )
                if not np.any(valid):
                    continue
                triangle = int(np.argmin(np.where(valid, distance, np.inf)))
                best = distance[triangle]
                normal = np.cross(edge1[triangle], edge2[triangle]) @ rotation.T
                normal /= np.linalg.norm(normal)
                result[ray] = [best, group_id, instance, triangle, *normal, 1.0]
    return result
