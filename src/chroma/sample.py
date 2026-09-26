import numpy as np
from chroma.transform import rotate

def uniform_sphere(size=None, dtype=np.double):
    """
    Generate random points isotropically distributed across the unit sphere.

    Args:
        - size: int, *optional*
            Number of points to generate. If no size is specified, a single
            point is returned.

    Source: Weisstein, Eric W. "Sphere Point Picking." Mathworld. 
    """

    theta, u = np.random.uniform(0.0, 2*np.pi, size), \
        np.random.uniform(-1.0, 1.0, size)

    c = np.sqrt(1-u**2)

    if size is None:
        return np.array([c*np.cos(theta), c*np.sin(theta), u])

    points = np.empty((size, 3), dtype)

    points[:,0] = c*np.cos(theta)
    points[:,1] = c*np.sin(theta)
    points[:,2] = u

    return points

def flashlight(phi=np.pi/4, direction=(0,0,1), size=None, dtype=np.double):
    theta, u = np.random.uniform(0.0, 2*np.pi, size), \
        np.random.uniform(np.cos(phi), 1, size)

    c = np.sqrt(1-u**2)

    if np.equal(direction, (0,0,1)).all():
        rotation_axis = (0,0,1)
        rotation_angle = 0.0
    else:
        rotation_axis = np.cross((0,0,1), direction)
        rotation_angle = \
            -np.arccos(np.dot(direction, (0,0,1))/np.linalg.norm(direction))

    if size is None:
        return rotate(np.array([c*np.cos(theta), c*np.sin(theta), u]),
                      rotation_angle, rotation_axis)

    points = np.empty((size, 3), dtype)

    points[:,0] = c*np.cos(theta)
    points[:,1] = c*np.sin(theta)
    points[:,2] = u

    return rotate(points, rotation_angle, rotation_axis)
