import numpy as np

def gen_rot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Generate a rotation matrix that rotates vector a to vector b.
    
    Parameters
    ----------
    a : array-like
        The vector to be rotated.
    b : array-like
        The target vector.
        
    Returns
    -------
    R : array-like
        The rotation matrix that rotates vector a to vector b.
    """
    a = np.array(a)
    b = np.array(b)
    
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    
    v = np.cross(a, b)
    c = np.dot(a, b)
    
    if np.isclose(c, 1):
        # Vectors are already aligned
        return np.identity(3)
    elif np.isclose(c, -1):
        # Vectors are anti-parallel
        # Find orthogonal vector to a to form a valid rotation axis
        orthogonal_vector = np.array([1, 0, 0])
        if np.allclose(a, orthogonal_vector):
            orthogonal_vector = np.array([0, 1, 0])
        v = np.cross(a, orthogonal_vector)
        v = v / np.linalg.norm(v)
        v_x, v_y, v_z = v
        v_skew = np.array([[    0, -v_z,  v_y],
                           [ v_z,     0, -v_x],
                           [-v_y,  v_x,     0]])
        return -np.identity(3) + 2 * np.dot(v[:, None], v[None, :])
    
    s = np.linalg.norm(v)
    v_x, v_y, v_z = v
    v_skew = np.array([[    0, -v_z,  v_y],
                       [ v_z,     0, -v_x],
                       [-v_y,  v_x,     0]])
    I = np.eye(3)
    R = I + v_skew + np.dot(v_skew, v_skew) * ((1 - c) / (s ** 2))
    return R


def cylinder(begin: np.ndarray, radius: float, length: float, direction: np.ndarray):
    """Create a cylinder with a given radius, length, and direction.
    
    Parameters
    ----------
    begin : array-like
        The beginning position of the cylinder.
    radius : float
        The radius of the cylinder.
    length : float
        The length of the cylinder.
    direction : array-like
        The direction of the cylinder.
        
    Returns
    -------
    cylinder : trimesh.Trimesh
        The cylinder mesh.

    """
    import trimesh

    # create a cylinder
    cylinder = trimesh.creation.cylinder(radius=radius, height=length)
    cylinder.vertices[:, 2] += length / 2
    # move the cylinder to the beginning
    # rotate the cylinder to the direction
    rotmat = gen_rot([0, 0, 1], direction)
    transformation_mx = np.eye(4)
    transformation_mx[:3, :3] = rotmat
    cylinder.apply_transform(transformation_mx)

    cylinder.apply_translation(begin)
    
    return cylinder

def um2mm(x):
    """Micron to millimeter conversion"""
    return x / 1e3

def mm2um(x):
    """Millimeter to micron conversion"""
    return x * 1e3

