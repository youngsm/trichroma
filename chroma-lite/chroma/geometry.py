import sys
import os
from itertools import chain
from hashlib import md5
import pickle as pickle
import gzip
import numpy as np
import time

from chroma.itertoolset import *
from chroma.tools import timeit, profile_if_possible, filled_array, \
    memoize_method_with_dictionary_arg
from chroma.log import logger

# all material/surface properties are interpolated at these
# wavelengths when they are sent to the gpu
standard_wavelengths = np.arange(60, 1000, 5).astype(np.float32)

class Mesh(object):
    "Triangle mesh object."
    def __init__(self, vertices, triangles, remove_duplicate_vertices=False, round=True, remove_null_triangles=True):
        vertices = np.asarray(vertices, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.int32)

        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError('shape mismatch')

        if len(triangles.shape) != 2 or triangles.shape[1] != 3:
            raise ValueError('shape mismatch')

        if (triangles < 0).any():
            raise ValueError('indices in `triangles` must be positive.')

        if (triangles >= len(vertices)).any():
            raise ValueError('indices in `triangles` must be less than the '
                             'length of the vertex array.')

        self.vertices = vertices
        self.triangles = triangles
        if len(self.vertices) == 0:
            logger.warning("Generated mesh has no vertices.")
        if len(self.triangles) == 0:
            logger.warning("Generated mesh has no triangles.")
        if round:
            self.vertices = self.vertices.round(decimals=12)
        if remove_duplicate_vertices:
            self.remove_duplicate_vertices()
        if remove_null_triangles:
            self.remove_null_triangles()

    def get_triangle_centers(self):
        "Returns the x,y,z coordinate of the center of each triangle."
        return np.mean(self.assemble(),axis=1)

    def get_bounds(self):
        "Return the lower and upper bounds for the mesh as a tuple."
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def remove_duplicate_vertices(self):
        "Remove any duplicate vertices in the mesh."
        # view the vertices as a structured array in order to identify unique
        # rows, i.e. unique vertices
        unique_vertices, inverse = np.unique(self.vertices.view([('', self.vertices.dtype)]*self.vertices.shape[1]), return_inverse=True)
        # turn the structured vertex array back into a normal array
        self.vertices = unique_vertices.view(self.vertices.dtype).reshape((unique_vertices.shape[0], self.vertices.shape[1]))
        # remap the triangles to the unique vertices
        self.triangles = inverse[self.triangles]
        if self.triangles.ndim == 3:
            self.triangles = self.triangles.squeeze(-1)

    def remove_null_triangles(self):
        '''Remove any zero-area triangles from a mesh.
 
        Returns the mask of retained triangles, which may be applied to the
        material, surface, etc., arrays of an associated ``Solid``.
        '''
        if len(self.triangles) == 0:
            return
        mask = np.array([(len(set(x)) == 3) for x in self.triangles])
        self.triangles = self.triangles[mask]
        return mask

    def assemble(self, key=slice(None), group=True):
        """
        Return an assembled triangle mesh; i.e. return the vertex positions
        of every triangle. If `group` is True, the array returned will have
        an extra axis denoting the triangle number; i.e. if the mesh contains
        N triangles, the returned array will have the shape (N,3,3). If `group`
        is False, return just the vertex positions without any grouping; in
        this case the grouping is implied (i.e. elements [0:3] are the first
        triangle, [3:6] the second, and so on.

        The `key` argument is a slice object if you just want to assemble
        a certain range of the triangles, i.e. to get the assembled mesh for
        triangles 3 through 6, key = slice(3,7).
        """
        if group:
            vertex_indices = self.triangles[key]
        else:
            vertex_indices = self.triangles[key].flatten()

        return self.vertices[vertex_indices]

    def __add__(self, other):
        return Mesh(np.concatenate((self.vertices, other.vertices)), np.concatenate((self.triangles, other.triangles + len(self.vertices))))

    def md5(self):
        '''Return the MD5 hash of the vertices and triangles in this mesh as a 
        hexidecimal string.'''
        checksum = md5(self.vertices)
        checksum.update(self.triangles)
        return checksum.hexdigest()

    def __repr__(self):
        bounds_min, bounds_max = self.get_bounds()
        extent = bounds_max - bounds_min
        return (f"<Mesh: {len(self.triangles)} triangles, {len(self.vertices)} vertices, "
                f"extent=({extent[0]:.2f}, {extent[1]:.2f}, {extent[2]:.2f})>")

def silly_unique(arr):
    return np.asarray(list(set(arr)))

class Solid(object):
    """Solid object attaches materials, surfaces, and colors to each triangle
    in a Mesh object."""
    def __init__(self, mesh, material1=None, material2=None, surface=None, color=0x33ffffff, name=None, displacement=None, rotation=None):
        self.mesh = mesh
        self.name = name
        self.displacement = np.asarray(displacement, dtype=np.float32) if displacement is not None else None
        self.rotation = np.asarray(rotation, dtype=np.float32) if rotation is not None else None
        if np.iterable(material1):
            if len(material1) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.material1 = np.array(material1, dtype=object)
        else:
            self.material1 = np.tile(material1, len(self.mesh.triangles))

        if np.iterable(material2):
            if len(material2) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.material2 = np.array(material2, dtype=object)
        else:
            self.material2 = np.tile(material2, len(self.mesh.triangles))

        if np.iterable(surface):
            if len(surface) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.surface = np.array(surface, dtype=object)
        else:
            self.surface = np.tile(surface, len(self.mesh.triangles))

        if np.iterable(color):
            if len(color) != len(mesh.triangles):
                raise ValueError('shape mismatch')
            self.color = np.array(color, dtype=np.uint32)
        else:
            self.color = np.tile(color, len(self.mesh.triangles)).astype(np.uint32)

        self.unique_materials = \
            silly_unique(np.concatenate([self.material1, self.material2]))

        self.unique_surfaces = silly_unique(self.surface)

    def __add__(self, other):
        return Solid(self.mesh + other.mesh, np.concatenate((self.material1, other.material1)), np.concatenate((self.material2, other.material2)), np.concatenate((self.surface, other.surface)), np.concatenate((self.color, other.color)), name=None)

    def weld(self, other, shared_triangle_surface=None, shared_triangle_color=None):
        '''Merge this solid with another at any identical triangles.

        Triangles that are common to both solids will be reduced to one, with the
        surface and color properties of this one unless otherwise specified with
        ``shared_triangle_surface`` and/or ``shared_triangle_color``.

        Note that this is NOT a boolean union!
        '''
        # create arrays of sets of (three-tuple) points
        # use sets to take care of point order permutations, numpy element-wise-ness
        points_self = np.array([set(map(tuple, [self.mesh.vertices[x] for x in y])) for y in self.mesh.triangles], dtype=set)
        points_other = np.array([set(map(tuple, [other.mesh.vertices[x] for x in y])) for y in other.mesh.triangles], dtype=set)

        # find any triangles in other that also exist in self
        match = [np.where(points_other == x)[0] for x in points_self]
        mask = np.array([len(x)>0 for x in match])
        if mask.sum() == 0:
            raise Exception('cannot weld solids with no shared triangles')

        # get triangle ids of the duplicates
        duplicates = sorted(list(chain(*[np.where(points_other == x)[0] for x in points_self])))

        # create temporary second solid -- solid2 with duplicate triangles removed
        mesh = Mesh(other.mesh.vertices, np.delete(other.mesh.triangles, duplicates, 0))
        material1 = np.delete(other.material1, duplicates, 0)
        material2 = np.delete(other.material2, duplicates, 0)
        surface = np.delete(other.surface, duplicates, 0)
        color = np.delete(other.color, duplicates, 0)

        self.mesh = self.mesh + mesh
        self.material1 = np.concatenate((self.material1, material1))
        self.material2 = np.concatenate((self.material2, material2))
        self.surface = np.concatenate((self.surface, surface))
        self.color = np.concatenate((self.color, color))

        # set properties at interface
        self.material2[mask] = other.material1[0]
        if shared_triangle_surface is not None:
            self.surface[mask] = shared_triangle_surface
        if shared_triangle_color is not None:
            self.color[mask] = shared_triangle_color

    @memoize_method_with_dictionary_arg
    def material1_indices(self, material_lookup):
        return np.fromiter(map(material_lookup.get, self.material1), dtype=np.int32, count=len(self.material1))

    @memoize_method_with_dictionary_arg
    def material2_indices(self, material_lookup):
        return np.fromiter(map(material_lookup.get, self.material2), dtype=np.int32, count=len(self.material2))

    @memoize_method_with_dictionary_arg
    def surface_indices(self, surface_lookup):
        return np.fromiter(map(surface_lookup.get, self.surface), dtype=np.int32, count=len(self.surface))

    def __repr__(self):
        mat_names = set(m.name if m else 'None' for m in self.unique_materials)
        surf_names = set(s.name if s else 'None' for s in self.unique_surfaces)
        name_str = f"'{self.name}', " if self.name else ""
        parts = [f"<Solid {name_str}{len(self.mesh.triangles)} triangles, materials={mat_names}, surfaces={surf_names}"]
        if self.displacement is not None:
            d = self.displacement
            parts.append(f" pos=({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f})")
        if self.rotation is not None:
            r = self.rotation
            if np.allclose(r, np.eye(3)):
                parts.append(" rot=identity")
            else:
                parts.append(f" rot=[[{r[0,0]:.2f},...]]")
        parts.append(">")
        return "".join(parts)

class Material(object):
    """Material optical properties."""
    def __init__(self, name='none'):
        self.name = name

        self.refractive_index = None
        self.absorption_length = None
        self.scattering_length = None
        self.scintillation_spectrum = None
        self.scintillation_light_yield = None
        self.scintillation_rise_time = None
        self.scintillation_waveform = None
        self.scintillation_mod = None
        self.comp_reemission_prob = []
        self.comp_reemission_wvl_cdf = []
        self.comp_reemission_times = []
        self.comp_reemission_time_cdf = []
        self.comp_absorption_length = []
        self.density = 0.0 # g/cm^3
        self.composition = {} # by mass

    def __repr__(self):
        props = []
        if self.refractive_index is not None:
            n_vals = self.refractive_index[:, 1]
            props.append(f"n={n_vals.min():.3f}-{n_vals.max():.3f}")
        if self.absorption_length is not None:
            abs_vals = self.absorption_length[:, 1]
            props.append(f"abs_len={abs_vals.min():.1f}-{abs_vals.max():.1f}")
        if self.scattering_length is not None:
            scat_vals = self.scattering_length[:, 1]
            props.append(f"scat_len={scat_vals.min():.1f}-{scat_vals.max():.1f}")
        props_str = ', '.join(props) if props else 'unset'
        return f"<Material '{self.name}': {props_str}>"

    def set(self, name, value, wavelengths=standard_wavelengths):
        if np.iterable(value):
            if len(value) != len(wavelengths):
                raise ValueError('shape mismatch')
        else:
            value = np.tile(value, len(wavelengths))

        self.__dict__[name] = np.array(list(zip(wavelengths, value)), dtype=np.float32)

# Empty material
vacuum = Material('vacuum')
vacuum.set('refractive_index', 1.0)
vacuum.set('absorption_length', 1e6)
vacuum.set('scattering_length', 1e6)

class DichroicProps(object):
    def __init__(self,angles,reflect,transmit):
        self.angles = np.asarray(angles) #[angle]
        self.dichroic_reflect = np.asarray(reflect) #[angle][point]
        self.dichroic_transmit = np.asarray(transmit) #[angle][point]

    def __repr__(self):
        return (f"<DichroicProps: {len(self.angles)} angles "
                f"({np.degrees(self.angles.min()):.1f}-{np.degrees(self.angles.max()):.1f} deg)>")

class AngularProps(object):
    def __init__(self, angles, transmit, reflect_specular=None, reflect_diffuse=None):
        self.angles = np.asarray(angles)  # [angle in radians]
        self.transmit = np.asarray(transmit)  # [transmission probability]
        self.reflect_specular = np.asarray(reflect_specular) if reflect_specular is not None else np.zeros_like(transmit)
        self.reflect_diffuse = np.asarray(reflect_diffuse) if reflect_diffuse is not None else np.zeros_like(transmit)

    def __repr__(self):
        return (f"<AngularProps: {len(self.angles)} angles "
                f"({np.degrees(self.angles.min()):.1f}-{np.degrees(self.angles.max()):.1f} deg), "
                f"T={self.transmit.min():.2f}-{self.transmit.max():.2f}>")

class Surface(object):
    """Surface optical properties."""
    def __init__(self, name='none', model=0):
        self.name = name
        self.model = model

        self.set('detect', 0)
        self.set('absorb', 0)
        self.set('reemit', 0)
        self.set('reflect_diffuse', 0)
        self.set('reflect_specular', 0)
        self.set('eta', 0)
        self.set('k', 0)
        self.set('reemission_cdf', 0)
        
        self.dichroic_props = None
        self.angular_props = None

        self.thickness = 0.0
        self.transmissive = 0

    def set(self, name, value, wavelengths=standard_wavelengths):
        if np.iterable(value):
            if len(value) != len(wavelengths):
                raise ValueError('shape mismatch')
        else:
            value = np.tile(value, len(wavelengths))

        if (np.asarray(value) < 0.0).any():
            raise Exception('all probabilities must be >= 0.0')

        self.__dict__[name] = np.array(list(zip(wavelengths, value)), dtype=np.float32)

    def __repr__(self):
        props = []
        for attr in ['detect', 'absorb', 'reflect_diffuse', 'reflect_specular', 'reemit']:
            arr = getattr(self, attr, None)
            if arr is not None and arr[:, 1].max() > 0:
                val = arr[:, 1].mean()
                props.append(f"{attr}={val:.2f}")
        if self.dichroic_props:
            props.append("dichroic")
        if self.angular_props:
            props.append("angular")
        props_str = ', '.join(props) if props else 'passive'
        return f"<Surface '{self.name}': {props_str}>"
        
class Geometry(object):
    "Geometry object."
    def __init__(self, detector_material=None):
        self.detector_material = detector_material
        self.solids = []
        self.solid_rotations = []
        self.solid_displacements = []
        self.solid_names = []
        self.bvh = None

    def add_solid(self, solid, rotation=None, displacement=None, name=None):
        """
        Add the solid `solid` to the geometry. When building the final triangle
        mesh, `solid` will be placed by rotating it with the rotation matrix
        `rotation` and displacing it by the vector `displacement`.
        `name`: optional label for this solid (e.g. "pmt0"); used for repr/debugging.
        """
        name_to_store = name if name is not None else getattr(solid, 'name', None)
        self.solid_names.append(name_to_store)

        if rotation is None:
            rotation = np.identity(3)
        else:
            rotation = np.asarray(rotation, dtype=np.float32)

        if rotation.shape != (3,3):
            raise ValueError('rotation matrix has the wrong shape.')

        self.solid_rotations.append(rotation.astype(np.float32))

        if displacement is None:
            displacement = np.zeros(3)
        else:
            displacement = np.asarray(displacement, dtype=np.float32)

        if displacement.shape != (3,):
            raise ValueError('displacement vector has the wrong shape.')

        self.solid_displacements.append(displacement)

        self.solids.append(solid)

        return len(self.solids)-1

    def solid_repr(self, i):
        """repr for solid index i including its placement (displacement + rotation) in this geometry."""
        solid = self.solids[i]
        d = self.solid_displacements[i]
        r = self.solid_rotations[i]
        base = solid.__repr__().rstrip(">")
        pos_str = f" pos=({d[0]:.1f}, {d[1]:.1f}, {d[2]:.1f})"
        if np.allclose(r, np.eye(3)):
            rot_str = " rot=identity"
        else:
            rot_str = f" rot=[[{r[0,0]:.2f},{r[0,1]:.2f},{r[0,2]:.2f}; ...]"
        return base + pos_str + rot_str + ">"

    def flatten(self):
        """
        Create the flat list of triangles (and triangle properties)
        from the list of solids in this geometry.

        This does not build the BVH!  If you want to use the geometry
        for rendering or simulation, you should call build() instead.
        """

        # Don't run this function twice!
        if hasattr(self, 'mesh'):
            return

        nv = np.cumsum([0] + [len(solid.mesh.vertices) for solid in self.solids])
        nt = np.cumsum([0] + [len(solid.mesh.triangles) for solid in self.solids])

        vertices = np.empty((nv[-1],3), dtype=np.float32)
        triangles = np.empty((nt[-1],3), dtype=np.uint32)
        

        logger.info('Flattening detector mesh...')
        logger.info('  triangles: %d' % len(triangles))
        logger.info('  vertices:  %d' % len(vertices))


        for i, solid in enumerate(self.solids):
            vertices[nv[i]:nv[i+1]] = \
                np.inner(solid.mesh.vertices, self.solid_rotations[i]) + self.solid_displacements[i]
            triangles[nt[i]:nt[i+1]] = solid.mesh.triangles + nv[i]

        # Different solids are very unlikely to share vertices, so this goes much faster
        self.mesh = Mesh(vertices, triangles, remove_duplicate_vertices=True, remove_null_triangles=False)

        self.colors = np.concatenate([solid.color for solid in self.solids])

        self.solid_id = np.concatenate([filled_array(i, shape=len(solid.mesh.triangles), dtype=np.uint32) for i, solid in enumerate(self.solids)])

        self.unique_materials = list(silly_unique(np.concatenate([solid.unique_materials for solid in self.solids])))

        material_lookup = dict(list(zip(self.unique_materials, list(range(len(self.unique_materials))))))

        self.material1_index = np.concatenate([solid.material1_indices(material_lookup) for solid in self.solids])

        self.material2_index = np.concatenate([solid.material2_indices(material_lookup) for solid in self.solids])

        self.unique_surfaces = list(silly_unique(np.concatenate([solid.unique_surfaces for solid in self.solids])))

        surface_lookup = dict(list(zip(self.unique_surfaces, list(range(len(self.unique_surfaces))))))

        self.surface_index = np.concatenate([solid.surface_indices(surface_lookup) for solid in self.solids])

        try:
            self.surface_index[self.surface_index == surface_lookup[None]] = -1
        except KeyError:
            pass

    def __repr__(self):
        n_solids = len(self.solids)
        if hasattr(self, 'mesh'):
            n_tri = len(self.mesh.triangles)
            n_mat = len(self.unique_materials)
            n_surf = len(self.unique_surfaces)
            mat_str = f", {n_mat} materials, {n_surf} surfaces"
        else:
            n_tri = sum(len(s.mesh.triangles) for s in self.solids)
            mat_str = " (not flattened)"
        det_mat = self.detector_material.name if self.detector_material else 'None'
        bvh_str = ", bvh=built" if self.bvh else ""
        names = getattr(self, 'solid_names', None)
        name_str = f", names={names}" if names and any(n is not None for n in names) else ""
        return f"<Geometry: {n_solids} solids, {n_tri} triangles{mat_str}, detector_material='{det_mat}'{bvh_str}{name_str}>"

