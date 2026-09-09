'''chroma.cache: On-disk geometry and bounding volume hierarchy cache.

The ``Cache`` class is used to manage an on-disk cache of geometry and
BVH objects, which are slow to calculate.  By default, the cache is in
$HOME/.chroma.

Currently this cache is not thread or process-safe when a writing
process is present.
'''

import os
import pickle as pickle
import copy

from chroma.log import logger

cache_dir = os.environ.get('CHROMA_CACHE_DIR', os.path.expanduser('~/.chroma/'))

class GeometryNotFoundError(Exception):
    '''A requested geometry was not found in the on-disk cache.'''
    def __init__(self, msg):
        Exception.__init__(self, msg)

class BVHNotFoundError(Exception):
    '''A requested bounding volume hierarchy was not found in the
    on-disk cache.'''
    def __init__(self, msg):
        Exception.__init__(self, msg)

def verify_or_create_dir(dirname, exception_msg, logger_msg=None):
    '''Checks if ``dirname`` exists and is a directory.  If it does not exist,
    then it is created.  If it does exist, but is not a directory, an IOError
    is raised with ``exception_message`` as the description.

    If the directory is created, an info message will be sent to the 
    Chroma logger if ``logger_message`` is not None.
    '''
    if not os.path.isdir(dirname):
        if os.path.exists(dirname):
            raise IOError(exception_msg)
        else:
            if logger_msg is not None:
                logger.info(logger_msg)
            os.mkdir(dirname)

class Cache(object):
    '''Class for manipulating a Chroma disk cache directory.

    Use this class to read and write cached geometries or bounding
    volume hierarchies rather than reading and writing disk directly.

    Cached geometries are accessed by a name string.  The name of a
    geometry should be a legal Python identifier that does not start
    with an underscore.  (Note that the name string is directly
    mapped to a filename, so beware of taking Cache names from untrusted
    sources.  Don't let Little Bobby Tables ruin your day.)

    Bounding volume hierarchies are accessed by MD5 hash of the
    flattened geometry mesh using the hexadecimal string provided by
    Mesh.md5().  Multiple BVHs are possible for a given mesh, so the
    BVH can also be given an optional name (legal Python identifier,
    no underscore as with geometries).  The version of the BVH
    with the special name "default" will be the default BVH.
    '''
    
    def __init__(self, cache_dir=cache_dir):
        '''Open a Chroma cache stored at ``cache_dir``.
        
        If ``cache_dir`` does not already exist, it will be created.  By default,
        the cache is in the ``.chroma`` directory under the user's home directory.
        '''
        self.cache_dir = cache_dir
        verify_or_create_dir(self.cache_dir,
            exception_msg='Path for cache already exists, '
                          'but is not a directory: %s' % cache_dir,
            logger_msg='Creating new Chroma cache directory at %s' 
                       % cache_dir)


        self.geo_dir = os.path.join(cache_dir, 'geo')
        verify_or_create_dir(self.geo_dir,
            exception_msg='Path for geometry directory in cache '
                          'already exists, but is not a directory: %s' 
                          % self.geo_dir)

        self.bvh_dir = os.path.join(cache_dir, 'bvh')
        verify_or_create_dir(self.bvh_dir,
            exception_msg='Path for BVH directory in cache already '
                          'exists, but is not a directory: %s' % self.bvh_dir)


    def get_geometry_filename(self, name):
        '''Return the full pathname for the geometry file corresponding to 
        ``name``.
        '''
        return os.path.join(self.geo_dir, name)

    def list_geometry(self):
        '''Returns a list of all geometry names in the cache.'''
        return os.listdir(self.geo_dir)

    def save_geometry(self, name, geometry):
        '''Save ``geometry`` in the cache with the name ``name``.'''
        geo_file = self.get_geometry_filename(name)
        # exclude saving the BVH
        reduced_geometry = copy.copy(geometry)
        reduced_geometry.bvh = None
        reduced_geometry.solids = []
        reduced_geometry.solid_rotations = []
        reduced_geometry.solid_displacements = []

        with open(geo_file, 'wb') as output_file:
            pickle.dump(geometry.mesh.md5(), output_file, 
                        pickle.HIGHEST_PROTOCOL)
            pickle.dump(reduced_geometry, output_file, 
                        pickle.HIGHEST_PROTOCOL)

    def load_geometry(self, name):
        '''Returns the chroma.geometry.Geometry object associated with
        ``name`` in the cache.

           Raises ``GeometryNotFoundError`` if ``name`` is not in the cache.
        '''
        geo_file = self.get_geometry_filename(name)
        if not os.path.exists(geo_file):
            raise GeometryNotFoundError(name)
        with open(geo_file, 'rb') as input_file:
            _ = pickle.load(input_file) # skip mesh hash
            geo = pickle.load(input_file)
            return geo

    def remove_geometry(self, name):
        '''Remove the geometry file associated with ``name`` from the cache.
        
        If ``name`` does not exist, no action is taken.
        '''
        geo_file = self.get_geometry_filename(name)
        if os.path.exists(geo_file):
            os.remove(geo_file)

    def get_geometry_hash(self, name):
        '''Get the mesh hash for the geometry associated with ``name``.

        This is faster than loading the entire geometry file and calling the
        hash() method on the mesh.
        '''
        geo_file = self.get_geometry_filename(name)
        if not os.path.exists(geo_file):
            raise GeometryNotFoundError(name)
        with open(geo_file, 'rb') as input_file:
            return pickle.load(input_file)

    def load_default_geometry(self):
        '''Load the default geometry as set by previous call to
        set_default_geometry().

        If no geometry has been designated the default, raise 
        GeometryNotFoundError.
        '''
        return self.load_geometry('.default')

    def set_default_geometry(self, name):
        '''Set the geometry in the cache corresponding to ``name`` to
        be the default geometry returned by load_default_geometry().
        '''
        default_geo_file = self.get_geometry_filename('.default')
        geo_file = self.get_geometry_filename(name)
        if not os.path.exists(geo_file):
            raise GeometryNotFoundError(name)
        
        if os.path.exists(default_geo_file):
            if os.path.islink(default_geo_file):
                os.remove(default_geo_file)
            else:
                raise IOError('Non-symlink found where expected a symlink: '
                              +default_geo_file)
        os.symlink(geo_file, default_geo_file)


    def get_bvh_directory(self, mesh_hash):
        '''Return the full path to the directory corresponding to 
        ``mesh_hash``.
        '''
        return os.path.join(self.bvh_dir, mesh_hash)

    def get_bvh_filename(self, mesh_hash, name='default'):
        '''Return the full pathname for the BVH file corresponding to 
        ``name``.
        '''
        return os.path.join(self.get_bvh_directory(mesh_hash), name)


    def list_bvh(self, mesh_hash):
        '''Returns a list the names of all BVHs corresponding to 
        ``mesh_hash``.
        '''
        bvh_dir = self.get_bvh_directory(mesh_hash)
        if not os.path.isdir(bvh_dir):
            return []
        else:
            return os.listdir(bvh_dir)

    def exist_bvh(self, mesh_hash, name='default'):
        '''Returns True if a cached BVH exists corresponding to
        ``mesh_hash`` with the given ``name``.
        '''
        return os.path.isfile(self.get_bvh_filename(mesh_hash, name))

    def save_bvh(self, bvh, mesh_hash, name='default'):
        '''Saves the given chroma.bvh.BVH object to the cache, tagged
        by the given ``mesh_hash`` and ``name``.
        '''
        bvh_dir = self.get_bvh_directory(mesh_hash)
        verify_or_create_dir(bvh_dir, 
                             exception_msg='Non-directory already exists '
                             'where BVH directory should go: ' + bvh_dir)
        bvh_file = self.get_bvh_filename(mesh_hash, name)

        with open(bvh_file, 'wb') as output_file:
            pickle.dump(bvh, output_file, 
                        pickle.HIGHEST_PROTOCOL)

    def load_bvh(self, mesh_hash, name='default'):
        '''Returns the chroma.bvh.BVH object corresponding to ``mesh_hash``
        and the given ``name``.  

        If no BVH exists, raises ``BVHNotFoundError``.
        '''
        bvh_file = self.get_bvh_filename(mesh_hash, name)

        if not os.path.exists(bvh_file):
            raise BVHNotFoundError(mesh_hash + ':' + name)

        with open(bvh_file, 'rb') as input_file:
            bvh = pickle.load(input_file)
            return bvh

    def remove_bvh(self, mesh_hash, name='default'):
        '''Remove the BVH file associated with ``mesh_hash`` and
        ``name`` from the cache.
        
        If the BVH does not exist, no action is taken.
        '''
        bvh_file = self.get_bvh_filename(mesh_hash, name)
        if os.path.exists(bvh_file):
            os.remove(bvh_file)
