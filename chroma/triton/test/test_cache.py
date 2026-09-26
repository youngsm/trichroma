from .unittest_find import unittest
import os
import shutil
import tempfile
import binascii

from chroma.cache import verify_or_create_dir, Cache, GeometryNotFoundError,\
    BVHNotFoundError
from chroma.geometry import Geometry, Solid
from chroma.make import box

def random_tempdir(prefix):
    '''Select a random directory name inside the $TMP directory that
    starts with ``prefix`` and ends with a random hex string.'''
    subdir = prefix + '_' + binascii.b2a_hex(os.urandom(8))
    return os.path.join(tempfile.gettempdir(), subdir)

def remove_path(path):
    '''If path is a file, delete it.  If it is a directory, remove it.
    If it doesn't exist, do nothing.'''
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    

class TestVerifyOrCreateDir(unittest.TestCase):
    def setUp(self):
        self.test_dir = random_tempdir('vcd')

    def test_no_dir(self):
        assert not os.path.isdir(self.test_dir)
        verify_or_create_dir(self.test_dir, 'msg')
        assert os.path.isdir(self.test_dir)

    def test_exist_dir(self):
        os.mkdir(self.test_dir)
        assert os.path.isdir(self.test_dir)
        verify_or_create_dir(self.test_dir, 'msg')
        assert os.path.isdir(self.test_dir)

    def test_exist_file(self):
        f = open(self.test_dir, 'w')
        f.write('foo')
        f.close()
        with self.assertRaises(IOError):
            verify_or_create_dir(self.test_dir, 'msg')

    def tearDown(self):
        remove_path(self.test_dir)

class TestCacheCreation(unittest.TestCase):
    def setUp(self):
        self.cache_dir = random_tempdir('chroma_cache_test')

    def test_creation(self):
        assert not os.path.isdir(self.cache_dir)
        cache = Cache(self.cache_dir)
        assert os.path.isdir(self.cache_dir)

    def test_recreation(self):
        assert not os.path.isdir(self.cache_dir)
        cache = Cache(self.cache_dir)
        del cache
        assert os.path.isdir(self.cache_dir)
        cache = Cache(self.cache_dir)
        assert os.path.isdir(self.cache_dir)

    def tearDown(self):
        remove_path(self.cache_dir)

class TestCacheGeometry(unittest.TestCase):
    def setUp(self):
        self.cache_dir = random_tempdir('chroma_cache_test')
        self.cache = Cache(self.cache_dir)

        self.a = Geometry()
        self.a.add_solid(Solid(box(1,1,1)))
        self.a.add_solid(Solid(box(1,1,1)), displacement=(10,10,10))
        self.a.flatten()

        self.b = Geometry()
        self.b.add_solid(Solid(box(2,2,2)))
        self.b.add_solid(Solid(box(2,2,2)), displacement=(10,10,10))
        self.b.add_solid(Solid(box(2,2,2)), displacement=(-10,-10,-10))
        self.b.flatten()

    def test_list_geometry(self):
        self.assertEqual(len(self.cache.list_geometry()), 0)

        self.cache.save_geometry('a', self.a)
        l = self.cache.list_geometry()
        self.assertEqual(len(l), 1)
        self.assertIn('a', l)

        self.cache.save_geometry('b', self.b)
        l = self.cache.list_geometry()
        self.assertEqual(len(l), 2)
        self.assertIn('a', l)
        self.assertIn('b', l)

        self.cache.save_geometry('a', self.a)
        l = self.cache.list_geometry()
        self.assertEqual(len(l), 2)
        self.assertIn('a', l)
        self.assertIn('b', l)

    def test_load_geometry_not_found(self):
        with self.assertRaises(GeometryNotFoundError):
            self.cache.load_geometry('a')

    def test_save_load_new_geometry(self):
        self.cache.save_geometry('b', self.b)
        b = self.cache.load_geometry('b')

    def test_replace_geometry(self):
        self.cache.save_geometry('b', self.b)
        b = self.cache.load_geometry('b')
        self.assertEqual(b.mesh.md5(), self.b.mesh.md5())

        self.cache.save_geometry('b', self.b)
        b = self.cache.load_geometry('b')
        self.assertEqual(b.mesh.md5(), self.b.mesh.md5())

    def test_remove_geometry(self):
        self.cache.save_geometry('b', self.b)
        self.assertIn('b', self.cache.list_geometry())
        self.cache.remove_geometry('b')
        self.assertNotIn('b', self.cache.list_geometry())

    def test_get_geometry_hash(self):
        self.cache.save_geometry('b', self.b)
        self.assertEqual(self.cache.get_geometry_hash('b'), self.b.mesh.md5())

    def test_get_geometry_hash_not_found(self):
        with self.assertRaises(GeometryNotFoundError):
            self.cache.get_geometry_hash('a')        

    def test_default_geometry(self):
        self.cache.save_geometry('a', self.a)
        self.cache.save_geometry('b', self.b)

        with self.assertRaises(GeometryNotFoundError):
            self.cache.set_default_geometry('c')

        self.cache.set_default_geometry('b')
        b = self.cache.load_default_geometry()

        self.cache.set_default_geometry('a')
        a = self.cache.load_default_geometry()

    def test_default_geometry_corruption(self):
        self.cache.save_geometry('a', self.a)
        self.cache.save_geometry('b', self.b)

        # Put a file where a symlink should be
        default_symlink_path = self.cache.get_geometry_filename('.default')
        with open(default_symlink_path, 'w') as f:
            f.write('foo')

        with self.assertRaises(IOError):
            self.cache.set_default_geometry('b')

        # Verify file not modified
        assert os.path.isfile(default_symlink_path)
        with open(default_symlink_path) as f:
            self.assertEqual(f.read(), 'foo')

    def tearDown(self):
        remove_path(self.cache_dir)

class TestCacheBVH(unittest.TestCase):
    def setUp(self):
        self.cache_dir = random_tempdir('chroma_cache_test')
        self.cache = Cache(self.cache_dir)

        self.a = Geometry()
        self.a.add_solid(Solid(box(1,1,1)))
        self.a.add_solid(Solid(box(1,1,1)), displacement=(10,10,10))
        self.a.flatten()

        self.b = Geometry()
        self.b.add_solid(Solid(box(2,2,2)))
        self.b.add_solid(Solid(box(2,2,2)), displacement=(10,10,10))
        self.b.add_solid(Solid(box(2,2,2)), displacement=(-10,-10,-10))
        self.b.flatten()

        # c is not in cache
        self.c = Geometry()
        self.c.add_solid(Solid(box(2,2,2)))
        self.c.flatten()

        self.a_hash = self.a.mesh.md5()
        self.b_hash = self.b.mesh.md5()
        self.c_hash = self.c.mesh.md5()

        self.cache.save_geometry('a', self.a)
        self.cache.save_geometry('b', self.b)

    def test_list_bvh(self):
        self.assertEqual(len(self.cache.list_bvh(self.a_hash)), 0)
        self.cache.save_bvh([], self.a_hash)
        self.assertIn('default', self.cache.list_bvh(self.a_hash))
        self.cache.save_bvh([], self.a_hash, 'foo')
        self.assertIn('foo', self.cache.list_bvh(self.a_hash))
        self.assertEqual(len(self.cache.list_bvh(self.a_hash)), 2)

    def test_exist_bvh(self):
        self.cache.save_bvh([], self.a_hash)
        assert self.cache.exist_bvh(self.a_hash)
        self.cache.save_bvh([], self.a_hash, 'foo')
        assert self.cache.exist_bvh(self.a_hash, 'foo')
        
    def test_load_bvh_not_found(self):
        with self.assertRaises(BVHNotFoundError):
            self.cache.load_bvh(self.c_hash)

        with self.assertRaises(BVHNotFoundError):
            self.cache.load_bvh(self.a_hash, 'foo')

    def test_save_load_new_bvh(self):
        self.cache.save_bvh([], self.a_hash)
        self.cache.load_bvh(self.a_hash)
        self.cache.save_bvh([], self.a_hash, 'foo')
        self.cache.load_bvh(self.a_hash, 'foo')

    def test_remove_bvh(self):
        self.cache.remove_bvh(self.a_hash, 'does_not_exist')

        self.cache.save_bvh([], self.a_hash)
        self.cache.save_bvh([], self.a_hash, 'foo')
        assert self.cache.exist_bvh(self.a_hash)
        assert self.cache.exist_bvh(self.a_hash, 'foo')

        self.cache.remove_bvh(self.a_hash)
        assert not self.cache.exist_bvh(self.a_hash)
        assert self.cache.exist_bvh(self.a_hash, 'foo')

        self.cache.remove_bvh(self.a_hash, 'foo')
        assert not self.cache.exist_bvh(self.a_hash)
        assert not self.cache.exist_bvh(self.a_hash, 'foo')
        
    def tearDown(self):
        remove_path(self.cache_dir)
