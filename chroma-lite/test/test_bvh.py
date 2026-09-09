from .unittest_find import unittest
from chroma.bvh.bvh import BVH, BVHLayerSlice, WorldCoords, uint4, \
    OutOfRangeError, unpack_nodes
import numpy as np
from numpy.testing import assert_array_max_ulp, assert_array_equal, \
    assert_approx_equal

def lslice(layer_bounds, layer_number):
    '''Returns a slice object for retrieving a particular layer in an
    array of nodes.'''
    return slice(layer_bounds[layer_number], layer_bounds[layer_number+1])

class TestWorldCoords(unittest.TestCase):
    def setUp(self):
        self.coords = WorldCoords([-1,-1,-1], 0.1)

    def test_fixed_to_world(self):
        f = [0, 1, 100]
        assert_array_max_ulp(self.coords.fixed_to_world(f), 
                             [-1.0, -0.9, 9.0], dtype=np.float32)
    
    def test_world_to_fixed(self):
        w = [-1.0, -0.9, 9.0]
        assert_array_equal(self.coords.world_to_fixed(w), 
                           [0, 1, 100])


    def test_fixed_array_to_world(self):
        f = [[0, 1, 100],
             [20, 40, 60],
             [210, 310, 410]]

        assert_array_max_ulp(self.coords.fixed_to_world(f), 
                             [[-1.0, -0.9, 9.0], 
                              [1.0, 3.0, 5.0],
                              [20.0, 30.0, 40.0]],
                             dtype=np.float32)

    def test_world_array_to_fixed(self):
        w = [[-1.0, -0.9, 9.0], 
             [1.0, 3.0, 5.0],
             [20.0, 30.0, 40.0]]
        assert_array_equal(self.coords.world_to_fixed(w), 
                           [[0, 1, 100],
                            [20, 40, 60],
                            [210, 310, 410]])

    def test_out_of_range(self):
        with self.assertRaises(OutOfRangeError):
            self.coords.world_to_fixed([-2.0, 0.0, 0.0])
        with self.assertRaises(OutOfRangeError):
            self.coords.world_to_fixed([0.0, 1e9, 0.0])

def create_bvh():
    # 3 layer binary tree
    degree = 2
    world_coords = WorldCoords(np.array([-1.0,-1.0,-1.0]), 0.1)
    layer_bounds = [0, 1, 3, 7]
    nodes = np.empty(shape=layer_bounds[-1], dtype=uint4)

    # bottom layer
    layer = lslice(layer_bounds, 2)
    nodes['x'][layer] = [ 0x00010000, 0x00020001,
                          0x00010000, 0x00010000]
    nodes['y'][layer] = [ 0x00010000, 0x00010000,
                          0x00020001, 0x00010000]
    nodes['z'][layer] = [ 0x00010000, 0x00010000,
                          0x00010000, 0x00020001]
    nodes['w'][layer] = 0x80000000 # leaf nodes

    # middle layer
    layer = lslice(layer_bounds, 1)
    nodes['x'][layer] = [ 0x00020000, 0x00010000 ]
    nodes['y'][layer] = [ 0x00010000, 0x00020000 ]
    nodes['z'][layer] = [ 0x00010000, 0x00020000 ]
    nodes['w'][layer] = [ 0x00000003, 0x00000005 ]

    # top layer
    layer = lslice(layer_bounds, 0)
    nodes['x'][layer] = [ 0x00020000 ]
    nodes['y'][layer] = [ 0x00020000 ]
    nodes['z'][layer] = [ 0x00020000 ]
    nodes['w'][layer] = [ 0x00000001 ]
    
    layer_offsets = list(layer_bounds[:-1]) # trim last entry
    bvh = BVH(world_coords=world_coords,
              nodes=nodes, layer_offsets=layer_offsets)

    return bvh

def test_unpack_nodes():
    bvh = create_bvh()

    layer = bvh.get_layer(2)
    unpack = unpack_nodes(layer.nodes)
    assert_array_equal(unpack['xlo'], [0, 1, 0, 0])
    assert_array_equal(unpack['xhi'], [1, 2, 1, 1])
    assert_array_equal(unpack['ylo'], [0, 0, 1, 0])
    assert_array_equal(unpack['yhi'], [1, 1, 2, 1])
    assert_array_equal(unpack['zlo'], [0, 0, 0, 1])
    assert_array_equal(unpack['zhi'], [1, 1, 1, 2])

    layer = bvh.get_layer(1)
    unpack = unpack_nodes(layer.nodes)
    assert_array_equal(unpack['xlo'], [0, 0])
    assert_array_equal(unpack['xhi'], [2, 1])
    assert_array_equal(unpack['ylo'], [0, 0])
    assert_array_equal(unpack['yhi'], [1, 2])
    assert_array_equal(unpack['zlo'], [0, 0])
    assert_array_equal(unpack['zhi'], [1, 2])
    assert_array_equal(unpack['child'], [3, 5])

    layer = bvh.get_layer(0)
    unpack = unpack_nodes(layer.nodes)
    assert_array_equal(unpack['xlo'], [0])
    assert_array_equal(unpack['xhi'], [2])
    assert_array_equal(unpack['ylo'], [0])
    assert_array_equal(unpack['yhi'], [2])
    assert_array_equal(unpack['zlo'], [0])
    assert_array_equal(unpack['zhi'], [2])
    assert_array_equal(unpack['child'], [1])


class TestBVH(unittest.TestCase):
    def setUp(self):
        self.bvh = create_bvh()

    def test_len(self):
        self.assertEqual(len(self.bvh), 7)

    def test_layer_count(self):
        self.assertEqual(self.bvh.layer_count(), 3)

    def test_get_layer(self):
        layer = self.bvh.get_layer(0)
        self.assertEqual(len(layer), 1)

        layer = self.bvh.get_layer(1)
        self.assertEqual(len(layer), 2)

        layer = self.bvh.get_layer(2)
        self.assertEqual(len(layer), 4)

class TestBVHLayer(unittest.TestCase):
    def setUp(self):
        self.bvh = create_bvh()

    def test_area(self):
        layer = self.bvh.get_layer(2)
        assert_approx_equal(layer.area(), 4*6*0.1**2)

        layer = self.bvh.get_layer(1)
        assert_approx_equal(layer.area(), (4*2+2 + 4*2+2*4)*0.1**2)

        layer = self.bvh.get_layer(0)
        assert_approx_equal(layer.area(), (6*2*2)*0.1**2)
