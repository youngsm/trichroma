import pycuda.autoinit
from .unittest_find import unittest
from chroma.bvh import make_simple_bvh, BVH
from chroma.bvh.bvh import node_areas
import chroma.models

import numpy as np
#from numpy.testing import assert_array_max_ulp, assert_array_equal, \
#    assert_approx_equal

def build_simple_bvh(degree):
    mesh = chroma.models.lionsolid()
    bvh = make_simple_bvh(mesh, degree)

    nodes = bvh.nodes
    layer_bounds = np.append(bvh.layer_offsets, len(nodes))
    world_coords = bvh.world_coords

    for i, (layer_start, layer_end) in enumerate(zip(layer_bounds[:-1], 
                                                     layer_bounds[1:])):
        print(i, node_areas(nodes[layer_start:layer_end]).sum() * world_coords.world_scale**2)
    

    assert isinstance(bvh, BVH)

def test_simple():
    yield build_simple_bvh, 2
    yield build_simple_bvh, 3
    yield build_simple_bvh, 4
