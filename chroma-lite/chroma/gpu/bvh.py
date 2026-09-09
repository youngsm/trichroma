import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray as ga
from pycuda import characterize

from chroma.gpu.tools import get_cu_module, cuda_options, \
    chunk_iterator, to_uint3, to_float3, GPUFuncs, mapped_empty, Mapped

from chroma.bvh.bvh import WorldCoords, node_areas, NCHILD_MASK, CHILD_BITS

def round_up_to_multiple(x, multiple):
    remainder = x % multiple
    if remainder == 0:
        return x
    else:
        return x + multiple - remainder

def create_leaf_nodes(mesh, morton_bits=16, round_to_multiple=1):
    '''Compute the leaf nodes surrounding a triangle mesh.

      ``mesh``: chroma.geometry.Mesh
        Triangles to box
      ``morton_bits``: int
        Number of bits to use per dimension when computing Morton code.
      ``round_to_multiple``: int
        Round the number of nodes created up to multiple of this number
        Extra nodes will be all zero.
        
    Returns (world_coords, nodes, morton_codes), where
      ``world_coords``: chroma.bvh.WorldCoords
        Defines the fixed point coordinate system
      ``nodes``: ndarray(shape=len(mesh.triangles), dtype=uint4)
        List of leaf nodes.  Child IDs will be set to triangle offsets.
      ``morton_codes``: ndarray(shape=len(mesh.triangles), dtype=np.uint64)
        Morton codes for each triangle, using ``morton_bits`` per axis.
        Must be <= 16 bits.
    '''
    # Load GPU functions
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)

    # compute world coordinates
    world_origin = mesh.vertices.min(axis=0)
    world_scale = np.max((mesh.vertices.max(axis=0) - world_origin)) \
        / (2**16 - 2)
    world_coords = WorldCoords(world_origin=world_origin, 
                               world_scale=world_scale)

    # Put triangles and vertices in mapped host memory
    triangles = mapped_empty(shape=len(mesh.triangles), dtype=ga.vec.uint3,
                             write_combined=True)
    triangles[:] = to_uint3(mesh.triangles)
    vertices = mapped_empty(shape=len(mesh.vertices), dtype=ga.vec.float3,
                            write_combined=True)
    vertices[:] = to_float3(mesh.vertices)
    
    # Call GPU to compute nodes
    nodes = ga.zeros(shape=round_up_to_multiple(len(triangles), 
                                                round_to_multiple),
                     dtype=ga.vec.uint4)
    morton_codes = ga.empty(shape=len(triangles), dtype=np.uint64)

    # Convert world coords to GPU-friendly types
    world_origin = ga.vec.make_float3(*world_origin)
    world_scale = np.float32(world_scale)

    nthreads_per_block = 256
    for first_index, elements_this_iter, nblocks_this_iter in \
            chunk_iterator(len(triangles), nthreads_per_block, 
                           max_blocks=30000):
        bvh_funcs.make_leaves(np.uint32(first_index),
                              np.uint32(elements_this_iter),
                              Mapped(triangles), Mapped(vertices),
                              world_origin, world_scale,
                              nodes, morton_codes,
                              block=(nthreads_per_block,1,1),
                              grid=(nblocks_this_iter,1))

    morton_codes_host = morton_codes.get() >> (16 - morton_bits)
    return world_coords, nodes.get(), morton_codes_host


def merge_nodes_detailed(nodes, first_child, nchild):
    '''Merges nodes into len(first_child) parent nodes, using
    the provided arrays to determine the index of the first
    child of each parent, and how many children there are.'''
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)

    gpu_nodes = ga.to_gpu(nodes)
    gpu_first_child = ga.to_gpu(first_child.astype(np.int32))
    gpu_nchild = ga.to_gpu(nchild.astype(np.int32))

    nparent = len(first_child)
    gpu_parent_nodes = ga.empty(shape=nparent, dtype=ga.vec.uint4)

    nthreads_per_block = 256
    for first_index, elements_this_iter, nblocks_this_iter in \
            chunk_iterator(nparent, nthreads_per_block, max_blocks=10000):

        bvh_funcs.make_parents_detailed(np.uint32(first_index),
                                        np.uint32(elements_this_iter),
                                        gpu_nodes,
                                        gpu_parent_nodes,
                                        gpu_first_child,
                                        gpu_nchild,
                                        block=(nthreads_per_block,1,1),
                                        grid=(nblocks_this_iter,1))

    return gpu_parent_nodes.get()

def collapse_chains(nodes, layer_bounds):
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)
    
    gpu_nodes = ga.to_gpu(nodes)

    bounds = list(zip(layer_bounds[:-1], layer_bounds[1:]))[:-1]
    bounds.reverse()
    nthreads_per_block = 256
    for start, end in bounds:
        bvh_funcs.collapse_child(np.uint32(start),
                                 np.uint32(end),
                                 gpu_nodes,
                                 block=(nthreads_per_block,1,1),
                                 grid=(120,1))
    return gpu_nodes.get()

def area_sort_nodes(gpu_geometry, layer_bounds):
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)

    bounds = list(zip(layer_bounds[:-1], layer_bounds[1:]))[:-1]
    bounds.reverse()
    nthreads_per_block = 256
    for start, end in bounds:
        bvh_funcs.area_sort_child(np.uint32(start),
                                  np.uint32(end),
                                  gpu_geometry,
                                  block=(nthreads_per_block,1,1),
                                  grid=(120,1))
    return gpu_geometry.nodes.get()


def merge_nodes(nodes, degree, max_ratio=None):
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)
    
    nparent = len(nodes) / degree
    if len(nodes) % degree != 0:
        nparent += 1

    if nparent == 1:
        nparent_pad = nparent
    else:
        nparent_pad = round_up_to_multiple(nparent, 1)#degree)
    gpu_parent_nodes = ga.zeros(shape=nparent_pad, dtype=ga.vec.uint4)

    nthreads_per_block = 256
    for first_index, elements_this_iter, nblocks_this_iter in \
            chunk_iterator(nparent, nthreads_per_block, max_blocks=10000):
        bvh_funcs.make_parents(np.uint32(first_index),
                               np.uint32(elements_this_iter),
                               np.uint32(degree),
                               gpu_parent_nodes,
                               cuda.In(nodes),
                               np.uint32(0),
                               np.uint32(len(nodes)),
                               block=(nthreads_per_block,1,1),
                               grid=(nblocks_this_iter,1))

    parent_nodes = gpu_parent_nodes.get()

    if max_ratio is not None:
        areas = node_areas(parent_nodes)
        child_areas = node_areas(nodes)

        excessive_area = np.zeros(shape=len(areas), dtype=bool)
        for i, parent_area in enumerate(areas):
            nchild = parent_nodes['w'][i] >> CHILD_BITS
            child_index = parent_nodes['w'][i] & ~NCHILD_MASK
            child_area = child_areas[child_index:child_index+nchild].sum()
            #if parent_area > 1e9:
            #    print i, 'Children: %e, Parent: %e' % (child_area, parent_area)
            if child_area/parent_area < 0.3:
                excessive_area[i] = True
                #print i, 'Children: %e, Parent: %e' % (child_area, parent_area)

        extra_slots = round_up_to_multiple((degree - 1) * np.count_nonzero(excessive_area), 1)
        print('Extra slots:', extra_slots)
        new_parent_nodes = np.zeros(shape=len(parent_nodes) + extra_slots,
                                    dtype=parent_nodes.dtype)
        new_parent_nodes[:len(parent_nodes)] = parent_nodes

        offset = 0
        for count, index in enumerate(np.argwhere(excessive_area)):
            index = index[0] + offset
            nchild = new_parent_nodes['w'][index] >> CHILD_BITS
            child_index = new_parent_nodes['w'][index] & ~NCHILD_MASK
            new_parent_nodes[index] = nodes[child_index]
            #new_parent_nodes['w'][index] = 1 << CHILD_BITS | child_index
            tmp_nchild = new_parent_nodes['w'][index] >> CHILD_BITS
            tmp_child_index = new_parent_nodes['w'][index] & ~NCHILD_MASK
            new_parent_nodes['w'][index] = tmp_nchild << CHILD_BITS | (tmp_child_index + len(nodes))

            if nchild == 1:
                continue

            # slide everyone over
            #print index, nchild, len(new_parent_nodes)
            new_parent_nodes[index+nchild:] = new_parent_nodes[index+1:-nchild+1]
            offset += nchild - 1
            for sibling in range(nchild - 1):
                new_parent_index = index + 1 + sibling
                new_parent_nodes[new_parent_index] = nodes[child_index + sibling + 1]
                if new_parent_nodes['x'][new_parent_index] != 0:
                    tmp_nchild = new_parent_nodes['w'][new_parent_index] >> CHILD_BITS
                    tmp_child_index = new_parent_nodes['w'][new_parent_index] & ~NCHILD_MASK
                    new_parent_nodes['w'][new_parent_index] = tmp_nchild << CHILD_BITS | (tmp_child_index + len(nodes))

                    #new_parent_nodes['w'][new_parent_index] = 1 << CHILD_BITS | (child_index + sibling + 1)


            #print 'intermediate: %e' % node_areas(new_parent_nodes).max()
        print('old: %e' % node_areas(parent_nodes).max())
        print('new: %e' % node_areas(new_parent_nodes).max())
        if len(new_parent_nodes) < len(nodes):
            # Only adopt new set of parent nodes if it actually reduces the
            # total number of nodes at this level by 1.
            parent_nodes = new_parent_nodes

    return parent_nodes

def concatenate_layers(layers):
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)
    # Put 0 at beginning of list
    layer_bounds = np.insert(np.cumsum(list(map(len, layers))), 0, 0)
    nodes = ga.empty(shape=int(layer_bounds[-1]), dtype=ga.vec.uint4)
    nthreads_per_block = 256

    for layer_start, layer_end, layer in zip(layer_bounds[:-1],
                                             layer_bounds[1:],
                                             layers):
        if layer_end == layer_bounds[-1]:
            # leaf nodes need no offset
            child_offset = 0
        else:
            child_offset = layer_end

        for first_index, elements_this_iter, nblocks_this_iter in \
                chunk_iterator(layer_end-layer_start, nthreads_per_block,
                               max_blocks=10000):
            bvh_funcs.copy_and_offset(np.uint32(first_index),
                                      np.uint32(elements_this_iter),
                                      np.uint32(child_offset),
                                      cuda.In(layer),
                                      nodes[layer_start:],
                                      block=(nthreads_per_block,1,1),
                                      grid=(nblocks_this_iter,1))
    return nodes.get(), layer_bounds

def optimize_layer(orig_nodes):
    bvh_module = get_cu_module('bvh.cu', options=cuda_options,
                               include_source_directory=True)
    bvh_funcs = GPUFuncs(bvh_module)

    nodes = ga.to_gpu(orig_nodes)
    n = len(nodes)
    areas = ga.empty(shape=n/2, dtype=np.uint64)
    nthreads_per_block = 128

    min_areas = ga.empty(shape=int(np.ceil(n/float(nthreads_per_block))), dtype=np.uint64)
    min_index = ga.empty(shape=min_areas.shape, dtype=np.uint32)

    update = 10000

    skip_size = 1
    flag = mapped_empty(shape=skip_size, dtype=np.uint32)

    i = 0
    skips = 0
    swaps = 0
    while i < n/2 - 1:
        # How are we doing?
        if i % update == 0:
            for first_index, elements_this_iter, nblocks_this_iter in \
                    chunk_iterator(n/2, nthreads_per_block, max_blocks=10000):

                bvh_funcs.pair_area(np.uint32(first_index),
                                    np.uint32(elements_this_iter),
                                    nodes,
                                    areas,
                                    block=(nthreads_per_block,1,1),
                                    grid=(nblocks_this_iter,1))
                
            areas_host = areas.get()
            #print nodes.get(), areas_host.astype(float)
            print('Area of parent layer so far (%d): %1.12e' % (i*2, areas_host.astype(float).sum()))
            print('Skips: %d, Swaps: %d' % (skips, swaps))

        test_index = i * 2

        blocks = 0
        look_forward = min(8192*50, n - test_index - 2)
        skip_this_round = min(skip_size, n - test_index - 1)
        flag[:] = 0
        for first_index, elements_this_iter, nblocks_this_iter in \
                chunk_iterator(look_forward, nthreads_per_block, max_blocks=10000):
            bvh_funcs.min_distance_to(np.uint32(first_index + test_index + 2),
                                      np.uint32(elements_this_iter),
                                      np.uint32(test_index),
                                      nodes,
                                      np.uint32(blocks),
                                      min_areas,
                                      min_index,
                                      Mapped(flag),
                                      block=(nthreads_per_block,1,1),
                                      grid=(nblocks_this_iter, skip_this_round))
            blocks += nblocks_this_iter
            #print i, first_index, nblocks_this_iter, look_forward
        cuda.Context.get_current().synchronize()

        if flag[0] == 0:
            flag_nonzero = flag.nonzero()[0]
            if len(flag_nonzero) == 0:
                no_swap_required = skip_size
            else:
                no_swap_required = flag_nonzero[0]
            i += no_swap_required
            skips += no_swap_required
            continue

        min_areas_host = min_areas[:blocks].get()
        min_index_host = min_index[:blocks].get()
        best_block = min_areas_host.argmin()
        better_i = min_index_host[best_block]

        swaps += 1
        #print 'swap', test_index+1, better_i
        assert 0 < better_i < len(nodes)
        assert 0 < test_index + 1 < len(nodes)
        bvh_funcs.swap(np.uint32(test_index+1), np.uint32(better_i),
                       nodes, block=(1,1,1), grid=(1,1))
        cuda.Context.get_current().synchronize()
        i += 1

    for first_index, elements_this_iter, nblocks_this_iter in \
            chunk_iterator(n/2, nthreads_per_block, max_blocks=10000):

        bvh_funcs.pair_area(np.uint32(first_index),
                            np.uint32(elements_this_iter),
                            nodes,
                            areas,
                            block=(nthreads_per_block,1,1),
                            grid=(nblocks_this_iter,1))
        
    areas_host = areas.get()

    print('Final area of parent layer: %1.12e' % areas_host.sum())
    print('Skips: %d, Swaps: %d' % (skips, swaps))

    return nodes.get()
