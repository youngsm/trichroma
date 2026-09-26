import numpy as np
from pycuda import gpuarray as ga

from chroma.gpu.tools import get_cu_module, cuda_options, GPUFuncs, \
    to_float3

class GPURays(object):
    """The GPURays class holds arrays of ray positions and directions
    on the GPU that are used to render a geometry."""
    def __init__(self, pos, dir, max_alpha_depth=10, nblocks=64):
        self.pos = ga.to_gpu(to_float3(pos))
        self.dir = ga.to_gpu(to_float3(dir))

        self.max_alpha_depth = max_alpha_depth

        self.nblocks = nblocks

        transform_module = get_cu_module('transform.cu', options=cuda_options)
        self.transform_funcs = GPUFuncs(transform_module)

        render_module = get_cu_module('render.cu', options=cuda_options)
        self.render_funcs = GPUFuncs(render_module)

        self.dx = ga.empty(max_alpha_depth*self.pos.size, dtype=np.float32)
        self.color = ga.empty(self.dx.size, dtype=ga.vec.float4)
        self.dxlen = ga.zeros(self.pos.size, dtype=np.uint32)

    def rotate(self, phi, n):
        "Rotate by an angle phi around the axis `n`."
        self.transform_funcs.rotate(np.int32(self.pos.size), self.pos, np.float32(phi), ga.vec.make_float3(*n), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))
        self.transform_funcs.rotate(np.int32(self.dir.size), self.dir, np.float32(phi), ga.vec.make_float3(*n), block=(self.nblocks,1,1), grid=(self.dir.size//self.nblocks+1,1))

    def rotate_around_point(self, phi, n, point):
        """"Rotate by an angle phi around the axis `n` passing through
        the point `point`."""
        self.transform_funcs.rotate_around_point(np.int32(self.pos.size), self.pos, np.float32(phi), ga.vec.make_float3(*n), ga.vec.make_float3(*point), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))
        self.transform_funcs.rotate(np.int32(self.dir.size), self.dir, np.float32(phi), ga.vec.make_float3(*n), block=(self.nblocks,1,1), grid=(self.dir.size//self.nblocks+1,1))

    def translate(self, v):
        "Translate the ray positions by the vector `v`."
        self.transform_funcs.translate(np.int32(self.pos.size), self.pos, ga.vec.make_float3(*v), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))

    def render(self, gpu_geometry, pixels, alpha_depth=10,
               keep_last_render=False,bg_color=0x00000000):
        """Render `gpu_geometry` and fill the GPU array `pixels` with pixel
        colors."""
        if not keep_last_render:
            self.dxlen.fill(0)

        if alpha_depth > self.max_alpha_depth:
            raise Exception('alpha_depth > max_alpha_depth')

        if not isinstance(pixels, ga.GPUArray):
            raise TypeError('`pixels` must be a %s instance.' % ga.GPUArray)

        if pixels.size != self.pos.size:
            raise ValueError('`pixels`.size != number of rays')

        self.render_funcs.render(np.int32(self.pos.size), self.pos, self.dir, gpu_geometry.gpudata, np.uint32(alpha_depth), pixels, self.dx, self.dxlen, self.color,np.uint32(bg_color), block=(self.nblocks,1,1), grid=(self.pos.size//self.nblocks+1,1))

    def snapshot(self, gpu_geometry, alpha_depth=10):
        "Render `gpu_geometry` and return a numpy array of pixel colors."
        pixels = ga.empty(self.pos.size, dtype=np.uint32)
        self.render(gpu_geometry, pixels, alpha_depth)
        return pixels.get()

