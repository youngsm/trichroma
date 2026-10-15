import os
import numpy as np
import time
from pycuda import autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.gpuarray as ga
from chroma.gpu.tools import to_float3
from chroma.transform import rotate, normalize
from chroma.cuda import srcdir as source_directory

print('device %s' % autoinit.device.name())

current_directory = os.path.split(os.path.realpath(__file__))[0]
source = open(current_directory + '/rotate_test.cu').read()
mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True)

rotate_gpu = mod.get_function('rotate')

nthreads_per_block = 1024
blocks = 4096

def test_rotate():
    n = nthreads_per_block*blocks

    a = np.random.rand(n,3).astype(np.float32)
    t = np.random.rand(n).astype(np.float32)*2*np.pi
    w = normalize(np.random.rand(3))

    a_gpu = ga.to_gpu(to_float3(a))
    t_gpu = ga.to_gpu(t)

    dest_gpu = ga.empty(n,dtype=ga.vec.float3)

    t0 = time.time()
    rotate_gpu(a_gpu,t_gpu,ga.vec.make_float3(*w),dest_gpu,
               block=(nthreads_per_block,1,1),grid=(blocks,1))
    autoinit.context.synchronize()
    elapsed = time.time() - t0;

    print('elapsed %f sec' % elapsed)

    r = rotate(a,t,w)

    assert np.allclose(r,dest_gpu.get().view(np.float32).reshape((-1,3)),
                       atol=1e-5)

if __name__ == '__main__':
    test_rotate()
