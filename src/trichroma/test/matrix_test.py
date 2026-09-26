import os
import numpy as np
np.seterr(divide='ignore')
from pycuda import autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from pycuda import gpuarray

float3 = gpuarray.vec.float3

print('device %s' % autoinit.device.name())

current_directory = os.path.split(os.path.realpath(__file__))[0]

from chroma.cuda import srcdir as source_directory

source = open(current_directory + '/matrix_test.cu').read()

mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True, cache_dir=False)

det = mod.get_function('det')
inv = mod.get_function('inv')
matrixadd = mod.get_function('matrixadd')
matrixsub = mod.get_function('matrixsub')
matrixmul = mod.get_function('matrixmul')
multiply = mod.get_function('multiply')
matrixaddfloat = mod.get_function('matrixaddfloat')
matrixsubfloat = mod.get_function('matrixsubfloat')
matrixmulfloat = mod.get_function('matrixmulfloat')
matrixdivfloat = mod.get_function('matrixdivfloat')
floataddmatrix = mod.get_function('floataddmatrix')
floatsubmatrix = mod.get_function('floatsubmatrix')
floatmulmatrix = mod.get_function('floatmulmatrix')
floatdivmatrix = mod.get_function('floatdivmatrix')
matrixaddequals = mod.get_function('matrixaddequals')
matrixsubequals = mod.get_function('matrixsubequals')
matrixaddequalsfloat = mod.get_function('matrixaddequalsfloat')
matrixsubequalsfloat = mod.get_function('matrixsubequalsfloat')
matrixmulequalsfloat = mod.get_function('matrixmulequalsfloat')
matrixdivequalsfloat = mod.get_function('matrixdivequalsfloat')
outer = mod.get_function('outer')
minusmatrix = mod.get_function('minusmatrix')

size = {'block': (1,1,1), 'grid': (1,1)}

# FIXME: Need to refactor this into proper unit tests
def test_matrix():
    for i in range(1):
        a = np.random.random_sample(size=9).astype(np.float32)
        b = np.random.random_sample(size=9).astype(np.float32)
        dest = np.empty(1, dtype=np.float32)
        c = np.int32(np.random.random_sample())

        print('testing det...', end=' ')

        det(cuda.In(a), cuda.Out(dest), **size)

        if not np.allclose(np.float32(np.linalg.det(a.reshape(3,3))), dest[0]):
            print('fail')
            print(np.float32(np.linalg.det(a.reshape(3,3))))
            print(dest[0])
        else:
            print('success')

        print('testing inv...', end=' ')

        dest = np.empty(9, dtype=np.float32)

        inv(cuda.In(a), cuda.Out(dest), **size)

        if not np.allclose(np.linalg.inv(a.reshape(3,3)).flatten().astype(np.float32), dest):
            print('fail')
            print(np.linalg.inv(a.reshape(3,3)).flatten().astype(np.float32))
            print(dest)
        else:
            print('success')

        print('testing matrixadd...', end=' ')

        matrixadd(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)

        if not np.allclose(a+b, dest):
            print('fail')
            print(a+b)
            print(dest)
        else:
            print('success')

        print('testing matrixsub...', end=' ')

        matrixsub(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)

        if not np.allclose(a-b, dest):
            print('fail')
            print(a-b)
            print(dest)
        else:
            print('sucess')

        print('testing matrixmul...', end=' ')

        matrixmul(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)

        if not np.allclose(np.dot(a.reshape(3,3),b.reshape(3,3)).flatten(), dest):
            print('fail')
            print(np.dot(a.reshape(3,3),b.reshape(3,3)).flatten())
            print(dest)
        else:
            print('success')

        print('testing multiply...', end=' ')

        x_cpu = np.random.random_sample(size=3).astype(np.float32)
        x_gpu = np.array((x_cpu[0], x_cpu[1], x_cpu[2]), dtype=float3)

        dest = np.empty(1, dtype=float3)
        
        multiply(cuda.In(a), cuda.In(x_gpu), cuda.Out(dest), **size)

        m = a.reshape(3,3)

        if not np.allclose(np.dot(x_cpu,m[0]), dest[0]['x']) or \
                not np.allclose(np.dot(x_cpu,m[1]), dest[0]['y']) or \
                not np.allclose(np.dot(x_cpu,m[2]), dest[0]['z']):
            print('fail')
            print(np.dot(x_cpu,m[0]))
            print(np.dot(x_cpu,m[1]))
            print(np.dot(x_cpu,m[2]))
            print(dest[0]['x'])
            print(dest[0]['y'])
            print(dest[0]['z'])
        else:
            print('success')

        print('testing matrixaddfloat...', end=' ')

        dest = np.empty(9, dtype=np.float32)

        matrixaddfloat(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(a+c, dest):
            print('fail')
            print(a+c)
            print(dest)
        else:
            print('success')

        print('testing matrixsubfloat...', end=' ')

        matrixsubfloat(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(a-c, dest):
            print('fail')
            print(a-c)
            print(dest)
        else:
            print('success')

        print('testing matrixmulfloat...', end=' ')

        matrixmulfloat(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(a*c, dest):
            print('fail')
            print(a-c)
            print(dest)
        else:
            print('success')

        print('testing matrixdivfloat...', end=' ')

        matrixdivfloat(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(a/c, dest):
            print('fail')
            print(a/c)
            print(dest)
        else:
            print('success')

        print('testing floataddmatrix...', end=' ')

        floataddmatrix(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(c+a, dest):
            print('fail')
            print(c+a)
            print(dest)
        else:
            print('success')

        print('testing floatsubmatrix...', end=' ')

        floatsubmatrix(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(c-a, dest):
            print('fail')
            print(c-a)
            print(dest)
        else:
            print('success')

        print('testing floatmulmatrix...', end=' ')

        floatmulmatrix(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(c*a, dest):
            print('fail')
            print(c*a)
            print(dest)
        else:
            print('success')

        print('testing floatdivmatrix...', end=' ')

        floatdivmatrix(cuda.In(a), c, cuda.Out(dest), **size)

        if not np.allclose(c/a, dest):
            print('fail')
            print(c/a)
            print(dest)
        else:
            print('success')

        print('testing matrixaddequals...', end=' ')

        dest = np.copy(a)

        matrixaddequals(cuda.InOut(dest), cuda.In(b), **size)

        if not np.allclose(a+b, dest):
            print('fail')
            print(a+b)
            print(dest)
        else:
            print('success')

        print('testing matrixsubequals...', end=' ')

        dest = np.copy(a)

        matrixsubequals(cuda.InOut(dest), cuda.In(b), **size)

        if not np.allclose(a-b, dest):
            print('fail')
            print(a-b)
            print(dest)
        else:
            print('success')

        print('testing matrixaddequalsfloat...', end=' ')

        dest = np.copy(a)

        matrixaddequalsfloat(cuda.InOut(dest), c, **size)

        if not np.allclose(a+c, dest):
            print('fail')
            print(a+c)
            print(dest)
        else:
            print('success')

        print('testing matrixsubequalsfloat...', end=' ')

        dest = np.copy(a)

        matrixsubequalsfloat(cuda.InOut(dest), c, **size)

        if not np.allclose(a-c, dest):
            print('fail')
            print(a-c)
            print(dest)
        else:
            print('success')

        print('testing matrixmulequalsfloat...', end=' ')

        dest = np.copy(a)

        matrixmulequalsfloat(cuda.InOut(dest), c, **size)

        if not np.allclose(a*c, dest):
            print('fail')
            print(a*c)
            print(dest)
        else:
            print('success')

        print('testing matrixdivequalsfloat...', end=' ')

        dest = np.copy(a)

        matrixdivequalsfloat(cuda.InOut(dest), c, **size)

        if not np.allclose(a/c, dest):
            print('fail')
            print(a*c)
            print(dest)
        else:
            print('success')

        print('testing outer...', end=' ')

        x1_cpu = np.random.random_sample(size=3).astype(np.float32)
        x2_cpu = np.random.random_sample(size=3).astype(np.float32)

        x1_gpu = np.array((x1_cpu[0], x1_cpu[1], x1_cpu[2]), dtype=float3)
        x2_gpu = np.array((x2_cpu[0], x2_cpu[1], x2_cpu[2]), dtype=float3)

        outer(x1_gpu, x2_gpu, cuda.Out(dest), **size)

        if not np.allclose(np.outer(x1_cpu, x2_cpu).flatten(), dest):
            print('fail')
            print(np.outer(x1_cpu, x2_cpu).flatten())
            print(dest)
        else:
            print('success')

        print('testing minus matrix...', end=' ')

        dest = np.copy(a)

        minusmatrix(cuda.In(a), cuda.Out(dest), **size)

        if not np.allclose(-a, dest):
            print('fail')
            print(-a)
            print(dest)
        else:
            print('success')
