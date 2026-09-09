import os
import numpy as np
from pycuda import autoinit
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
from pycuda import gpuarray

float3 = gpuarray.vec.float3

print('device %s' % autoinit.device.name())

current_directory = os.path.split(os.path.realpath(__file__))[0]
from chroma.cuda import srcdir as source_directory

source = open(current_directory + '/linalg_test.cu').read()

mod = SourceModule(source, options=['-I' + source_directory], no_extern_c=True, cache_dir=False)

float3add = mod.get_function('float3add')
float3addequal = mod.get_function('float3addequal')
float3sub = mod.get_function('float3sub')
float3subequal = mod.get_function('float3subequal')
float3addfloat = mod.get_function('float3addfloat')
float3addfloatequal = mod.get_function('float3addfloatequal')
floataddfloat3 = mod.get_function('floataddfloat3')
float3subfloat = mod.get_function('float3subfloat')
float3subfloatequal = mod.get_function('float3subfloatequal')
floatsubfloat3 = mod.get_function('floatsubfloat3')
float3mulfloat = mod.get_function('float3mulfloat')
float3mulfloatequal = mod.get_function('float3mulfloatequal')
floatmulfloat3 = mod.get_function('floatmulfloat3')
float3divfloat = mod.get_function('float3divfloat')
float3divfloatequal = mod.get_function('float3divfloatequal')
floatdivfloat3 = mod.get_function('floatdivfloat3')
dot = mod.get_function('dot')
cross = mod.get_function('cross')
norm = mod.get_function('norm')
minusfloat3 = mod.get_function('minusfloat3')

size = {'block': (256,1,1), 'grid': (1,1)}

a = np.empty(size['block'][0], dtype=float3)
b = np.empty(size['block'][0], dtype=float3)
c = np.float32(np.random.random_sample())

a['x'] = np.random.random_sample(size=a.size)
a['y'] = np.random.random_sample(size=a.size)
a['z'] = np.random.random_sample(size=a.size)

b['x'] = np.random.random_sample(size=b.size)
b['y'] = np.random.random_sample(size=b.size)
b['z'] = np.random.random_sample(size=b.size)

def testfloat3add():
    dest = np.empty(a.size, dtype=float3)
    float3add(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)
    if not np.allclose(a['x']+b['x'], dest['x']) or \
            not np.allclose(a['y']+b['y'], dest['y']) or \
            not np.allclose(a['z']+b['z'], dest['z']):
        assert False

def testfloat3sub():
    dest = np.empty(a.size, dtype=float3)
    float3sub(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)
    if not np.allclose(a['x']-b['x'], dest['x']) or \
            not np.allclose(a['y']-b['y'], dest['y']) or \
            not np.allclose(a['z']-b['z'], dest['z']):
        assert False

def testfloat3addequal():
    dest = np.copy(a)
    float3addequal(cuda.InOut(dest), cuda.In(b), **size)
    if not np.allclose(a['x']+b['x'], dest['x']) or \
            not np.allclose(a['y']+b['y'], dest['y']) or \
            not np.allclose(a['z']+b['z'], dest['z']):
        assert False

def testfloat3subequal():
    dest = np.copy(a)
    float3subequal(cuda.InOut(dest), cuda.In(b), **size)
    if not np.allclose(a['x']-b['x'], dest['x']) or \
            not np.allclose(a['y']-b['y'], dest['y']) or \
            not np.allclose(a['z']-b['z'], dest['z']):
        assert False

def testfloat3addfloat():
    dest = np.empty(a.size, dtype=float3)
    float3addfloat(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(a['x']+c, dest['x']) or \
            not np.allclose(a['y']+c, dest['y']) or \
            not np.allclose(a['z']+c, dest['z']):
        assert False

def testfloat3addfloatequal():
    dest = np.copy(a)
    float3addfloatequal(cuda.InOut(dest), c, **size)
    if not np.allclose(a['x']+c, dest['x']) or \
            not np.allclose(a['y']+c, dest['y']) or \
            not np.allclose(a['z']+c, dest['z']):
        assert False

def testfloataddfloat3():
    dest = np.empty(a.size, dtype=float3)
    floataddfloat3(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(c+a['x'], dest['x']) or \
            not np.allclose(c+a['y'], dest['y']) or \
            not np.allclose(c+a['z'], dest['z']):
        assert False

def testfloat3subfloat():
    dest = np.empty(a.size, dtype=float3)
    float3subfloat(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(a['x']-c, dest['x']) or \
            not np.allclose(a['y']-c, dest['y']) or \
            not np.allclose(a['z']-c, dest['z']):
        assert False

def testfloat3subfloatequal():
    dest = np.copy(a)
    float3subfloatequal(cuda.InOut(dest), c, **size)
    if not np.allclose(a['x']-c, dest['x']) or \
            not np.allclose(a['y']-c, dest['y']) or \
            not np.allclose(a['z']-c, dest['z']):
        assert False

def testfloatsubfloat3():
    dest = np.empty(a.size, dtype=float3)
    floatsubfloat3(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(c-a['x'], dest['x']) or \
            not np.allclose(c-a['y'], dest['y']) or \
            not np.allclose(c-a['z'], dest['z']):
        assert False

def testfloat3mulfloat():
    dest = np.empty(a.size, dtype=float3)
    float3mulfloat(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(a['x']*c, dest['x']) or \
            not np.allclose(a['y']*c, dest['y']) or \
            not np.allclose(a['z']*c, dest['z']):
        assert False

def testfloat3mulfloatequal():
    dest = np.copy(a)
    float3mulfloatequal(cuda.InOut(dest), c, **size)
    if not np.allclose(a['x']*c, dest['x']) or \
            not np.allclose(a['y']*c, dest['y']) or \
            not np.allclose(a['z']*c, dest['z']):
        assert False

def testfloatmulfloat3():
    dest = np.empty(a.size, dtype=float3)
    floatmulfloat3(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(c*a['x'], dest['x']) or \
            not np.allclose(c*a['y'], dest['y']) or \
            not np.allclose(c*a['z'], dest['z']):
        assert False

def testfloat3divfloat():
    dest = np.empty(a.size, dtype=float3)
    float3divfloat(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(a['x']/c, dest['x']) or \
            not np.allclose(a['y']/c, dest['y']) or \
            not np.allclose(a['z']/c, dest['z']):
        assert False

def testfloat3divfloatequal():
    dest = np.copy(a)
    float3divfloatequal(cuda.InOut(dest), c, **size)
    if not np.allclose(a['x']/c, dest['x']) or \
            not np.allclose(a['y']/c, dest['y']) or \
            not np.allclose(a['z']/c, dest['z']):
        assert False

def testfloatdivfloat3():
    dest = np.empty(a.size, dtype=float3)
    floatdivfloat3(cuda.In(a), c, cuda.Out(dest), **size)
    if not np.allclose(c/a['x'], dest['x']) or \
            not np.allclose(c/a['y'], dest['y']) or \
            not np.allclose(c/a['z'], dest['z']):
        assert false

def testdot():
    dest = np.empty(a.size, dtype=np.float32)
    dot(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)
    if not np.allclose(a['x']*b['x'] + a['y']*b['y'] + a['z']*b['z'], dest):
        assert False

def testcross():
    dest = np.empty(a.size, dtype=float3)
    cross(cuda.In(a), cuda.In(b), cuda.Out(dest), **size)
    for u, v, wdest in zip(a,b,dest):
        w = np.cross((u['x'], u['y'], u['z']),(v['x'],v['y'],v['z']))
        if not np.allclose(wdest['x'], w[0]) or \
                not np.allclose(wdest['y'], w[1]) or \
                not np.allclose(wdest['z'], w[2]):
            print(w)
            print(wdest)
            assert False

def testnorm():
    dest = np.empty(a.size, dtype=np.float32)
    norm(cuda.In(a), cuda.Out(dest), **size)

    for i in range(len(dest)):
        if not np.allclose(np.linalg.norm((a['x'][i],a['y'][i],a['z'][i])), dest[i]):
            assert False

def testminusfloat3():
    dest = np.empty(a.size, dtype=float3)
    minusfloat3(cuda.In(a), cuda.Out(dest), **size)
    if not np.allclose(-a['x'], dest['x']) or \
            not np.allclose(-a['y'], dest['y']) or \
            not np.allclose(-a['z'], dest['z']):
        assert False
