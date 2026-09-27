from setuptools import setup, find_packages, Extension
import subprocess
import os

setup(
    name = 'Chroma',
    version = '0.5',
    packages = find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    package_data={
        'chroma': ['cuda/*.cu', 'cuda/*.h'],
    },

    scripts = ['bin/chroma-sim', 'bin/chroma-cam',
               'bin/chroma-geo', 'bin/chroma-bvh',
               'bin/chroma-server'],
    setup_requires = [],
    install_requires = ['numpy>=1.6'],
    extras_require = {
        # The original PyCUDA backend (CHROMA_BACKEND=cuda).
        'cuda': ['pycuda', 'pytools==2022.1.2'],
        # The event viewer (chroma.camera), chroma-server and the histogram helpers.
        'viz': ['pygame', 'pyzmq', 'uncertainties', 'matplotlib'],
        'test': ['pytest'],
    },
    #test_suite = 'nose.collector',
    
)
