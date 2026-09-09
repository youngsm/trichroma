from setuptools import setup, find_packages

setup(
    name = 'Chroma',
    version = '0.5',
    packages = find_packages(),
    include_package_data=True,
    package_data={
        'chroma': ['cuda/*.cu', 'cuda/*.h'],
        'chroma.triton.webgpu': ['assets/*.html', 'assets/*.js', 'assets/*.wgsl', 'assets/*.css'],
    },

    scripts = ['bin/chroma-sim', 'bin/chroma-cam',
               'bin/chroma-geo', 'bin/chroma-bvh',
               'bin/chroma-server'],
    setup_requires = [],
    install_requires = ['uncertainties','pyzmq', 'pycuda','pytools==2022.1.2',
                        'numpy>=1.6', 'pygame', 'nose', 'sphinx'],
    #test_suite = 'nose.collector',
    
)
