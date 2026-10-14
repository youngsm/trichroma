import os
from urllib import urlretrieve
import subprocess
import logging
import sys
import traceback

def exists(name):
    "Returns True if `name` is found in $PATH."
    for path in os.environ['PATH'].split(os.pathsep):
        if os.path.exists(os.path.join(path,name)):
            return True

    return False

def log_call(args, **kwargs):
    "Logs a subprocess call."
    logging.info(' '.join(args))
    subprocess.check_call(args, **kwargs)

def download_untar(url):
    """Download and untar the url in the current working directory. Returns
    the absolute path of the folder untarred to."""
    filename = url.split('/')[-1]
    logging.info('downloading %s to %s' % (url, filename))
    urlretrieve(url,filename)
    dir = filename.rstrip('.tar.gz')

    try:
        os.mkdir(dir)
    except OSError:
        pass

    log_call(['tar','xf',filename,'--strip=1','-C',dir])

    return os.path.abspath(dir)

def check_output(args):
    "Sort of like subprocess.check_output in python 2.7"
    p = subprocess.Popen(args,stdout=subprocess.PIPE)
    output, err_msg = p.communicate()

    if p.poll():
        raise subprocess.CalledProcessError(err_msg)

    return output

# urls for various packages
get_pip_url = 'https://raw.github.com/pypa/pip/master/contrib/get-pip.py'
root_url = 'ftp://root.cern.ch/root/root_v5.32.04.source.tar.gz'
geant4_url = 'http://geant4.cern.ch/support/source/geant4.9.5.p01.tar.gz'
cmake_url = 'http://www.cmake.org/files/v2.8/cmake-2.8.10.1.tar.gz'

# list of package dependencies on yum
yum_pkgs = ['boost-python', 'boost-devel', 'python-devel', 'python-matplotlib', 'mesa-libGLU-devel', 'freeglut', 'uuid-devel', 'atlas-devel', 'lapack-devel', 'SDL-devel', 'gtk2-devel', 'libXpm-devel', 'libXft-devel', 'libXext-devel', 'libX11-devel', 'expat-devel', 'mercurial']

# list of package dependencies on apt-get
apt_pkgs = ['build-essential', 'xorg-dev', 'python-dev', 'python-virtualenv', 'python-numpy', 'python-pygame', 'libglu1-mesa-dev', 'glutg3-dev', 'cmake', 'uuid-dev', 'liblapack-dev', 'mercurial', 'git', 'subversion', 'python-matplotlib', 'libboost-all-dev', 'libatlas-base-dev']

if __name__ == '__main__':
    import optparse
    from os.path import join
    import shlex

    parser = optparse.OptionParser('%prog ENV_DIR (i.e. /home/user/chroma)')
    parser.add_option('-j', dest='j', default=1,
                      help='number of cores to build with')
    options, args = parser.parse_args()

    if not args:
        sys.exit(parser.format_help())

    # require these to import pycuda.driver near the end of the
    # install script
    required_ldpaths = ['/usr/local/cuda/lib64','/usr/local/cuda/lib']

    try:
        ld_library_path = os.environ['LD_LIBRARY_PATH'].split(os.pathsep)
    except KeyError:
        ld_library_path = []

    if not set(ld_library_path).issuperset(set(required_ldpaths)):
        print 'Run the following line in your interpreter:'
        print '\n    export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH\n' % \
            os.pathsep.join(required_ldpaths)
        print 'and then restart the script.'
        sys.exit(1)

    logging.basicConfig(filename='setup.log',level=logging.DEBUG,
                        format='%(asctime)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(ch)

    envdir = args[0]

    if not exists('pip'):
        # install pip
        filename, headers = urlretrieve(get_pip_url)
        log_call(['sudo','python',filename])

    if exists('apt-get'):
        # install packages with apt-get
        log_call(['sudo','apt-get','install'] + apt_pkgs)
    elif exists('yum'):
        # install packages with yum
        log_call(['sudo','yum','install'] + yum_pkgs)
    else:
        sys.exit("script requires `yum` or `apt-get`")

    if not exists('virtualenv'):
        # install virtualenv
        log_call(['sudo','pip','install','virtualenv'])

    if not os.path.exists(envdir):
        # create virtualenv directory
        log_call(['virtualenv',envdir])

    # activating virtual environment!
    activate_this = join(envdir,'bin','activate_this.py')
    execfile(activate_this,dict(__file__=activate_this))

    srcdir = join(envdir,'src')

    if not os.path.exists(srcdir):
        os.mkdir(srcdir)

    # ROOT installation
    os.chdir(srcdir)
    dir = download_untar(root_url)
    os.chdir(dir)
    log_call(['./configure', "--enable-minuit2","--enable-roofit"])
    log_call(['make','-j',options.j])
    # note: DON'T make install, it just causes problems

    with open(join(envdir,'bin','activate'),'a') as f:
        f.write('source %s\n' % join(dir,'bin/thisroot.sh'))

    # install our own version of cmake since default versions seem to have
    # trouble installing geant4
    os.chdir(srcdir)
    dir = download_untar(cmake_url)
    os.chdir(dir)
    log_call(['./bootstrap','--prefix=%s' % envdir])
    log_call(['make'])
    log_call(['make','install'])

    # GEANT4 installation
    os.chdir(srcdir)
    idir = download_untar(geant4_url)
    # geant4 requires a separate build directory
    bdir = idir.rstrip('/') + '-build'

    try:
        os.mkdir(bdir)
    except OSError:
        pass

    os.chdir(bdir)

    log_call(['cmake',idir,'-DCMAKE_INSTALL_PREFIX=%s' % envdir,
              '-DGEANT4_INSTALL_DATA=ON'])
    log_call(['make','install'])

    # need site-packages location for g4py install
    from distutils.sysconfig import get_python_lib

    try:
        import g4py
    except ImportError:
        print 'installing g4py...'
        # install g4py
        os.chdir(srcdir)
        if not os.path.exists('g4py'):
            log_call(['hg','clone','https://bitbucket.org/seibert/g4py#geant4.9.5.p01'])
        os.chdir('g4py')
        log_call(['./configure','linux64','--prefix=%s' % envdir,
                  '--with-g4-incdir=%s' % join(envdir,'include/Geant4'),
                  '--with-g4-libdir=%s' % join(envdir,'lib'),
                  '--libdir=%s' % get_python_lib()])
        log_call(['make','install'])

    with open(join(envdir,'bin','activate'),'a') as f:
        f.write('\n\n')
        f.write('export PATH=/usr/local/cuda/bin:$PATH\n')
        f.write('export LD_LIBRARY_PATH=/usr/local/cuda/lib64:'
                '/usr/local/cuda/lib:$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH\n')
        f.write('source $VIRTUAL_ENV/bin/geant4.sh\n')

    # need nvcc on path for pycuda install
    os.environ['PATH'] += ':/usr/local/cuda/bin'

    aksetup_filename = join(os.environ['HOME'],'.aksetup-defaults.py')

    with open(aksetup_filename,'w') as f:
        f.write("BOOST_INC_DIR = ['/usr/include/boost']\n"
                "BOOST_LIB_DIR = ['/usr/lib64']\n")
        if exists('yum'):
            # from bitbucket.org/chroma/chroma/issue/3/
            f.write("BOOST_PYTHON_LIBNAME = ['boost_python-mt']\n")
        else:
            # from install docs at chroma.bitbucket.org
            f.write("BOOST_PYTHON_LIBNAME = ['boost_python-mt-py27']\n")

    log_call(['pip','install','numpy'])
    log_call(['pip','install','-U','distribute'])
    log_call(['pip','install','pyublas'])
    log_call(['pip','install','matplotlib'])

    if not os.path.exists(join(envdir,'local')):
        os.mkdir(join(envdir,'local'))
        log_call(['ln','-s',join(envdir,'lib'),join(envdir,'local/lib')])

    # needed by chroma/setup.py, otherwise won't include /include dirs
    os.environ['VIRTUAL_ENV'] = envdir

    log_call(['pip','install','-e',
              'hg+http://bitbucket.org/chroma/chroma#egg=Chroma'])

    # determine the card with the highest compute capability
    # and set it as the default chroma device
    import pycuda.driver as cuda

    cuda.init()
    devices = [cuda.Device(i) for i in range(cuda.Device.count())]
    default_device = sorted([(device.compute_capability(),i) for i, device \
                                 in enumerate(devices)])[-1][1]

    logging.info('setting default nvidia device = %i' % default_device)

    with open(join(envdir,'bin','activate'),'a') as f:
        f.write('export CUDA_DEVICE=%i\n' % default_device)
