Installation
============

Chroma development tends to live on the bleeding-edge.  Installation
of Chroma requires a more significant hardware and software investment
than other packages, but we think the rewards are worth it!

.. _hardware-requirements:

Hardware Requirements
---------------------

At a minimum, Chroma requires:

* An x86 or x86-64 CPU.
* A NVIDIA GPU that supports CUDA compute capability 2.0 or later.

We highly recommend that you run Chroma with:

* An x86-64 CPU with at least four cores.
* 8 GB or more of system RAM.
* An NVIDIA GPU that supports CUDA compute capability 2.0 or later,
  and has at least 1 GB of device memory.

Memory requirements on the CPU and GPU scale with the complexity of your
model.  A detector represented with 60.1 million triangles (corresponding to
20,000 detailed photomultipler tubes) requires 2.2 GB of CUDA device memory,
and more than 6 GB of host memory during detector construction.  Chroma can
take advantage of multiple CPU cores to generate Cherenkov light with GEANT4.

.. note:: The Chroma interactive renderer includes optional support for
  the `Space Navigator 3D mouse <http://www.3dconnexion.com/products/spacenavigator.html>`_, which makes it 10x more fun to fly
  through the detector geometry!

OS Specific Prerequisites
-------------------------

First, follow one of the following OS-specific guides to install the system-level prerequisites for Chroma:

.. toctree::
    :maxdepth: 1

    ubuntu
    rhel
    macosx


.. _common-install:

Common Installation Guide
-------------------------

We have tried to streamline the Chroma installation process to be portable to
many platforms.  If you have problems following these instructions, please
`open an issue
<http://bitbucket.org/chroma/chroma/issues?status=new&status=open>`_.


Step 1: Create virtualenv
^^^^^^^^^^^^^^^^^^^^^^^^^

Chroma should never be installed into your system Python directories.  Instead
create a self-contained virtualenv::

  virtualenv --system-site-package chroma_env
  source chroma_env/bin/activate

You only need to delete the chroma_env directory to completely remove Chroma from your system.


Step 2: Install Chroma Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chroma depends on several C and C++ libraries that are not typically included
in the package managers of many platforms.  Using `shrinkwrap
<http://shrinkwrap.rtfd.org>`_, we have automated the installation of these
libraries into the virtualenv, isolating them from the rest of your system::

  # Create configuration file for PyCUDA
  echo -e "import os\nvirtual_env = os.environ['VIRTUAL_ENV']\nBOOST_INC_DIR = [os.path.join(virtual_env, 'include')]\nBOOST_LIB_DIR = [os.path.join(virtual_env, 'lib')]\nBOOST_PYTHON_LIBNAME = ['boost_python']" > ~/.aksetup-defaults.py
  # Search this site for shrinkwrap packages used by Chroma
  export PIP_EXTRA_INDEX_URL=http://mtrr.org/chroma_pkgs/

  # On RHEL/Centos/Scientific Linux ONLY, uncomment and run the following commands
  # pip install -U numpy
  # easy_install pygame

  # This will take a LONG time.
  # If interrupted, run the command again and it will resume where it left off
  pip install chroma_deps

  # Refresh environment variables
  source $VIRTUAL_ENV/bin/activate

Step 3: Install Chroma
^^^^^^^^^^^^^^^^^^^^^^

Now we can checkout a copy of Chroma and install it.  By default, we will put it into the $VIRTUAL_ENV/src directory, but anywhere is fine::

  cd $VIRTUAL_ENV/src
  hg clone https://bitbucket.org/chroma/chroma
  cd chroma
  python setup.py develop

If everything has succeeded, you are ready to move onto the :ref:`tour`!
