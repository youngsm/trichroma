Ubuntu Installation
===================

Step 1: ``apt-get`` packages with Ubuntu package manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We need to install several system packages::

    sudo apt-get install python-pygame python-matplotlib python-virtualenv \
        build-essential xorg-dev python-dev libglu1-mesa-dev  freeglut3-dev \
        uuid-dev liblapack-dev mercurial git subversion libatlas-base-dev \
        libbz2-dev

Step 2: CUDA Toolkit and Driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CUDA requires the use of the official NVIDIA graphics driver, rather than the
open source Nouveau driver that is included with Ubuntu.  The NVIDIA driver
can be installed by going to the `CUDA Downloads <https://developer.nvidia.com
/cuda-downloads>`_ and downloading the package corresponding to your Ubuntu
version. This single package includes a current NVIDIA driver, the CUDA
compiler toolkit, and sample programs.

.. note:: Although NVIDIA only lists support up to Ubuntu 11.10 in CUDA 5, we have found the package to also work with Ubuntu 12.04 LTS.

To install the NVIDIA drivers, you will need to switch to a text console (Ctrl-Alt-F1) and shut down the X server::

  # This next will kill everything running on your graphical desktop!

  # On Ubuntu 12.04: sudo service lightdm stop 
  sudo service gdm stop 

  chmod +x cuda_5.0.35_linux_64_ubuntu11.10-1.run
  sudo ./cuda_5.0.35_linux_64_ubuntu11.10-1.run
  # Accept the license and pick the defaults

  # On Ubuntu 12.04: sudo service lightdm start
  sudo service gdm start

Once installed, you can ensure the CUDA compiler and libraries are in your path by adding the following lines to your bash login script (usually $HOME/.bashrc)::

  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

.. warning:: Non-bash shells will need to be adjusted appropriately.  If you are using a 32-bit distribution, then lib64/ should be changed to lib/.

Step 3: Continue to Common Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rest of the installation process is described in :ref:`common-install`.