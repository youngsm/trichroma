RHEL/Centos/Scientific Linux 6 Installation
===========================================

Chroma only supports RHEL-derived distributions if they are based on version 6 or later.  The packages included with RHEL 5 are too old to run Chroma.

Step 1: Install packages with Yum package manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As the root user, run the following commands::

    yum groupinstall "Development tools"
    yum install python-matplotlib python-devel uuid-devel lapack-devel atlas-devel \
        mercurial git subversion mesa-libGLU-devel freeglut-devel SDL-devel gtk2-devel \
        libXpm-devel libXft-devel libXext-devel libXlibX11-devel expat-devel bzip2-devel \
        libXt-devel
    easy_install virtualenv


Step 2: Disable the Nouveau Graphics Driver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
CUDA requires the use of the official NVIDIA graphics driver, rather
than the open source Nouveau driver that is included with RHEL. 

Edit /etc/grub.conf and add ``rdblacklist=nouveau`` to the end of the kernel line::

    kernel /vmlinuz-2.6.32-279.11.1.el6.x86_64 ... rdblacklist=nouveau

Create the file /etc/modprobe.d/blacklist-nouveau.conf with the line::

    blacklist nouveau

Step 3: Install the CUDA Driver and Toolkit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The NVIDIA driver can be installed by going to the `CUDA Downloads
<https://developer.nvidia.com/cuda-downloads>`_ and downloading the rpm package
corresponding to your operating system, and then following the installation instructions `here <http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html#package-manager-installation>`_. This single package includes a current
NVIDIA driver, the CUDA compiler toolkit, and sample programs.

Drop to a console terminal by pressing `CTRL+ALT+F1` and then enter::

    su -
    init 3

Login again at the prompt and enter the following commands::

    su -
    cd /home/user/Downloads
    chmod +x cuda_5.0.35_linux_64_ubuntu11.10-1.run
    .cuda_5.0.35_linux_64_ubuntu11.10-1.run

During the installation you can pick all the default options.

Once installed, you can ensure the CUDA compiler and libraries are in your path by adding the following lines to your bash login script (usually $HOME/.bashrc)::

  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

.. warning:: Non-bash shells will need to be adjusted appropriately.  If you are using a 32-bit distribution, then lib64/ should be changed to lib/.

Step 4: Continue to Common Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rest of the installation process is described in :ref:`common-install`.
