Mac OS X Installation
=====================

Most Mac systems lack the GPU required to run Chroma, with the notable exception of the current 15" MacBook Pro (both Retina and Standard) models, which use an NVIDIA GeForce GT 650M GPU.  These instructions have been tested on OS X 10.8, which ships with the above systems.

.. warning: We have only tested Chroma on the 15" MacBook Pro with Retina display and 1 GB of GPU memory.  Models with 512 MB of video memory may have difficulty running Chroma depending on how much video memory is used by the driver and GUI.

Step 1: Install Xcode
^^^^^^^^^^^^^^^^^^^^^

Xcode can be `installed from the Mac App Store <http://itunes.apple.com/us/app/xcode/id497799835?ls=1&mt=12>`_.  Once installed, it is important to start Xcode and accept the license agreement.  Once started, open the Preferences window and go to the Downloads pane.  There should be a "Command Line Tools" component listed.  Click the "Install" button next to it if it is not already listed as "Installed."

Step 2: Install XQuartz
^^^^^^^^^^^^^^^^^^^^^^^

XQuartz is the OS X port of the X.Org X Window system, and is a prerequisite for some packages used by Chroma.  It is much more up to date than the X11.app that has been shipped with OS X in the past.  Download XQuartz from `here <http://xquartz.macosforge.org/landing/>`_ and install it.

Step 3: Install CUDA
^^^^^^^^^^^^^^^^^^^^

CUDA on OS X requires a special driver.  It can be downloaded from the `CUDA Downloads Page <https://developer.nvidia.com/cuda-downloads>`_.  The package also includes the CUDA compiler and sample programs, installed to /Developer/NVIDIA/CUDA-5.0.

Once installed, you can ensure the CUDA compiler and libraries are in your path by adding the following lines to your bash login script ($HOME/.bashrc)::

  export PATH=/Developer/NVIDIA/CUDA-5.0/bin:$PATH
  export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-5.0/lib/:$DYLD_LIBRARY_PATH


Step 4: Install MacPorts
^^^^^^^^^^^^^^^^^^^^^^^^

There are several packaging systems that simplify the installation of Open Source software on the Mac.  We have tested and recommend `MacPorts <http://www.macports.org/>`_, but other systems like `Fink <http://www.finkproject.org>`_ and `Homebrew <http://mxcl.github.com/homebrew/>`_ should also work if you install the same packages.  We will assume MacPorts below.

Follow the `MacPorts installation instructions <http://www.macports.org/install.php>`_.  Once installed, open a terminal and run the following command::

    sudo port install py27-matplotlib mercurial py27-game py27-virtualenv Xft2 xpm
    sudo port select virtualenv virtualenv27

Step 5: Continue to Common Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rest of the installation process is described in :ref:`common-install`.