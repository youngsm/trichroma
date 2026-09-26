.. chroma documentation master file, created by
   sphinx-quickstart on Sat Sep  3 12:36:34 2011.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Chroma: Ultra-fast Photon Monte Carlo
=====================================

Chroma is a high performance optical photon simulation for particle physics detectors.  It tracks individual photons passing through a triangle-mesh detector geometry, simulating standard physics processes like diffuse and specular reflections, refraction, Rayleigh scattering and absorption.  

With the assistance of a CUDA-enabled GPU, Chroma can propagate 2.5 million photons per second in a detector with 29,000 photomultiplier tubes.  This is 200x faster than the same simulation with GEANT4.

Development
-----------

Chroma is under heavy development.  Tagged releases will happen soon,
but in the meantime we encourage people to obtain the code directly
from the Mercurial repository hosted at Bitbucket:

  http://bitbucket.org/chroma/chroma

For questions, ideas, and discussion, join the Chroma development mailing list:

  http://groups.google.com/group/chroma-sim

For bug reports, please create a Bitbucket account and use the Bitbucket issue tracker:

  https://bitbucket.org/chroma/chroma/issues?status=new&status=open


Documentation
-------------

The core concepts are described in the :download:`Chroma whitepaper
<chroma.pdf>`.

The software documentation is evolving rapidly, so
many of the following sections are incomplete at the moment.
Subscribe to the mailing list for update announcements!

.. toctree::
   :maxdepth: 2

   install/overview
   tour
   geometry
   surface
   detector
   render
   simulation
   likelihood
   examples

Authors
-------

Chroma is developed by Anthony LaTorre and `Stan Seibert
<mailto:sseibert@hep.upenn.edu>`_. Chroma contains some material
properties found in the `WCSim application
<https://wiki.bnl.gov/dusel/index.php/WCSim>`_.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

