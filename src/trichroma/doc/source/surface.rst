Surface Models
==============

Chroma includes a variety of surface models, which control the interaction of photons with objects. The ``Surface`` class contains many (typically wavelength-dependent) surface parameters, some of which apply to each surface model.

To select a particular model for a surface, set ``surface.model = MODEL`` where ``MODEL`` is one of ``{ SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS }``.

``SURFACE_DEFAULT``
-------------------

The default surface model approximates real surfaces by combining a diffuse and specular reflection component. Surfaces may also be absorbing and detecting. The wavelength-dependent probabilities for detection, absorption, diffuse reflection, and specular reflection are provided in uniformly-spaced float arrays of length ``n``, and should sum to unity. The corresponding wavelengths for these arrays are computed from the first (``wavelength0``) and the step size (``step``).

Parameters::

    float *detect
    float *absorb
    float *reflect_diffuse
    float *reflect_specular
    unsigned int n
    float step
    float wavelength0

``SURFACE_COMPLEX``
-------------------

This surface model uses a complex index of refraction (``eta``, ``k``) to compute transmission, (specular) reflection, and absorption probabilities, taking into account the incoming photon polarization and surface thickness (``thickness``). Surfaces may also be photon detectors, but detection is conditional on absorption so ``detect`` should be normalized independently. Reflection may be either specular or diffuse, with probabilities given in ``reflect_specular`` and ``reflect_diffuse``; these should sum to 1 at each wavelength. By default surfaces are not transmissive (use ``bool transmissive`` to set), and transmitted photons are absorbed.

Parameters::

    float *detect
    float *reflect_specular
    float *reflect_diffuse
    float *eta
    float *k
    unsigned int n
    float step
    float wavelength0
    float thickness
    bool transmissive

``SURFACE_WLS``
---------------

This surface model is used for wavelength-shifting surfaces. Surfaces may absorb, specularly reflect, diffusely reflect, or transmit photons; ``1 - absorb - reflect_diffuse - reflect_specular`` gives the transmission probability. If a photon is absorbed, it may be reemitted with a probability given in ``float *reemit``. The reemission spectrum CDF is defined in ``float* reemission_cdf``. The CDF must start at 0 and end at 1. This model does not enforce conservation of energy, and cannot reemit multiple photons!

Parameters::

    float *absorb
    float *reflect_specular
    float *reflect_diffuse
    float *reemit
    float *reemission_cdf
    unsigned int n
    float step
    float wavelength0

