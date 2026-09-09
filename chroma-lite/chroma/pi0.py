import numpy as np

_kg_per_MeV = 1.782661758e-36/1e-6
_pi0_mass = 134.9766*_kg_per_MeV

def rocket_to_lab(energy, momentum, v):
    """
    Return the energy and momentum of a particle in the lab frame from its
    energy and momentum in a rocket frame traveling at a velocity `v` with
    respect to the lab frame.

    Args:
        - energy: float, kg
            The energy of the particle in the rocket frame.
        - momentum: array, kg
            The momentum of the particle in the rocket frame.
        - v: array, units of c
            The velocity of the rocket frame as seen in the lab frame.
    """
    e0 = float(energy)
    p0 = np.asarray(momentum, float)
    v = np.asarray(v, float)

    assert e0**2 - p0.dot(p0) >= -1.0e-70

    g = 1.0/np.sqrt(1.0-v.dot(v))

    x = np.dot(p0, v)/np.linalg.norm(v)
    p = p0 + ((g-1.0)*x + g*np.linalg.norm(v)*e0)*v/np.linalg.norm(v)
    e = np.sqrt(e0**2 - p0.dot(p0) + p.dot(p))

    return e, p

def pi0_decay(energy, direction, theta, phi):
    """
    Return the energy and directions for two photons produced from a pi0
    decay in the lab frame given that one of the photons decayed at polar
    angles `theta` and `phi` in the pi0 rest frame.

    Args:
        - energy: float, MeV
            The total energy of the pi0.
        - direction: array
            The direction of the pi0's velocity in the lab frame.

    Returns:
        - e1: float, MeV
            The energy of the first photon in the lab frame.
        - v1: array,
            The direction of the first photon in the lab frame.
        - e2: float, MeV
            The energy of the second photon in the lab frame.
        - v2: array,
            The direction of the second photon in the lab frame.
    """
    direction = np.asarray(direction)/np.linalg.norm(direction)
    pi0_e = float(energy)*_kg_per_MeV
    pi0_p = np.sqrt(pi0_e**2-_pi0_mass**2)*direction
    pi0_v = pi0_p/pi0_e

    photon_e0 = _pi0_mass/2.0
    photon_p0 = photon_e0*np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])

    e1, p1 = rocket_to_lab(photon_e0, photon_p0, pi0_v)
    v1 = p1/np.linalg.norm(p1)
    e2, p2 = rocket_to_lab(photon_e0, -photon_p0, pi0_v)
    v2 = p2/np.linalg.norm(p2)

    return (e1/_kg_per_MeV, v1), (e2/_kg_per_MeV, v2)
