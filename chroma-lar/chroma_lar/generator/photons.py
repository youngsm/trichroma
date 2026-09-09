import numpy as np
from chroma.event import Photons

def photon_gen(nphotons=10000,wavelength=450,pos=[0,0,0],dir=None,pol=None):
    if dir is None:
        costheta = np.random.random(nphotons)*2-1
        sintheta = np.sqrt(1-np.square(costheta))
        phi = np.random.random(nphotons)*2*np.pi
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        pdir = np.transpose([sintheta*cosphi, sintheta*sinphi, costheta])
    else:
        pdir = np.tile(dir,(nphotons,1))
            
    if pol is None:
        costheta = np.random.random(nphotons)*2-1
        sintheta = np.sqrt(1-np.square(costheta))
        phi = np.random.random(nphotons)*2*np.pi
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        rand_unit = np.transpose([sintheta*cosphi, sintheta*sinphi, costheta])
        ppol = np.cross(pdir, rand_unit)
        ppol = ppol / np.linalg.norm(ppol, ord=2, axis=1, keepdims=True)
    else:
        ppol = np.tile(pol,(nphotons,1))

    if type(wavelength) is tuple:
        pwavelength = np.random.random(nphotons)*(wavelength[1]-wavelength[0])+wavelength[0]
    else:
        pwavelength = np.tile(wavelength,nphotons)

    photons = Photons(pos=np.tile(pos,(nphotons,1)),
                          dir=pdir, pol=ppol, 
                          wavelengths=pwavelength)
    return photons    