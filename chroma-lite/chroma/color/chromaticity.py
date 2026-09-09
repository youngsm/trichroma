import numpy as np
from os.path import realpath, dirname
from chroma.tools import read_csv

color_map = read_csv(dirname(realpath(__file__)) + '/ciexyz64_1.csv')

def map_wavelength(wavelength):
    r = np.interp(wavelength, color_map[:,0], color_map[:,1])
    g = np.interp(wavelength, color_map[:,0], color_map[:,2])
    b = np.interp(wavelength, color_map[:,0], color_map[:,3])

    if np.iterable(wavelength):
        rgb = np.empty((len(wavelength),3))

        rgb[:,0] = r
        rgb[:,1] = g
        rgb[:,2] = b

        return rgb
    else:
        return np.array([r,g,b])
