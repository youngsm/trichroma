import os
from math import sin, cos, sqrt

import numpy as np
import numpy.linalg

from chroma.make import sphere
from chroma.stl import mesh_from_stl
from chroma.geometry import Solid
from chroma.detector import Detector
from chroma.transform import rotate, make_rotation_matrix, normalize

from chroma.demo.pmt import build_8inch_pmt_with_lc
from chroma.demo.optics import water, black_surface

from chroma.demo.checkerboard import build_checkerboard_scene as checkerboard_scene
from chroma.log import logger

def spherical_spiral(radius, spacing):
    '''Returns iterator generating points on a spiral wrapping the
    surface of a sphere.  Points should be approximately equidistiant
    along the spiral.'''
    dl = spacing / radius
    t = 0.0
    a = np.pi / dl

    while t < np.pi:
        yield np.array([sin(t) * sin(a*t), sin(t) * cos(a*t), cos(t)])*radius
        dt = dl / sqrt(1 + a**2 * sin(t) ** 2)
        t += dt

def detector(pmt_radius=14000.0, sphere_radius=14500.0, spiral_step=350.0):
    pmt = build_8inch_pmt_with_lc()
    geo = Detector(water)

    geo.add_solid(Solid(sphere(sphere_radius,nsteps=200), 
                        water, water, 
                        surface=black_surface,
                        color=0xBBFFFFFF))

    for position in spherical_spiral(pmt_radius, spiral_step):
        direction = -normalize(position)

        # Orient PMT that starts facing Y axis
        y_axis = np.array((0.0,1.0,0.0))
        axis = np.cross(direction, y_axis)
        angle = np.arccos(np.dot(y_axis, direction))
        rotation = make_rotation_matrix(angle, axis)

        # Place PMT (note that position is front face of PMT)
        geo.add_pmt(pmt, rotation, position)
        
    
    time_rms = 1.5 # ns
    charge_mean = 1.0
    charge_rms = 0.1 # Don't I wish!
    
    geo.set_time_dist_gaussian(time_rms, -5 * time_rms, 5*time_rms)
    geo.set_charge_dist_gaussian(charge_mean, charge_rms, 0.0, charge_mean + 5*charge_rms)

    logger.info('Demo detector: %d PMTs' % geo.num_channels())
    logger.info('               %1.1f ns time RMS' % time_rms)
    logger.info('               %1.1f%% charge RMS' % (100.0*charge_rms/charge_mean))
    return geo

def tiny():
    return detector(2000.0, 2500.0, 700.0)

