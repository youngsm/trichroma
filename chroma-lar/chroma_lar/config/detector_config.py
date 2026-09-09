"""
Default configuration dictionary for detector configurations.

This module provides a default configuration dictionary for detector simulations.
"""

import numpy as np
import yaml
from chroma_lar.database import Database
import os

def get_config():
    """
    Get the default configuration dictionary.
    
    Returns
    -------
    dict
        Default configuration as a dictionary
    """

    # load ./detector_config.yaml
    with open(os.path.join(os.path.dirname(__file__), 'detector_config.yaml'), 'r') as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)


    db = Database('chroma_lar.geometry.custom_optics')

    default_config = {
        # Active volume dimensions
        "active_dimensions": yaml_config["TPC"]["active_volume"],
        # Cavity parameters
        "cavity_scale": 1.5,
        "include_cavity": True,
        # PMT parameters
        "include_pmts": True,
        "pmt_spacing": yaml_config["PMT"]["sensor_spacing"],  # mm
        "pmt_gap": yaml_config["PMT"]["gap_pmt_active"],  # mm
        "pmt_nsteps": 20,
        "pmt_diameter_in": 8,  # inches
        "pmt_photocathode_surface": db.perfect_pmt_photocathode,
        "pmt_back_surface": db.glossy_surface,
        # Wire plane parameters
        "include_wires": True,
        "wire_diameter": 0.15,  # mm
        "wire_pitch": 3.0,  # mm
        "wire_angles": [np.pi / 2, np.pi / 3, -np.pi / 3],  # U, V, Y planes
        "wire_offsets": [0.0, -3.0, -6.0],  # mm
        "wire_nsteps": 32,
        "wire_inner_material": db.steel_material,
        "wire_surface": db.steel_surface,
        # Cathode parameters
        "include_cathode": True,
        "cathode_thickness": 6.0,  # mm
        "cathode_inner_material": db.steel_material,
        "cathode_surface": db.steel_surface,
        # general material parameters
        "default_optics": db,
        "target_material": db.lar,  # what everything is submerged in
        # the surface of the cube!
        "active_surface": db.steel_surface,
        # Other parameters
        "include_active": True,
    }
    
    return default_config 