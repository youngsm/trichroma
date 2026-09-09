"""
Configuration for the pixel LArTPC detector.

Pixel pads (gold on FR-4) on ±X faces, PMTs on ±Y walls, cathode at X = 0.
"""

import os
import yaml
from chroma_lar.database import Database


def get_config():
    """Return the pixel detector configuration dictionary."""

    with open(
        os.path.join(os.path.dirname(__file__), "detector_config_pixel.yaml"), "r"
    ) as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)

    db = Database("chroma_lar.geometry.custom_optics")

    return {
        "detector_type": yaml_config.get("detector_type", "pixel"),
        # Active volume
        "active_dimensions": yaml_config["TPC"]["active_volume"],
        # Cavity
        "cavity_scale": 1.5,
        "include_cavity": True,
        # PMTs (±Y walls)
        "include_pmts": True,
        "pmt_n_x_half": yaml_config["PMT"]["n_x_half"],
        "pmt_n_z": yaml_config["PMT"]["n_z"],
        "pmt_gap": yaml_config["PMT"]["gap_pmt_active"],
        "pmt_wall_margin": yaml_config["PMT"]["wall_margin"],  # defaults to pmt_radius
        "pmt_nsteps": 20,
        "pmt_diameter_in": yaml_config["PMT"]["diameter_in"],
        "pmt_photocathode_surface": db.perfect_pmt_photocathode,
        "pmt_back_surface": db.glossy_surface,
        "pmt_glass_material": db.glass,
        # Pixels (±X faces)
        "include_pixels": True,
        "pixel_pitch": yaml_config["pixel"]["pitch"],
        "pixel_pad_size": yaml_config["pixel"]["pad_size"],
        "pixel_chamfer_radius": yaml_config["pixel"]["chamfer_radius"],
        "n_pixels_y": yaml_config["pixel"]["n_pixels_y"],
        "n_pixels_z": yaml_config["pixel"]["n_pixels_z"],
        "pixel_surface": db.gold,
        "pcb_surface": db.fr4,
        "shield_surface": db.copper,
        "pixel_simplified": True,
        # Cathode
        "include_cathode": True,
        "cathode_thickness": yaml_config["TPC"]["cathode_thickness"],
        "cathode_inner_material": db.steel_material,
        "cathode_surface": db.polished_steel_surface,
        # Materials
        "default_optics": db,
        "target_material": db.lar,
        "active_surface": db.polished_steel_surface,
        "include_active": True,
    }
