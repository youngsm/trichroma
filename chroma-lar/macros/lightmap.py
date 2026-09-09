import argparse
import math
import sys
import time
from typing import List, Tuple

import chroma
import numpy as np
from chroma.event import Photons
from chroma.loader import load_bvh
from chroma.sample import uniform_sphere
from chroma.sim import Simulation
from tqdm import tqdm

from geometry.builder import build_detector_from_yaml
from generator.photons import create_photon_bomb
from utils.log import logger
from utils.output import H5Logger, print_table


def __configure__(db):
    """Modify fields in the database here"""
    db.seed = None
    db.dry = False
    db.n_photons = 100_000
    db.single_channel = False
    db.output_file = "test.h5"
    db.wavelength = 175
    db.positions_path = (
        "/home/sam/sw/chroma-lxe/data/lightmap_points_2.5mm_orthofill.npy"
    )
    db.config_file = "/home/sam/sw/chroma-lxe/geometry/config/detector.yaml"
    
    db.chroma_g4_processes = 0
    db.chroma_keep_hits = not db.single_channel
    db.chroma_keep_flat_hits = True
    db.chroma_photon_tracking = db.dry
    db.chroma_daq = db.dry
    db.chroma_keep_photons_beg = db.dry
    db.chroma_keep_photons_end = db.dry

def __define_geometry__(db):
    """Returns a chroma Detector or Geometry"""
    geometry = build_detector_from_yaml(db.config_file, flat=True)
    geometry.bvh = load_bvh(geometry, read_bvh_cache=True)
    db.geometry = geometry
    return geometry

def __event_generator__(db):
    """A generator to yield chroma Events (or something a chroma Simulation can
    convert to a chroma Event)."""
    yield from (
        create_photon_bomb(db.n_photons, db.wavelength, position)
        for position in db.photon_positions
    )


def __simulation_start__(db):
    """Called at the start of the event loop"""
    db.photon_positions = np.load(db.positions_path)[:100]
    db.num_events = len(db.photon_positions)
    db.n_channels = db.geometry.num_channels()
    
    # create variable labels
    variables = ["posX", "posY", "posZ", "n", "detected", "pte"]
    if not db.single_channel:
        zfill_width = int(math.log10(db.n_channels)) + 1
        for i in range(db.n_channels):
            channel_id = str(i).zfill(zfill_width)
            variables += [f"ch{channel_id}_detected", f"ch{channel_id}_pte"]
    variables += ["time_spent"]
    db.writer = H5Logger(db.output_file, variables)

    db.event_idx = 0
    db.total_detected = 0
    db.total_pte = 0
    db.total_time = 0
    db.start_time = time.time()

def __process_event__(db, ev):
    """Called for each generated event"""
    output = {}
    position = db.photon_positions[db.event_idx]
    output["posX"] = position[0]
    output["posY"] = position[1]
    output["posZ"] = position[2]
    output["n"] = db.n_photons

    detected = len(ev.flat_hits)
    output["detected"] = detected
    output["pte"] = detected / db.n_photons

    if not db.single_channel:
        zfill_width = int(math.log10(db.n_channels)) + 1
        for c in range(db.n_channels):
            channel_id = str(c).zfill(zfill_width)
            hits = ev.hits
            channel_detected = len(hits.get(c, []))
            output[f"ch{channel_id}_detected"] = channel_detected
            output[f"ch{channel_id}_pte"] = channel_detected / db.n_photons

    ev_time = time.time() - db.start_time
    output["time_spent"] = ev_time
    db.writer.write(**output)

    db.total_detected += output["detected"]
    db.total_pte += output["pte"]
    db.start_time += ev_time
    db.total_time += ev_time
    db.event_idx += 1

def __simulation_end__(db):
    """Called at the end of the event loop"""
    db.writer.close()

    n_positions = len(db.photon_positions)
    results = dict(
        output_path=db.output_file,
        n_positions=n_positions,
        n_photons_per_position=db.n_photons,
        n_detected=db.total_detected,
        n_detected_per_position=db.total_detected / n_positions,
        avg_pte_per_position=db.total_pte / n_positions,
        total_time=db.total_time,
        sec_per_position=db.total_time / n_positions,
        positions_per_sec=n_positions / db.total_time,
        photons_per_sec=n_positions * db.n_photons / db.total_time,
    )
    print_table(**results)