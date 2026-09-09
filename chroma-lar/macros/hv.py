import argparse
import logging
import sys
import time
from typing import Generator

import chroma
import numpy as np
import yaml
from chroma.event import (
    Photons,
    SURFACE_DETECT,
    NO_HIT,
    NAN_ABORT,
    SURFACE_ABSORB,
    BULK_ABSORB,
)
from chroma.loader import load_bvh
from chroma.sim import Simulation
from tqdm import tqdm

from geometry.fiber import M114L01
from geometry.builder import build_detector_from_yaml
from utils.output import H5Logger, print_table

sys.path.append("../geometry")

logging.getLogger("chroma").setLevel(logging.DEBUG)


def __configure__(db):
    """Modify fields in the database here"""

    db.n_photons_per_fiber = 100_000
    db.fiber = M114L01
    db.seed = None
    db.config_file = (
        "/home/sam/sw/chroma-lxe/geometry/config/ea-hv_4_fibers_100mm_extended.yaml"
    )
    db.fiber_positions_file = (
        "/home/sam/sw/chroma-lxe/data/stl/fibers/fiber_positions_100mm_extended.yaml"
    )

    db.chroma_g4_processes = 0
    db.chroma_keep_hits = True
    db.chroma_keep_flat_hits = True
    db.chroma_photon_tracking = True
    db.chroma_particle_tracking = False
    db.chroma_photons_per_batch = 1_000_000
    db.chroma_max_steps = 100
    db.chroma_daq = True
    db.chroma_keep_photons_beg = True
    db.chroma_keep_photons_end = True

def __define_geometry__(db):
    """Returns a chroma Detector or Geometry"""
    geometry = build_detector_from_yaml(db.config_file)
    geometry.bvh = load_bvh(geometry, read_bvh_cache=True)
    db.geometry = geometry
    return geometry

def __event_generator__(db) -> Generator[Photons, None, None]:
    """A generator to yield chroma Events"""
    posdir = yaml.safe_load(open(db.fiber_positions_file, "r"))
    fibers = [
        db.fiber(
            position=posdir[f"fiber_{i}"]["position"],
            direction=posdir[f"fiber_{i}"]["direction"],
        )
        for i in range(4)
    ]
    for i in range(4):
        np.set_printoptions(precision=3)
        # print(fibers[i].rotation_matrix)
    
    batch_size = 100_000
    total_photons = db.n_photons_per_fiber * len(fibers)
    
    while total_photons > 0:
        current_batch = min(batch_size, total_photons)
        photons_per_fiber = current_batch // len(fibers)
        remainder = current_batch % len(fibers)
        
        batch = Photons()
        for i, fiber in enumerate(fibers):
            fiber_photons = photons_per_fiber + (1 if i < remainder else 0)
            if fiber_photons > 0:
                batch += fiber.generate_photons(fiber_photons)
                print('actual direction\n', fiber.direction)
        yield batch
        total_photons -= current_batch

def __simulation_start__(db):
    """Called at the start of the event loop"""
    total_photons = db.n_photons_per_fiber * 4
    db.num_events = np.ceil(total_photons / 100_000)
    db.start_time = time.time()
    db.total_detected = 0
    db.total_photons = 0

def __process_event__(db, ev):
    """Called for each generated event"""
    detected = (ev.photons_end.flags & SURFACE_DETECT).astype(bool)
    db.total_detected += detected.sum()
    db.total_photons += len(detected)
    print_stats(ev)


def __simulation_end__(db):
    """Called at the end of the event loop"""
    total_time = time.time() - db.start_time
    results = dict(
        n_photons=db.total_photons,
        n_detected=db.total_detected,
        pte=db.total_detected / db.total_photons,
        total_time=total_time,
        output=db.output
    )
    print_table(**results)

def count_test(flags, test, none_of=None):
    if none_of is not None:
        has_test = np.bitwise_and(flags, test) == test
        has_none_of = np.bitwise_and(flags, none_of) == 0
        return np.count_nonzero(np.logical_and(has_test, has_none_of))
    else:
        return np.count_nonzero(np.bitwise_and(flags, test) == test)

def print_stats(ev):
    detected = (ev.photons_end.flags & SURFACE_DETECT).astype(bool)
    photon_detection_efficiency = detected.sum() / len(detected)
    print("in event loop")
    print("# detected", detected.sum(), "# photons", len(detected))
    print(f"fraction of detected photons: {photon_detection_efficiency:.4f}")

    p_flags = ev.photons_end.flags
    print("\tDetect", count_test(p_flags, SURFACE_DETECT))
    print("\tNoHit", count_test(p_flags, NO_HIT))
    print("\tAbort", count_test(p_flags, NAN_ABORT))
    print("\tSurfaceAbsorb", count_test(p_flags, SURFACE_ABSORB))
    print("\tBulkAbsorb", count_test(p_flags, BULK_ABSORB))
