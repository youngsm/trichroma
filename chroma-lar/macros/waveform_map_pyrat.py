import os
import logging
import numpy as np
import h5py
import time

from chroma.log import logger
logger.setLevel(logging.INFO)
from chroma.event import Photons

from chroma_lar.geometry import build_detector_from_config
from photonlib.meta import VoxelMeta


def sample_photon_bomb(nphotons, pos, voxel_size=30, wavelength=128) -> Photons:
    # random direction
    costheta = np.random.random(nphotons)*2-1
    sintheta = np.sqrt(1-np.square(costheta))
    phi = np.random.random(nphotons)*2*np.pi
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    pdir = np.transpose([sintheta*cosphi, sintheta*sinphi, costheta])

    # random polarization
    costheta = np.random.random(nphotons)*2-1
    sintheta = np.sqrt(1-np.square(costheta))
    phi = np.random.random(nphotons)*2*np.pi
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    rand_unit = np.transpose([sintheta*cosphi, sintheta*sinphi, costheta])
    ppol = np.cross(pdir, rand_unit)
    ppol = ppol / np.linalg.norm(ppol, ord=2, axis=1, keepdims=True)

    # wavelength
    if type(wavelength) is tuple:
        pwavelength = (
            np.random.random(nphotons) * (wavelength[1] - wavelength[0]) + wavelength[0]
        )
    else:
        pwavelength = np.tile(wavelength, nphotons)

    # random position between -voxel_size/2 and voxel_size/2 wrt voxel center
    x = np.random.random(nphotons)*voxel_size-voxel_size/2+pos[0]
    y = np.random.random(nphotons)*voxel_size-voxel_size/2+pos[1]
    z = np.random.random(nphotons)*voxel_size-voxel_size/2+pos[2]
    ppos = np.transpose([x,y,z])

    return Photons(pos=ppos, dir=pdir, pol=ppol, wavelengths=pwavelength)

def __configure__(db):
    """Modify fields in the database here"""
    db.output_filename = "waveform_map.h5"
    db.detector_config = "detector_config_reflect_reflect3wires"
    db.voxel_ranges = None
    db.voxel_size = None

    db.voxel_index_start = 0
    db.batch_size = 720
    db.nphotons = 200_000
    db.wavelength = 450
    db.num_ticks = 1000
    db.max_time = 100
    
    db.chroma_photon_tracking = 0
    db.chroma_daq = False
    db.chroma_photons_per_batch = db.nphotons
    db.chroma_keep_photons_beg = False
    db.chroma_keep_photons_end = False
    db.chroma_keep_hits = False
    db.chroma_keep_flat_hits = True
    db.chroma_max_steps = 1000


def __define_geometry__(db):
    """Returns a chroma Detector or Geometry"""
    geometry = build_detector_from_config(
        db.detector_config,
        flatten=True,
        include_wires=True,
        include_active=True,
        include_cathode=True,
        include_cavity=True,
    )
    db.geometry = geometry
    return geometry


def __event_generator__(db):
    """A generator to yield chroma Events (or something a chroma Simulation can
    convert to a chroma Event)."""
    meta = VoxelMeta(shape=db.voxel_shape, ranges=db.voxel_ranges)
    db.meta = meta
    db.voxel_ids = range(db.voxel_index_start, db.voxel_index_start + db.batch_size)
    for idx in db.voxel_ids:
        pos = meta.voxel_to_coord(idx).numpy()
        yield sample_photon_bomb(db.nphotons, pos, voxel_size=db.voxel_size, wavelength=db.wavelength)

def __simulation_start__(db):
    """Called at the start of the event loop"""
    if db.voxel_ranges is None:
        db.voxel_ranges = ((-2310, 0), (-2160, 2160), (-2160, 2160))
    if db.voxel_size is None:
        db.voxel_size = 30
    
    db.voxel_shape = (
        (db.voxel_ranges[0][1] - db.voxel_ranges[0][0]) // db.voxel_size,
        (db.voxel_ranges[1][1] - db.voxel_ranges[1][0]) // db.voxel_size,
        (db.voxel_ranges[2][1] - db.voxel_ranges[2][0]) // db.voxel_size,
    )
    db.num_pmts = db.geometry.num_channels()

    # create h5 file
    if os.path.exists(db.output_filename):
        logger.warning(f"File {db.output_filename} already exists! Removing...")
        os.remove(db.output_filename)
    db.file = h5py.File(db.output_filename, "w")
    db.file.create_dataset(
        "counts",
        shape=(0,db.num_pmts*db.num_ticks // 2),
        maxshape=(None,db.num_pmts*db.num_ticks // 2),
        dtype=np.uint16, # max: 65535
        chunks=True
    )
    db.file.create_dataset(
        "voxel_ids",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint32, # max: a lot more than 65535
        chunks=True
    )
    db.file.create_dataset(
        "vis_counts",
        shape=(0,db.num_pmts),
        maxshape=(None,db.num_pmts),
        dtype=np.uint32, # max: a lot more than 65535
        chunks=True
    )
    db.current_ev_idx = 0
    db.t_start = time.time()

def __process_event__(db, ev):
    """Called for each generated event"""
    logger.info(f"Processing event {db.current_ev_idx} of {db.batch_size} in {time.time() - db.t_start:.2f} seconds")
    logger.info(f'\t detections: {len(ev.flat_hits)}/{db.nphotons}')
    unique_channels, flat_counts = np.unique(ev.flat_hits.channel, return_counts=True)
    logger.info(f'\t unique channels: {len(unique_channels)}')
    logger.info(f"\t pos: {db.meta.voxel_to_coord(db.voxel_ids[db.current_ev_idx]).numpy()}")

    db.t_start = time.time()
    
    # fill hist2d
    times = ev.flat_hits.t
    channels = ev.flat_hits.channel
    counts = np.histogram2d(
        channels,
        times,
        bins=(db.num_pmts // 2, db.num_ticks),
        range=((0, db.num_pmts // 2), (0, db.max_time)),
    )[0].flatten()
    logger.info(f"\t counts: {sum(counts)}")

    # save to h5 file! uint
    if np.any(counts > 65535):
        logger.warning(f"Per-bin waveform count exceeds uint16 maximum value ({counts[counts > 65535]}); clamping to max of 65535")
        counts = np.clip(counts, None, 65535)
    counts = counts.astype(np.uint16)
    db.file["counts"].resize(db.file["counts"].shape[0] + 1, axis=0)
    db.file["counts"][-1] = counts
    db.file["voxel_ids"].resize(db.file["voxel_ids"].shape[0] + 1, axis=0)
    db.file["voxel_ids"][-1] = db.voxel_ids[db.current_ev_idx]

    out_vis_counts = np.zeros(db.num_pmts, dtype=np.uint32)
    out_vis_counts[unique_channels] = flat_counts.astype(np.uint32)
    db.file["vis_counts"].resize(db.file["vis_counts"].shape[0] + 1, axis=0)
    db.file["vis_counts"][-1] = out_vis_counts

    if (db.current_ev_idx + 1) % 10 == 0:
        db.file.flush()

    # increment event index
    db.current_ev_idx += 1

def __simulation_end__(db):
    """Called at the end of the event loop"""
    db.file.close() 