"""
This macro is used to simulate the waveform LUT of the detector.

The first 10 ns are stored in a 32-bit unsigned integer, and the rest are stored in a 16-bit unsigned integer.


"""

import os
import logging
import numpy as np
import h5py
import time

from chroma.log import logger
from chroma.event import Photons

from chroma_lar.geometry import build_detector_from_config
from photonlib.meta import VoxelMeta

MAX_UINT32 = np.iinfo(np.uint32).max
MAX_UINT16 = np.iinfo(np.uint16).max

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
    
    # voxel subset mode: path to .npy file containing voxel IDs to simulate
    db.voxel_id_file = None
    db.voxel_ids_array = None  # loaded array of voxel IDs
    
    db.chroma_photon_tracking = 0
    db.chroma_daq = False
    db.chroma_photons_per_batch = db.nphotons
    db.chroma_keep_photons_beg = False
    db.chroma_keep_photons_end = False
    db.chroma_keep_hits = False
    db.chroma_keep_flat_hits = True
    db.chroma_max_steps = 1000
    db.chroma_use_packed = True  # use float4 packed format for A100 optimization


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
    
    # determine which voxel IDs to simulate
    if db.voxel_ids_array is not None:
        # subset mode: slice the loaded array
        end_idx = min(db.voxel_index_start + db.batch_size, len(db.voxel_ids_array))
        db.voxel_ids = db.voxel_ids_array[db.voxel_index_start:end_idx]
    else:
        # full grid mode: sequential indices
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
    
    # load voxel IDs from file if specified (subset mode)
    if db.voxel_id_file is not None:
        if not os.path.exists(db.voxel_id_file):
            raise FileNotFoundError(f"Voxel ID file not found: {db.voxel_id_file}")
        
        logger.info(f"Loading voxel IDs from {db.voxel_id_file}")
        db.voxel_ids_array = np.load(db.voxel_id_file).astype(np.int32)
        
        if db.voxel_ids_array.ndim != 1:
            raise ValueError(f"Voxel ID array must be 1D, got shape {db.voxel_ids_array.shape}")
        
        # validate voxel IDs are within valid range
        total_voxels = db.voxel_shape[0] * db.voxel_shape[1] * db.voxel_shape[2]
        if np.any(db.voxel_ids_array < 0) or np.any(db.voxel_ids_array >= total_voxels):
            raise ValueError(f"Voxel IDs must be in range [0, {total_voxels}), found min={np.min(db.voxel_ids_array)}, max={np.max(db.voxel_ids_array)}")
        
        logger.info(f"Loaded {len(db.voxel_ids_array)} voxel IDs (subset mode)")
        logger.info(f"This job will process voxels {db.voxel_index_start} to {min(db.voxel_index_start + db.batch_size, len(db.voxel_ids_array))-1}")
    else:
        logger.info(f"Using full grid mode: {db.voxel_shape} = {db.voxel_shape[0] * db.voxel_shape[1] * db.voxel_shape[2]} voxels")
        logger.info(f"This job will process voxels {db.voxel_index_start} to {db.voxel_index_start + db.batch_size - 1}")

    # create h5 file
    if os.path.exists(db.output_filename):
        logger.warning(f"File {db.output_filename} already exists! Removing...")
        os.remove(db.output_filename)
    db.file = h5py.File(db.output_filename, "w")
    
    db.num_ticks_early = 100
    num_channels = db.num_pmts // 2
    
    db.file.create_dataset(
        "counts_early",
        shape=(0, num_channels*db.num_ticks_early),
        maxshape=(None, num_channels*db.num_ticks_early),
        dtype=np.uint32, 
        chunks=True
    )-
    db.file.create_dataset(
        "counts_late",
        shape=(0, num_channels*(db.num_ticks - db.num_ticks_early)),
        maxshape=(None, num_channels*(db.num_ticks - db.num_ticks_early)),
        dtype=np.uint16,
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
    print(f"Processing event {db.current_ev_idx} of {db.batch_size} in {time.time() - db.t_start:.2f} seconds")
    print(f'\t detections: {len(ev.flat_hits)}/{db.nphotons}')
    unique_channels, flat_counts = np.unique(ev.flat_hits.channel, return_counts=True)
    print(f'\t unique channels: {len(unique_channels)}')
    print(f"\t pos: {db.meta.voxel_to_coord(db.voxel_ids[db.current_ev_idx]).numpy()}")

    db.t_start = time.time()
    
    # fill hist2d
    times = ev.flat_hits.t
    channels = ev.flat_hits.channel
    counts_2d = np.histogram2d(
        channels,
        times,
        bins=(db.num_pmts // 2, db.num_ticks),
        range=((0, db.num_pmts // 2), (0, db.max_time)),
    )[0]

    counts_early = counts_2d[:, :db.num_ticks_early].flatten()
    counts_late = counts_2d[:, db.num_ticks_early:].flatten()
    print(f"\t counts: {np.sum(counts_2d)}")

    # Early counts: uint32
    if np.any(counts_early > MAX_UINT32):
        logger.warning(f"Per-bin early waveform count exceeds uint32 maximum value {counts_early[counts_early > MAX_UINT32]}; clamping to max of {MAX_UINT32}")
        counts_early = np.clip(counts_early, None, MAX_UINT32)
    counts_early = counts_early.astype(np.uint32)
    db.file["counts_early"].resize(db.file["counts_early"].shape[0] + 1, axis=0)
    db.file["counts_early"][-1] = counts_early

    # Late counts: uint16
    if np.any(counts_late > MAX_UINT16):
        logger.warning(f"Per-bin late waveform count exceeds uint16 maximum value ({counts_late[counts_late > MAX_UINT16]}); clamping to max of {MAX_UINT16}")
        counts_late = np.clip(counts_late, None, MAX_UINT16)
    counts_late = counts_late.astype(np.uint16)
    db.file["counts_late"].resize(db.file["counts_late"].shape[0] + 1, axis=0)
    db.file["counts_late"][-1] = counts_late
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