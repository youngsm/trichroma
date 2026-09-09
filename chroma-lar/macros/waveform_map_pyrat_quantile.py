"""
Waveform map macro that stores compact timing shape representations directly.

Per voxel, this macro stores:
  - yield_counts[channel]: number of detected photons
  - t0[channel]: minimum detected time for that PMT in this voxel
  - quantiles[channel, k]: empirical time quantiles Q(u_k) for u_k in u_grid

Optional audit mode can also store raw hits for a deterministic subset of voxels
to preserve ground truth for later validation.
"""

import os
import sys
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
    costheta = np.random.random(nphotons) * 2 - 1
    sintheta = np.sqrt(1 - np.square(costheta))
    phi = np.random.random(nphotons) * 2 * np.pi
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    pdir = np.transpose([sintheta * cosphi, sintheta * sinphi, costheta])

    # random polarization
    costheta = np.random.random(nphotons) * 2 - 1
    sintheta = np.sqrt(1 - np.square(costheta))
    phi = np.random.random(nphotons) * 2 * np.pi
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    rand_unit = np.transpose([sintheta * cosphi, sintheta * sinphi, costheta])
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
    x = np.random.random(nphotons) * voxel_size - voxel_size / 2 + pos[0]
    y = np.random.random(nphotons) * voxel_size - voxel_size / 2 + pos[1]
    z = np.random.random(nphotons) * voxel_size - voxel_size / 2 + pos[2]
    ppos = np.transpose([x, y, z])

    return Photons(pos=ppos, dir=pdir, pol=ppol, wavelengths=pwavelength)


def _parse_dtype(dtype_name):
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "float64": np.float64,
    }
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported quantile_dtype={dtype_name!r}; choose from {list(dtype_map)}")
    return dtype_map[dtype_name]


def _quantiles_from_sorted_samples(sorted_samples, u_grid):
    n = sorted_samples.size
    if n == 0:
        return None
    if n == 1:
        return np.full(u_grid.shape, sorted_samples[0], dtype=np.float64)

    # Linear interpolation between adjacent order statistics.
    idxf = u_grid * (n - 1)
    i0 = np.floor(idxf).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    alpha = idxf - i0
    return (1.0 - alpha) * sorted_samples[i0] + alpha * sorted_samples[i1]


def _compute_yield_and_quantiles_and_t0(times, channels, num_channels, u_grid, time_clip_max=None):
    yield_counts = np.bincount(channels, minlength=num_channels).astype(np.uint32)
    q = np.full((num_channels, u_grid.shape[0]), np.nan, dtype=np.float64)
    t0 = np.full((num_channels,), np.nan, dtype=np.float64)

    if times.size == 0:
        return yield_counts, q, t0

    # Group by channel without repeated boolean masks.
    order = np.argsort(channels, kind="stable")
    ch_sorted = channels[order]
    t_sorted = times[order]
    unique_ch, start_idx, counts = np.unique(ch_sorted, return_index=True, return_counts=True)

    for ch, start, count in zip(unique_ch, start_idx, counts):
        end = start + count
        t_ch_all = t_sorted[start:end]
        t0[ch] = np.min(t_ch_all)
        t_ch = t_ch_all
        if time_clip_max is not None:
            t_ch = t_ch_all[t_ch_all <= time_clip_max]
        if t_ch.size == 0:
            continue
        t_ch = np.sort(t_ch)
        q_ch = _quantiles_from_sorted_samples(t_ch, u_grid)
        if q_ch is not None:
            q[ch] = q_ch

    return yield_counts, q, t0


def __configure__(db):
    """Modify fields in the database here"""
    db.output_filename = "waveform_map_quantile.h5"
    db.detector_config = "detector_config_reflect_reflect3wires"
    db.voxel_ranges = None
    db.voxel_size = None

    db.voxel_index_start = 0
    db.batch_size = 720
    db.nphotons = 200_000
    db.wavelength = 450

    # Quantile representation controls.
    db.num_quantiles = 512
    db.quantile_dtype = "float32"
    db.quantile_u_min = 1e-4
    db.quantile_u_max = 1.0 - 1e-4
    db.time_clip_max = None

    # Channel controls.
    db.num_output_channels = None  # default: half geometry channels (historical waveform convention)

    # Optional voxel subset mode.
    db.voxel_id_file = None
    db.voxel_ids_array = None

    # Optional audit mode: keep raw hits for every Nth processed voxel.
    db.audit_every_n_voxels = 16_000 # 0 disables audit storage

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
    logger.info(f"Building geometry...")
    geometry = build_detector_from_config(
        db.detector_config,
        flatten=True,
        include_wires=True,
        include_active=True,
        include_cathode=True,
        include_cavity=True,
    )
    logger.info(f"Built geometry.")
    db.geometry = geometry
    return geometry


def __event_generator__(db):
    """A generator to yield chroma events."""
    meta = VoxelMeta(shape=db.voxel_shape, ranges=db.voxel_ranges)
    db.meta = meta

    # determine which voxel IDs to simulate
    if db.voxel_ids_array is not None:
        end_idx = min(db.voxel_index_start + db.batch_size, len(db.voxel_ids_array))
        db.voxel_ids = db.voxel_ids_array[db.voxel_index_start:end_idx]
    else:
        db.voxel_ids = range(db.voxel_index_start, db.voxel_index_start + db.batch_size)

    for idx in db.voxel_ids:
        pos = meta.voxel_to_coord(idx).numpy()
        yield sample_photon_bomb(
            db.nphotons, pos, voxel_size=db.voxel_size, wavelength=db.wavelength
        )


def __simulation_start__(db):
    """Called at the start of the event loop"""
    assert "voxel_ranges" in db, "voxel_ranges must be set"
    assert "voxel_size" in db, "voxel_size must be set"
    logger.info(f"Initializing simulation...")

    db.voxel_shape = (
        (db.voxel_ranges[0][1] - db.voxel_ranges[0][0]) // db.voxel_size,
        (db.voxel_ranges[1][1] - db.voxel_ranges[1][0]) // db.voxel_size,
        (db.voxel_ranges[2][1] - db.voxel_ranges[2][0]) // db.voxel_size,
    )
    db.num_pmts = db.geometry.num_channels()
    db.num_output_channels = (
        db.num_pmts // 2 if db.num_output_channels is None else int(db.num_output_channels)
    )

    if db.num_output_channels <= 0:
        raise ValueError(f"num_output_channels must be positive, got {db.num_output_channels}")

    if db.quantile_u_min <= 0 or db.quantile_u_max >= 1 or db.quantile_u_min >= db.quantile_u_max:
        raise ValueError(
            "Require 0 < quantile_u_min < quantile_u_max < 1; "
            f"got ({db.quantile_u_min}, {db.quantile_u_max})"
        )

    if db.num_quantiles < 2:
        raise ValueError(f"num_quantiles must be >= 2, got {db.num_quantiles}")

    db.u_grid = np.linspace(db.quantile_u_min, db.quantile_u_max, db.num_quantiles, dtype=np.float64)
    db.q_dtype = _parse_dtype(db.quantile_dtype)

    # load voxel IDs from file if specified
    if db.voxel_id_file is not None:
        if not os.path.exists(db.voxel_id_file):
            raise FileNotFoundError(f"Voxel ID file not found: {db.voxel_id_file}")

        logger.info(f"Loading voxel IDs from {db.voxel_id_file}")
        db.voxel_ids_array = np.load(db.voxel_id_file).astype(np.int32)

        if db.voxel_ids_array.ndim != 1:
            raise ValueError(f"Voxel ID array must be 1D, got shape {db.voxel_ids_array.shape}")

        total_voxels = db.voxel_shape[0] * db.voxel_shape[1] * db.voxel_shape[2]
        if np.any(db.voxel_ids_array < 0) or np.any(db.voxel_ids_array >= total_voxels):
            raise ValueError(
                f"Voxel IDs must be in range [0, {total_voxels}), "
                f"found min={np.min(db.voxel_ids_array)}, max={np.max(db.voxel_ids_array)}"
            )

        logger.info(f"Loaded {len(db.voxel_ids_array)} voxel IDs (subset mode)")
        logger.info(
            "This job will process voxels %d to %d",
            db.voxel_index_start,
            min(db.voxel_index_start + db.batch_size, len(db.voxel_ids_array)) - 1,
        )
    else:
        logger.info(
            "Using full grid mode: %s = %d voxels",
            db.voxel_shape,
            db.voxel_shape[0] * db.voxel_shape[1] * db.voxel_shape[2],
        )
        logger.info(
            "This job will process voxels %d to %d",
            db.voxel_index_start,
            db.voxel_index_start + db.batch_size - 1,
        )

    # create h5 output
    if os.path.exists(db.output_filename):
        logger.warning(f"File {db.output_filename} already exists! Removing...")
        os.remove(db.output_filename)
    db.file = h5py.File(db.output_filename, "w")

    db.file.create_dataset(
        "yield_counts",
        shape=(0, db.num_output_channels),
        maxshape=(None, db.num_output_channels),
        dtype=np.uint32,
        chunks=True,
    )
    db.file.create_dataset(
        "quantiles",
        shape=(0, db.num_output_channels, db.num_quantiles),
        maxshape=(None, db.num_output_channels, db.num_quantiles),
        dtype=db.q_dtype,
        chunks=True,
        fillvalue=np.nan,
    )
    db.file.create_dataset(
        "voxel_id",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint32,
        chunks=True,
    )
    db.file.create_dataset(
        "t0",
        shape=(0, db.num_output_channels),
        maxshape=(None, db.num_output_channels),
        dtype=np.float32,
        chunks=True,
        fillvalue=np.nan,
    )
    db.file.create_dataset(
        "pos",
        shape=(0, 3),
        maxshape=(None, 3),
        dtype=np.float32,
        chunks=True,
    )
    db.file.create_dataset(
        "u_grid",
        data=db.u_grid.astype(np.float32),
        dtype=np.float32,
    )

    # Optional audit datasets for raw-hit validation.
    db.audit_enabled = int(db.audit_every_n_voxels) > 0
    if db.audit_enabled:
        db.file.create_dataset("audit_t", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)
        db.file.create_dataset("audit_channel", shape=(0,), maxshape=(None,), dtype=np.uint16, chunks=True)
        db.file.create_dataset("audit_voxel_id", shape=(0,), maxshape=(None,), dtype=np.uint32, chunks=True)
        db.file.create_dataset("audit_pos", shape=(0, 3), maxshape=(None, 3), dtype=np.float32, chunks=True)
        db.file.create_dataset("audit_offset", shape=(0,), maxshape=(None,), dtype=np.uint64, chunks=True)
        db.file.create_dataset("audit_nhits", shape=(0,), maxshape=(None,), dtype=np.uint32, chunks=True)

    # Metadata for downstream reconstruction.
    db.file.attrs["nphotons_generated"] = int(db.nphotons)
    db.file.attrs["num_output_channels"] = int(db.num_output_channels)
    db.file.attrs["num_quantiles"] = int(db.num_quantiles)
    db.file.attrs["quantile_dtype"] = db.quantile_dtype
    db.file.attrs["quantile_u_min"] = float(db.quantile_u_min)
    db.file.attrs["quantile_u_max"] = float(db.quantile_u_max)
    db.file.attrs["time_clip_max"] = float(db.time_clip_max) if db.time_clip_max is not None else -1.0
    db.file.attrs["audit_every_n_voxels"] = int(db.audit_every_n_voxels)

    db.current_ev_idx = 0
    db.t_start = time.time()
    
    logger.info(f"Simulation initialized.")


def __process_event__(db, ev):
    """Called for each generated event"""
    elapsed = time.time() - db.t_start
    logger.info(f"Processing event {db.current_ev_idx} of {db.batch_size} in {elapsed:.2f} seconds")
    logger.info(f"\t detections: {len(ev.flat_hits)}/{db.nphotons}")

    db.t_start = time.time()

    times = np.asarray(ev.flat_hits.t, dtype=np.float64)
    channels = np.asarray(ev.flat_hits.channel, dtype=np.int64)

    # Keep only the configured channel range.
    valid = (channels >= 0) & (channels < db.num_output_channels) & np.isfinite(times)
    dropped = np.count_nonzero(~valid)
    if dropped > 0:
        logger.info(f"\t dropped hits outside output channel range or invalid time: {dropped}")
    times = times[valid]
    channels = channels[valid]

    yields, quantiles, t0 = _compute_yield_and_quantiles_and_t0(
        times=times,
        channels=channels,
        num_channels=db.num_output_channels,
        u_grid=db.u_grid,
        time_clip_max=db.time_clip_max,
    )

    vid = np.uint32(db.voxel_ids[db.current_ev_idx])
    pos = db.meta.voxel_to_coord(db.voxel_ids[db.current_ev_idx]).numpy().astype(np.float32)

    # Append compact outputs.
    ds = db.file["yield_counts"]
    ds.resize(ds.shape[0] + 1, axis=0)
    ds[-1] = yields

    ds = db.file["quantiles"]
    ds.resize(ds.shape[0] + 1, axis=0)
    ds[-1] = quantiles.astype(db.q_dtype, copy=False)

    ds = db.file["voxel_id"]
    ds.resize(ds.shape[0] + 1, axis=0)
    ds[-1] = vid

    ds = db.file["t0"]
    ds.resize(ds.shape[0] + 1, axis=0)
    ds[-1] = t0.astype(np.float32, copy=False)

    ds = db.file["pos"]
    ds.resize(ds.shape[0] + 1, axis=0)
    ds[-1] = pos

    # Optional audit: store full raw hits for every Nth processed voxel.
    if db.audit_enabled and ((db.current_ev_idx % int(db.audit_every_n_voxels)) == 0):
        logger.info(f"Storing audit data for voxel {db.voxel_ids[db.current_ev_idx]}")
        audit_offset = db.file["audit_t"].shape[0]
        n = len(times)

        if n > 0:
            ds = db.file["audit_t"]
            ds.resize(ds.shape[0] + n, axis=0)
            ds[-n:] = times.astype(np.float32, copy=False)

            ds = db.file["audit_channel"]
            ds.resize(ds.shape[0] + n, axis=0)
            ds[-n:] = channels.astype(np.uint16, copy=False)

        for key, value in (
            ("audit_voxel_id", vid),
            ("audit_pos", pos),
            ("audit_offset", np.uint64(audit_offset)),
            ("audit_nhits", np.uint32(n)),
        ):
            ds = db.file[key]
            ds.resize(ds.shape[0] + 1, axis=0)
            ds[-1] = value

    if (db.current_ev_idx + 1) % 10 == 0:
        db.file.flush()

    db.current_ev_idx += 1


def __simulation_end__(db):
    """Called at the end of the event loop"""
    logger.info(f"Simulation ended.")
    logger.info(f"Closing file...")
    db.file.close()
