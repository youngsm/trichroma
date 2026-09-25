"""Native RNG tape: format, writer and reader.

A tape records one CUDA Chroma run made through ``chroma.sim.Simulation``
(``CHROMA_BACKEND=cuda``, ``CHROMA_TRITON_TAPE=record:<dir>``) so that another
implementation can replay it bit for bit. The CUDA kernels run unmodified;
see :mod:`chroma.triton.legacy.record_cuda` for how the draws are obtained.

Directory layout
----------------
``manifest.json``
    Format name/version, environment (Python, NumPy, PyCUDA, nvcc, device,
    driver, ``cuda_options``), SHA-256 of every ``chroma/*.py|.cu|.h`` source
    file, one entry per ``Simulation`` and one entry per batch (file name,
    parameters, array SHA-256).
``sim<S>_scene.npz``
    The exact words the CUDA backend uploaded for simulation ``S``
    (see :func:`chroma.triton.legacy.scene.scene_words` for the key list).
``sim<S>_batch<B>.npz``
    One ``_simulate_batch`` call. ``N`` photons, ``E`` events, ``L``
    propagate launches (host-loop iterations), ``C`` kernel chunks.

Batch arrays (all little endian; ``u32`` float draws are IEEE-754 bits)
-----------------------------------------------------------------------
Inputs, exactly as ``GPUPhotons`` holds them before ``propagate``:
    ``in_pos``/``in_dir``/``in_pol`` f32[N,3], ``in_wavelengths``/``in_t``
    f32[N], ``in_last_hit_triangles`` i32[N] (all -1), ``in_flags`` u32[N],
    ``in_weights`` f32[N] (all 1), ``in_evidx`` u32[N];
    ``event_bounds`` i64[E+1] (photon ranges of the events).
Propagation draws (CSR by photon):
    ``draws`` u32[D]: every ``curand_uniform`` value photon ``i`` consumed in
    ``propagate``/``propagate_packed``, over all launches, in call order, is
    ``draws[offsets[i]:offsets[i+1]]``; ``offsets`` i64[N+1].
Launch schedule:
    ``launch_starts`` i32[L]: global step index at which launch ``k`` began
    (every photon alive at that step is re-normalized on entry);
    ``launch_nsteps`` i32[L]; ``launch_nphotons`` i32[L] (queue length);
    ``chunk_launch``/``chunk_first``/``chunk_count``/``chunk_blocks`` i32[C];
    ``queue`` u32[Q] and ``queue_offsets`` i64[L+1]: the actual input queue of
    every launch; ``queue_draws`` u32[Q]: draws consumed by that entry in
    that launch.
DAQ draws (CSR by photon; only events with ``run_daq``):
    ``daq_draws`` u32[Dd], ``daq_offsets`` i64[N+1], ``daq_event_ran`` u8[E].
Outputs:
    ``final_*`` (same fields as ``in_*``): device photon state right after
    propagation. With ``use_packed`` the position/direction/polarization/
    wavelength/time/weight words come from the packed float4 arrays.
    ``end_*``: ``photons_end`` exactly as returned to the user (only with
    ``keep_photons_end``; with ``use_packed`` CUDA returns the *initial*
    position/direction/polarization/wavelength/time/weight words).
    ``hits_*`` and ``hits_event`` i32: ``flat_hits`` as returned (CUDA order
    is warp-atomic, compare as a multiset or in photon order).
    ``hits_expected_*`` and ``hits_expected_photon`` i64: the same hits
    derived from ``final_*`` in photon order (``hits_expected_event`` i32).
    ``channels_t``/``channels_q`` f32[E,nch], ``channels_flags`` u32[E,nch],
    ``channels_hit`` u8[E,nch] for events with DAQ.
    Tracking: ``track_step_offsets`` i64[S+1], ``track_ids`` u32[T] and
    ``track_*`` photon fields [T]: the ``(step_photon_ids, step_photons)``
    lists that ``GPUPhotons.propagate(track=True)`` returned.

Draw order inside ``propagate`` is documented in
:mod:`chroma.triton.engine.exact`.
"""

import hashlib
import json
import os
import time

import numpy as np

FORMAT_NAME = "chroma-triton-tape"
FORMAT_VERSION = 1

#: photon fields: (suffix, dtype, per-photon shape)
PHOTON_FIELDS = (
    ("pos", np.float32, (3,)),
    ("dir", np.float32, (3,)),
    ("pol", np.float32, (3,)),
    ("wavelengths", np.float32, ()),
    ("t", np.float32, ()),
    ("last_hit_triangles", np.int32, ()),
    ("flags", np.uint32, ()),
    ("weights", np.float32, ()),
    ("evidx", np.uint32, ()),
)
PHOTON_FIELD_NAMES = tuple(f[0] for f in PHOTON_FIELDS)

SURFACE_DETECT = 1 << 2


class TapeError(RuntimeError):
    """A tape is malformed, incomplete or does not match the replayed run."""


def digest(array):
    """SHA-256 of an array's raw bytes."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def photon_words(fields):
    """[N,15] uint32 words: pos, dir, pol, wavelength, t, flags, triangle, weight, evidx."""
    parts = []
    for name in ("pos", "dir", "pol", "wavelengths", "t", "flags", "last_hit_triangles", "weights", "evidx"):
        value = np.ascontiguousarray(fields[name])
        parts.append(value.view(np.uint32).reshape(len(value), -1))
    return np.ascontiguousarray(np.concatenate(parts, axis=1))


WORD_NAMES = ("pos.x", "pos.y", "pos.z", "dir.x", "dir.y", "dir.z", "pol.x", "pol.y", "pol.z",
              "wavelength", "t", "flags", "last_hit_triangle", "weight", "evidx")


def photon_order_hits(fields, solid_id, solid_id_to_channel_index, event_bounds=None):
    """Hits in photon order from final photon fields (Simulation/get_flat_hits rule)."""
    tri = np.asarray(fields["last_hit_triangles"], np.int64)
    flags = np.asarray(fields["flags"], np.uint32)
    detected = ((flags & SURFACE_DETECT) != 0) & (tri > -1)
    channel = np.full(len(tri), -1, np.int64)
    idx = np.flatnonzero(detected)
    if len(idx):
        channel[idx] = np.asarray(solid_id_to_channel_index)[np.asarray(solid_id)[tri[idx]]]
    rows = np.flatnonzero(detected & (channel >= 0))
    out = {name: np.ascontiguousarray(np.asarray(fields[name])[rows]) for name in PHOTON_FIELD_NAMES}
    out["channel"] = channel[rows].astype(np.uint32)
    out["photon"] = rows.astype(np.int64)
    if event_bounds is not None:
        out["event"] = (np.searchsorted(np.asarray(event_bounds), rows, side="right") - 1).astype(np.int32)
    return out


# --------------------------------------------------------------------- writer


class TapeWriter(object):
    """Writes a tape directory. Used by the CUDA recorder (no PyCUDA here)."""

    def __init__(self, directory, manifest_extra=None):
        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, "manifest.json")
        if os.path.exists(path):
            with open(path) as f:
                old = json.load(f)
            if old.get("format") != FORMAT_NAME:
                raise TapeError("%s exists and is not a %s manifest" % (path, FORMAT_NAME))
            raise TapeError("refusing to overwrite an existing tape in %s" % self.directory)
        self.manifest = dict(format=FORMAT_NAME, version=FORMAT_VERSION,
                             created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             simulations=[], batches=[], complete=False)
        if manifest_extra:
            self.manifest.update(manifest_extra)
        self.flush()

    def flush(self):
        path = os.path.join(self.directory, "manifest.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=1, sort_keys=True, default=_json_default)
        os.replace(tmp, path)

    def add_simulation(self, entry, scene=None):
        index = len(self.manifest["simulations"])
        entry = dict(entry, index=index)
        if scene is not None:
            name = "sim%04d_scene.npz" % index
            np.savez_compressed(os.path.join(self.directory, name), **scene)
            entry["scene_file"] = name
            entry["scene_sha256"] = {k: digest(v) for k, v in scene.items()}
        self.manifest["simulations"].append(entry)
        self.flush()
        return index

    def add_batch(self, simulation, arrays, entry):
        index = sum(1 for b in self.manifest["batches"] if b["simulation"] == simulation)
        name = "sim%04d_batch%05d.npz" % (simulation, index)
        np.savez(os.path.join(self.directory, name), **arrays)
        entry = dict(entry, simulation=simulation, index=index, file=name,
                     sha256={k: digest(v) for k, v in arrays.items()})
        self.manifest["batches"].append(entry)
        self.flush()
        return entry


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(repr(value))


# --------------------------------------------------------------------- reader


def compact(directory):
    """Rewrite every array file of a tape with zlib compression (archiving;
    the manifest hashes cover array contents, not files)."""
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".npz"):
            continue
        path = os.path.join(directory, name)
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)


class TapeBatch(object):
    """One recorded ``_simulate_batch`` call."""

    def __init__(self, tape, entry):
        self.tape = tape
        self.entry = entry
        self.simulation = entry["simulation"]
        self.index = entry["index"]
        self.nphotons = int(entry["nphotons"])
        self.params = entry.get("params", {})
        self._arrays = None

    # ------------------------------------------------------------- numpy view

    @property
    def arrays(self):
        if self._arrays is None:
            path = os.path.join(self.tape.directory, self.entry["file"])
            with np.load(path) as data:
                self._arrays = {k: data[k] for k in data.files}
            if self.tape.verify_hashes:
                for key, value in self.entry.get("sha256", {}).items():
                    if digest(self._arrays[key]) != value:
                        raise TapeError("%s: array %s does not match its manifest hash" % (path, key))
        return self._arrays

    def __getitem__(self, key):
        return self.arrays[key]

    def __contains__(self, key):
        return key in self.arrays

    @property
    def event_bounds(self):
        return self.arrays["event_bounds"]

    @property
    def nevents(self):
        return len(self.event_bounds) - 1

    def fields(self, prefix):
        """Photon fields ``prefix_*`` as a dict of numpy arrays (or None)."""
        if prefix + "_pos" not in self.arrays:
            return None
        return {name: self.arrays[prefix + "_" + name] for name in PHOTON_FIELD_NAMES}

    def inputs(self):
        return self.fields("in")

    def final(self):
        return self.fields("final")

    def photon_draws(self, i):
        """Float32 draws of photon ``i`` (numpy)."""
        o = self.arrays["offsets"]
        return self.arrays["draws"][o[i]:o[i + 1]].view(np.float32)

    # ------------------------------------------------------------ device view

    def _torch(self, key, device, dtype=None, view=None):
        import torch

        value = self.arrays[key]
        if view is not None:
            value = value.view(view)
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        if dtype is not None:
            tensor = tensor.to(dtype)
        return tensor.to(device)

    def draws(self, device="cuda"):
        """float32 tensor [D] (bit-identical to the CUDA ``curand_uniform`` values)."""
        return self._torch("draws", device, view=np.float32)

    def offsets(self, device="cuda"):
        """int64 tensor [N+1]."""
        return self._torch("offsets", device)

    def launch_starts(self, device="cuda"):
        """int32 tensor [L]."""
        return self._torch("launch_starts", device)

    def launch_nsteps(self, device="cuda"):
        return self._torch("launch_nsteps", device)

    def daq_draws(self, device="cuda"):
        return self._torch("daq_draws", device, view=np.float32)

    def daq_offsets(self, device="cuda"):
        return self._torch("daq_offsets", device)

    def device_inputs(self, device="cuda", ids_start=0):
        """Recorded inputs as :class:`chroma.triton.engine.api.DevicePhotons`."""
        import torch
        from chroma.triton.engine.api import DevicePhotons

        f = self.inputs()
        count = self.nphotons

        def t(name, dtype=None):
            value = np.ascontiguousarray(f[name])
            if value.dtype == np.uint32:
                value = value.view(np.int32)
            return torch.from_numpy(value.copy()).to(device)

        return DevicePhotons(pos=t("pos"), dir=t("dir"), pol=t("pol"), wavelengths=t("wavelengths"),
                             t=t("t"), last_hit_triangles=t("last_hit_triangles"), flags=t("flags"),
                             weights=t("weights"), evidx=t("evidx"),
                             ids=torch.arange(ids_start, ids_start + count, device=device, dtype=torch.int64))

    def check_inputs(self, photons):
        """Raise :class:`TapeError` unless ``photons`` equals the recorded inputs bitwise.

        ``photons`` is a DevicePhotons or a dict of numpy arrays. Only the
        fields the CUDA backend copies are compared (``pos``, ``dir``, ``pol``,
        ``wavelengths``, ``t``, ``flags``, ``evidx``); triangles are -1 and
        weights 1 in both backends.
        """
        expected = self.inputs()
        if hasattr(photons, "to_numpy"):
            photons = photons.to_numpy()
        if len(photons["wavelengths"]) != self.nphotons:
            raise TapeError("batch %d: replay has %d photons, tape has %d"
                            % (self.index, len(photons["wavelengths"]), self.nphotons))
        for name in ("pos", "dir", "pol", "wavelengths", "t", "flags", "evidx"):
            a = np.ascontiguousarray(expected[name]).view(np.uint32).reshape(self.nphotons, -1)
            b = np.ascontiguousarray(photons[name]).view(np.uint32).reshape(self.nphotons, -1)
            if not np.array_equal(a, b):
                row = int(np.flatnonzero(np.any(a != b, axis=1))[0])
                raise TapeError("batch %d: input field %s differs from the tape first at photon %d"
                                % (self.index, name, row))


class Tape(object):
    """Read a tape directory written by the CUDA recorder."""

    def __init__(self, directory, verify_hashes=True):
        self.directory = os.path.abspath(directory)
        self.verify_hashes = verify_hashes
        path = os.path.join(self.directory, "manifest.json")
        if not os.path.exists(path):
            raise TapeError("no tape manifest at %s" % path)
        with open(path) as f:
            self.manifest = json.load(f)
        if self.manifest.get("format") != FORMAT_NAME:
            raise TapeError("%s is not a %s manifest" % (path, FORMAT_NAME))
        if int(self.manifest.get("version", 0)) != FORMAT_VERSION:
            raise TapeError("unsupported tape version %r" % self.manifest.get("version"))
        self._batches = [TapeBatch(self, e) for e in self.manifest["batches"]]

    @property
    def simulations(self):
        return self.manifest["simulations"]

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)

    def batch(self, i):
        return self._batches[i]

    def batches_of(self, simulation):
        return [b for b in self._batches if b.simulation == simulation]

    def scene(self, simulation=0):
        entry = self.simulations[simulation]
        with np.load(os.path.join(self.directory, entry["scene_file"])) as data:
            return {k: data[k] for k in data.files}


class TapeReplay(object):
    """Sequential consumer for an engine replaying one recorded Simulation.

    ``next_batch(photons)`` returns the next recorded batch after checking
    that ``photons`` equals its recorded inputs. A tape recorded from several
    ``Simulation`` objects is replayed by constructing one ``TapeReplay`` per
    simulation in the same order (``simulation=`` index).
    """

    def __init__(self, directory, simulation=None, verify_hashes=True):
        self.tape = Tape(directory, verify_hashes=verify_hashes)
        if simulation is None:
            simulation = _claim_simulation(self.tape.directory)
        if simulation >= len(self.tape.simulations):
            raise TapeError("tape %s has %d recorded simulations; replay needs #%d"
                            % (directory, len(self.tape.simulations), simulation))
        self.simulation = simulation
        self.info = self.tape.simulations[simulation]
        self._batches = self.tape.batches_of(simulation)
        self._next = 0

    def check_simulation(self, seed=None, nthreads_per_block=None, max_blocks=None,
                         photon_tracking=None, use_packed=None):
        for key, value in (("seed", seed), ("nthreads_per_block", nthreads_per_block),
                           ("max_blocks", max_blocks), ("photon_tracking", photon_tracking),
                           ("use_packed", use_packed)):
            if value is not None and self.info.get(key) != value:
                raise TapeError("replayed Simulation has %s=%r, the tape has %r"
                                % (key, value, self.info.get(key)))

    def next_batch(self, photons=None, **params):
        if self._next >= len(self._batches):
            raise TapeError("the replay requests more batches than the tape recorded (%d)" % len(self._batches))
        batch = self._batches[self._next]
        self._next += 1
        if photons is not None:
            batch.check_inputs(photons)
        for key, value in params.items():
            if key in batch.params and batch.params[key] != value:
                raise TapeError("batch %d: replay has %s=%r, the tape has %r"
                                % (batch.index, key, value, batch.params[key]))
        return batch

    @property
    def remaining(self):
        return len(self._batches) - self._next


_CLAIMS = {}


def _claim_simulation(directory):
    """Simulations of one tape are replayed in construction order per process."""
    index = _CLAIMS.get(directory, 0)
    _CLAIMS[directory] = index + 1
    return index
