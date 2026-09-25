"""Tape replay adapter for :mod:`chroma.triton.compat` (validation only).

``chroma.triton.compat.simulation.Simulation`` constructs
``LegacyEngine(detector, seed=, device=, nthreads_per_block=, max_blocks=,
tape=)`` when ``CHROMA_TRITON_TAPE`` is set. Until the production engine's
exact mode takes over that role, this adapter implements the
:class:`chroma.triton.engine.api.TransportEngine` protocol by replaying the
tape through the reference harness (:mod:`chroma.triton.legacy.reference`,
one step per kernel pair). It is a correctness tool for demonstrating the
public-API path (batching, events, hits, DAQ, tracking), not an engine.

Only ``replay:<dir>`` is supported. Fail-closed cases: a different detector
(uploaded words differ from the tape's), different Simulation parameters or
inputs, ``use_packed`` (the compat layer returns true final states), a tape
from a different number of batches.
"""

import numpy as np

from chroma.triton.engine.api import DaqChannels, DevicePhotons
from chroma.triton.legacy import tape as tapefmt
from chroma.triton.legacy.scene import check_legacy_limits, compare_scene_relabeled, scene_words


class LegacyEngine(object):
    def __init__(self, detector, seed=None, device=None, nthreads_per_block=512, max_blocks=1024, tape=None):
        import torch
        from chroma.triton.legacy import reference

        if tape is None or tape.mode != "replay":
            raise NotImplementedError(
                "the Triton legacy adapter replays native tapes only: record with CHROMA_BACKEND=cuda "
                "CHROMA_TRITON_TAPE=record:<dir>, then run CHROMA_BACKEND=triton CHROMA_TRITON_TAPE=replay:<dir>")
        self.device = torch.device(device if device is not None else "cuda")
        self.replay = tapefmt.TapeReplay(tape.directory)
        info = self.replay.info
        if info.get("use_packed"):
            raise NotImplementedError("use_packed tapes are verified with `verify tape` (the compat layer returns "
                                      "true final photon states, the CUDA backend the initial ones)")
        self.replay.check_simulation(seed=int(seed) if seed is not None else None,
                                     nthreads_per_block=int(nthreads_per_block), max_blocks=int(max_blocks))
        recorded = self.replay.tape.scene(self.replay.simulation)
        # Without geometry.bvh (no PyCUDA builder, no cached BVH) the recorded
        # BVH is used; every other uploaded word must still match.
        mine = scene_words(detector, require_bvh=False)
        problems = compare_scene_relabeled(mine, recorded)
        if problems:
            raise tapefmt.TapeError("the detector differs from the recorded one: %s" % problems[:5])
        self.bvh_from_tape = "nodes" not in mine
        check_legacy_limits(recorded)
        self.scene = reference.DeviceScene(recorded, self.device)
        self.solid_id = torch.from_numpy(np.asarray(recorded["solid_id_map"], np.uint32).view(np.int32).copy()).to(
            self.device)
        if "solid_id_to_channel_index" in recorded:
            self.solid_id_to_channel_index = torch.from_numpy(
                np.asarray(recorded["solid_id_to_channel_index"], np.int32).copy()).to(self.device)
        else:
            self.solid_id_to_channel_index = None
        self._batch = None
        self._daq = None

    def propagate(self, photons, *, max_steps, use_weights=False, track=False):
        import torch
        from chroma.triton.legacy import reference

        batch = self.replay.next_batch(photons, max_steps=int(max_steps), use_weights=bool(use_weights))
        self._batch = batch
        self._daq = None
        host = photons.to_numpy()
        snapshots = [] if track else None
        final, report = reference.propagate(self.scene, host, batch, self.device, snapshots=snapshots)
        if report["draw_count_mismatch"] or report["harness_errors"]:
            raise tapefmt.TapeError("batch %d replay diverged from the tape: %s" % (batch.index, report))
        for name in ("pos", "dir", "pol", "wavelengths", "t", "last_hit_triangles", "flags", "weights"):
            value = np.ascontiguousarray(final[name])
            if value.dtype == np.uint32:
                value = value.view(np.int32)
            getattr(photons, name).copy_(torch.from_numpy(value).to(self.device).reshape(getattr(photons, name).shape))
        if not track:
            return None
        steps = []
        for ids, state in snapshots:
            ids_t = torch.from_numpy(np.asarray(ids, np.int64)).to(self.device)
            dp = DevicePhotons(**{k: torch.from_numpy(np.ascontiguousarray(
                v.view(np.int32) if v.dtype == np.uint32 else v)).to(self.device) for k, v in state.items()},
                ids=ids_t.clone())
            steps.append((ids_t, dp))
        return steps

    def acquire(self, photons, start, count):
        from chroma.triton.legacy import reference

        if self._batch is None:
            raise tapefmt.TapeError("acquire before propagate")
        if self._daq is None:
            self._daq, mism = reference.daq(self.scene, photons.to_numpy(), self._batch)
            if mism:
                raise tapefmt.TapeError("DAQ replay consumed a different number of draws than recorded")
        bounds = self._batch.event_bounds
        event = int(np.searchsorted(bounds, start, side="right") - 1)
        if int(bounds[event]) != int(start) or int(bounds[event + 1] - bounds[event]) != int(count):
            raise tapefmt.TapeError("acquire(%d, %d) does not match a recorded event" % (start, count))
        result = self._daq.get(event)
        if result is None:
            raise tapefmt.TapeError("the tape has no DAQ record for event %d" % event)
        return DaqChannels(t=np.asarray(result["t"], np.float32), q=np.asarray(result["q"], np.float32),
                           flags=np.asarray(result["flags"], np.uint32))
