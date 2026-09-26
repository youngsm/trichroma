"""Reference loop: replay a native RNG tape on the host-driven launch schedule.

The production engine's exact mode (:mod:`trichroma.engine.exact_mode`)
replays tapes with its device-queue scheduler. This module drives the *same*
exact kernels the way W's host loop does -- launch by launch, the active rows
compacted on the host after every step -- which makes it an independent check
of the scheduling and the timing baseline for the engine. It needs no
detector: the scene comes from the tape.

For every batch :func:`replay_tape` checks, byte for byte, the final photon
words (all fields), ``photons_end`` (with W's ``use_packed`` quirk), the hits
in photon order, the DAQ channels, ``photon_tracks``, and that every photon
consumed exactly its recorded number of draws.
"""

import time

import numpy as np
import torch
import triton

from trichroma.engine import exact_mode as M
from trichroma.engine.exact_mode import ExactScene, bvh_stack_bound  # noqa: F401 (re-exported)
from trichroma.tape import format as tapefmt

DeviceScene = ExactScene  # historical name
BLOCK = M.BLOCK


def _torch():
    return torch


# ------------------------------------------------------------------ replay


def _snapshot(n, rows, tensors, fields_in):
    pos, dirs, pols, wls, times, lht, flags, weights = tensors
    idx = rows.long()
    state = dict(pos=pos.view(n, 3)[idx].cpu().numpy(), dir=dirs.view(n, 3)[idx].cpu().numpy(),
                 pol=pols.view(n, 3)[idx].cpu().numpy(), wavelengths=wls[idx].cpu().numpy(),
                 t=times[idx].cpu().numpy(), last_hit_triangles=lht[idx].cpu().numpy(),
                 flags=flags[idx].cpu().numpy().view(np.uint32), weights=weights[idx].cpu().numpy(),
                 evidx=np.asarray(fields_in["evidx"], np.uint32)[rows.cpu().numpy()])
    return rows.cpu().numpy().astype(np.int64), state


def propagate(scene, fields, batch, device="cuda", snapshots=None, trace=None):
    """Replay the recorded propagation of one batch. Returns (fields, report).

    With ``snapshots`` (a list), appends GPUPhotons.propagate(track=True)
    style records: all photons before propagation, then after every step the
    photons that were in that launch's queue (all photons for the first one).
    With ``trace`` (a dict photon -> list), appends (step, cursor, words) of
    those photons after every step (debugging).
    """
    n = batch.nphotons
    as_t = lambda x: torch.from_numpy(np.ascontiguousarray(x).copy()).to(device)
    pos = as_t(np.asarray(fields["pos"], np.float32).reshape(-1))
    dirs = as_t(np.asarray(fields["dir"], np.float32).reshape(-1))
    pols = as_t(np.asarray(fields["pol"], np.float32).reshape(-1))
    wls = as_t(np.asarray(fields["wavelengths"], np.float32))
    times = as_t(np.asarray(fields["t"], np.float32))
    lht = as_t(np.asarray(fields["last_hit_triangles"], np.int32))
    flags_np = np.asarray(fields["flags"], np.uint32)
    live0 = (flags_np & M.TERMINAL_16) == 0
    flags = as_t(flags_np.view(np.int32))
    weights = as_t(np.asarray(fields["weights"], np.float32))
    draws = batch.draws(device)
    offsets = batch.offsets(device)
    cap = max(1, n)
    steps = torch.zeros(cap, dtype=torch.int32, device=device)
    norm = torch.full((cap,), -1, dtype=torch.int32, device=device)
    cursor = torch.zeros(cap, dtype=torch.int64, device=device)
    errors = torch.zeros(cap, dtype=torch.int32, device=device)
    live = torch.from_numpy(live0).to(device)
    starts = [int(x) for x in batch["launch_starts"]]
    nsteps = [int(x) for x in batch["launch_nsteps"]]
    use_weights = int(bool(batch.params.get("use_weights", False)))
    max_steps = int(batch.params.get("max_steps", (starts[-1] + nsteps[-1]) if starts else 0))
    start_mask = np.zeros(max([s + c for s, c in zip(starts, nsteps)] + [0]) + 1, np.int32)
    start_mask[starts] = 1
    start_mask = torch.from_numpy(start_mask).to(device)
    ws = dict(tri=torch.empty(cap, dtype=torch.int32, device=device),
              dist=torch.empty(cap, dtype=torch.float32, device=device),
              surface=torch.empty(cap, dtype=torch.int32, device=device),
              m1=torch.empty(cap, dtype=torch.int32, device=device),
              m2=torch.empty(cap, dtype=torch.int32, device=device),
              normal=torch.empty(cap * 3, dtype=torch.float32, device=device),
              flag=torch.empty(cap, dtype=torch.int32, device=device),
              stack=torch.empty(cap * scene.stack, dtype=torch.int32, device=device))
    count_dev = torch.zeros(1, dtype=torch.int32, device=device)
    scratch_rows = torch.empty(cap, dtype=torch.int32, device=device)
    scratch_count = torch.zeros(1, dtype=torch.int32, device=device)
    steps_done = 0
    s = scene
    tensors = (pos, dirs, pols, wls, times, lht, flags, weights)
    all_rows = torch.arange(n, dtype=torch.int32, device=device)
    if snapshots is not None:
        snapshots.append(_snapshot(n, all_rows, tensors, fields))
    kw = dict(BLOCK=M.BLOCK, num_warps=M.NUM_WARPS)
    for launch_index, (start, count) in enumerate(zip(starts, nsteps)):
        if start != steps_done:
            raise tapefmt.TapeError("launch schedule is not contiguous")
        rows = torch.nonzero(live).flatten().to(torch.int32)
        if len(rows) == 0:
            if launch_index == 0 and snapshots is not None:
                snapshots.append(_snapshot(n, all_rows, tensors, fields))
            break
        count_dev.fill_(len(rows))
        M.exact_renorm_kernel[(triton.cdiv(len(rows), M.BLOCK),)](
            rows, count_dev, len(rows), dirs, pols, steps, norm, start_mask, int(start_mask.numel()), **kw)
        for _ in range(count):
            if len(rows) == 0:
                break
            grid = (triton.cdiv(len(rows), M.BLOCK),)
            count_dev.fill_(len(rows))
            scratch_count.zero_()
            M.exact_geometry_kernel[grid](rows, count_dev, len(rows), pos, dirs, lht, s.nodes, s.vertices,
                                          s.triangles, s.material_codes, s.planes, s.nplanes, ws["stack"],
                                          ws["tri"], ws["dist"], ws["surface"], ws["m1"], ws["m2"], ws["normal"],
                                          ws["flag"], s.world[0], s.world[1], s.world[2], s.scale,
                                          STACK=s.stack, **kw)
            M.exact_step_kernel[grid](rows, count_dev, len(rows), pos, dirs, pols, wls, times, lht, flags, weights,
                                      steps, ws["tri"], ws["dist"], ws["surface"], ws["m1"], ws["m2"],
                                      ws["normal"], ws["flag"], draws, offsets, cursor, errors,
                                      s.rindex, s.absorption, s.scattering, s.stride, s.wl_start, s.wl_step, s.nw,
                                      s.comp_first, s.num_comp, s.comp_absorption, s.comp_prob, s.comp_wvl_cdf,
                                      s.comp_time_cdf, s.t_start, s.t_step, s.nt,
                                      s.surface_model, s.surface_transmissive, s.surface_thickness,
                                      s.surface["detect"], s.surface["absorb"], s.surface["reemit"],
                                      s.surface["reflect_diffuse"], s.surface["reflect_specular"], s.surface["eta"],
                                      s.surface["k"], s.surface_reemission_cdf,
                                      s.dichroic_first, s.dichroic_count, s.dichroic_angles, s.dichroic_reflect,
                                      s.dichroic_transmit,
                                      s.angular_first, s.angular_count, s.angular_angles, s.angular_transmit,
                                      s.angular_reflect_specular, s.angular_reflect_diffuse,
                                      use_weights, scratch_rows, scratch_count, max_steps, **kw)
            steps_done += 1
            if trace:
                ids = torch.tensor(sorted(trace), dtype=torch.long, device=device)
                snap = _snapshot(n, ids.to(torch.int32), tensors, fields)[1]
                cur = cursor[ids].cpu().numpy()
                words = tapefmt.photon_words(snap)
                for j, pid in enumerate(sorted(trace)):
                    trace[pid].append((steps_done, int(cur[j]), words[j].copy()))
            if snapshots is not None:
                snapshots.append(_snapshot(n, all_rows if (launch_index == 0 and steps_done == 1) else rows,
                                           tensors, fields))
            alive = (flags.index_select(0, rows.long()) & M.TERMINAL_16) == 0
            rows = rows[alive]
        live = torch.zeros(n, dtype=torch.bool, device=device)
        if len(rows):
            live[rows.long()] = True
        steps_done = start + count
    torch.cuda.synchronize()
    out = dict(pos=pos.view(n, 3).cpu().numpy(), dir=dirs.view(n, 3).cpu().numpy(),
               pol=pols.view(n, 3).cpu().numpy(), wavelengths=wls.cpu().numpy(), t=times.cpu().numpy(),
               last_hit_triangles=lht.cpu().numpy(), flags=flags.cpu().numpy().view(np.uint32),
               weights=weights.cpu().numpy(), evidx=np.asarray(fields["evidx"], np.uint32))
    used = cursor[:n].cpu().numpy()
    expected = np.diff(batch["offsets"])
    errs = errors[:n].cpu().numpy()
    report = dict(photons=int(n), launches=len(starts), steps=int(steps_done),
                  draws=int(expected.sum()), draw_count_mismatch=int(np.count_nonzero(used != expected)),
                  harness_errors=int(np.count_nonzero(errs)))
    if report["draw_count_mismatch"]:
        report["first_draw_count_mismatch"] = int(np.flatnonzero(used != expected)[0])
    if report["harness_errors"]:
        report["first_harness_errors"] = np.flatnonzero(errs)[:20].tolist()
    return out, report


def daq(scene, fields, batch, device="cuda"):
    """Replay run_daq for every event that ran it with the engine's exact DAQ kernel.

    Returns ({event: dict(t, q, flags, hit)}, number of events whose photons
    used a different number of draws than recorded).
    """
    if not hasattr(scene, "nchannels"):
        return {}, 0
    from trichroma.tape.scene import check_daq_limits

    check_daq_limits(scene.words)
    bounds = batch.event_bounds
    ran = batch["daq_event_ran"] if "daq_event_ran" in batch else np.zeros(len(bounds) - 1, np.uint8)
    as_t = lambda x, dt: torch.from_numpy(np.ascontiguousarray(np.asarray(x, dt)).copy()).to(device)
    t = as_t(fields["t"], np.float32)
    w = as_t(fields["weights"], np.float32)
    flags = as_t(np.asarray(fields["flags"], np.uint32).view(np.int32), np.int32)
    lht = as_t(fields["last_hit_triangles"], np.int32)
    solid_map = as_t(np.asarray(scene.solid_id_map, np.uint32).view(np.int32), np.int32)
    channel = as_t(scene.channel_of_solid, np.int32)
    ddraws = batch.daq_draws(device)
    doffs = batch.daq_offsets(device)
    tx, ty = as_t(scene.time_cdf[0], np.float32), as_t(scene.time_cdf[1], np.float32)
    qx, qy = as_t(scene.charge_cdf[0], np.float32), as_t(scene.charge_cdf[1], np.float32)
    nch = scene.nchannels
    results = {}
    mism = 0
    for e in range(len(bounds) - 1):
        if not ran[e]:
            continue
        lo, hi = int(bounds[e]), int(bounds[e + 1])
        time_bits = torch.full((nch,), int(np.float32(1e9).view(np.int32)) ^ -2147483648, dtype=torch.int32,
                               device=device)
        q_int = torch.zeros(nch, dtype=torch.int32, device=device)
        hist = torch.zeros(nch, dtype=torch.int32, device=device)
        errors = torch.zeros(1, dtype=torch.int32, device=device)
        if hi > lo:
            M.exact_daq_kernel[(triton.cdiv(hi - lo, 128),)](
                lo, hi - lo, t, w, flags, lht, solid_map, channel, ddraws, doffs, tx, ty, len(scene.time_cdf[0]),
                qx, qy, len(scene.charge_cdf[0]), scene.charge_unit, 1.0, time_bits, q_int, hist, errors, BLOCK=128)
        mism += int(errors.item() != 0)
        q = torch.empty(nch, dtype=torch.float32, device=device)
        M.charge_float_kernel[(triton.cdiv(nch, 128),)](q_int, scene.charge_unit, q, nch, BLOCK=128)
        tt = (time_bits.cpu().numpy() ^ np.int32(-2147483648)).view(np.float32).copy()
        results[e] = dict(t=tt, q=q.cpu().numpy(), flags=hist.cpu().numpy().view(np.uint32).copy(), hit=tt < 1e8)
    return results, mism


def _compare_words(a, b):
    wa = tapefmt.photon_words(a)
    wb = tapefmt.photon_words(b)
    diff = np.flatnonzero(np.any(wa != wb, axis=1))
    if not len(diff):
        return None
    r = int(diff[0])
    c = int(np.flatnonzero(wa[r] != wb[r])[0])
    return dict(photon=r, word=tapefmt.WORD_NAMES[c], expected=hex(int(wa[r, c])), actual=hex(int(wb[r, c])),
                differing_photons=int(len(diff)))


def _compare_tracks(batch, snapshots):
    """photon_tracks: every recorded step against the replay (ids compared as sets;
    the recorded order is the launch queue order)."""
    offsets = batch["track_step_offsets"]
    ids = batch["track_ids"]
    fields = batch.fields("track")
    if len(offsets) - 1 != len(snapshots):
        return False, "recorded %d track steps, replay produced %d" % (len(offsets) - 1, len(snapshots))
    for step, (rows, state) in enumerate(snapshots):
        a, b = int(offsets[step]), int(offsets[step + 1])
        rec_ids = np.asarray(ids[a:b], np.int64)
        order = np.argsort(rec_ids, kind="stable")
        if not np.array_equal(rec_ids[order], np.sort(np.asarray(rows, np.int64))):
            return False, "step %d: different photons" % step
        rec = {k: np.asarray(v[a:b])[order] for k, v in fields.items()}
        mine_order = np.argsort(np.asarray(rows, np.int64), kind="stable")
        mine = {k: np.asarray(v)[mine_order] for k, v in state.items()}
        d = _compare_words(rec, mine)
        if d is not None:
            d["step"] = step
            return False, d
    return True, None


def replay_tape(directory, device="cuda"):
    """Replay every batch of a tape; compare with its recorded outputs."""
    tape = tapefmt.Tape(directory)
    report = dict(tape=directory, batches=[], equal=True)
    scenes = {}
    for batch in tape:
        t0 = time.time()
        if batch.simulation not in scenes:
            scenes[batch.simulation] = DeviceScene(tape.scene(batch.simulation), device)
        scene = scenes[batch.simulation]
        info = tape.simulations[batch.simulation]
        tracked = "track_step_offsets" in batch
        snapshots = [] if tracked else None
        final, rep = propagate(scene, batch.inputs(), batch, device, snapshots=snapshots)
        rep["seconds"] = time.time() - t0
        if tracked:
            rep["tracks_equal"], rep["tracks_first_difference"] = _compare_tracks(batch, snapshots)
        rep["final_equal"] = None
        expected_final = batch.final()
        if expected_final is not None:
            d = _compare_words(expected_final, final)
            rep["final_equal"] = d is None
            rep["final_first_difference"] = d
        packed = bool(info.get("use_packed", False))
        visible = dict(final)
        if packed:
            inputs = batch.inputs()
            for key in ("pos", "dir", "pol", "wavelengths", "t", "weights"):
                visible[key] = inputs[key]
        if "end_pos" in batch:
            d = _compare_words(batch.fields("end"), visible)
            rep["photons_end_equal"] = d is None
            rep["photons_end_first_difference"] = d
        if "hits_expected_pos" in batch and hasattr(scene, "channel_of_solid"):
            hits = tapefmt.photon_order_hits(final, scene.solid_id_map, scene.channel_of_solid, batch.event_bounds)
            exp = {k[len("hits_expected_"):]: batch[k] for k in batch.arrays if k.startswith("hits_expected_")}
            same = len(hits["photon"]) == len(exp["photon"]) and np.array_equal(hits["photon"], exp["photon"])
            d = _compare_words(exp, hits) if same else "different hit photons"
            same = same and d is None and np.array_equal(hits["channel"], exp["channel"])
            rep["hits_equal"] = bool(same)
            rep["hits"] = int(len(exp["photon"]))
            if not same:
                rep["hits_first_difference"] = d
        if "channels_t" in batch:
            params = batch.params
            daq_fields = dict(final)
            if packed and not (params.get("keep_hits") or params.get("keep_flat_hits")):
                inputs = batch.inputs()
                daq_fields["t"] = inputs["t"]
                daq_fields["weights"] = inputs["weights"]
            channels, dmism = daq(scene, daq_fields, batch)
            ok = dmism == 0
            for row, e in enumerate(batch["channels_event"]):
                got = channels.get(int(e))
                for key in ("t", "q", "flags", "hit"):
                    exp = batch["channels_" + key][row]
                    val = got[key] if got is not None else None
                    if val is None or np.asarray(exp).astype(np.asarray(val).dtype).tobytes() != np.asarray(val).tobytes():
                        ok = False
            rep["channels_equal"] = bool(ok)
            rep["daq_draw_count_mismatch"] = int(dmism)
        rep["equal"] = bool(all(rep.get(k, True) in (True, None) for k in
                                ("final_equal", "photons_end_equal", "hits_equal", "channels_equal",
                                 "tracks_equal"))
                            and rep["draw_count_mismatch"] == 0 and rep["harness_errors"] == 0)
        report["batches"].append(rep)
        report["equal"] = report["equal"] and rep["equal"]
    return report
