"""One weighted simulate() call on chroma-lar's reflect3wires detector with either backend; saves every photon's end state.

usage: weighted_run.py WORKLOAD SEED N OUT.npz  (backend: CHROMA_BACKEND; engine options: CHROMA_TRITON_*)
  W1: the LUT fixture (generate_lut.py): N photons at (-450, 60, -120), isotropic, polarization +x, 128 nm
  W2: N photons uniform in the simulated half detector (x -2310..0, y/z -2160..2160), isotropic, transverse
      random polarization, 128 nm
  W1nw, W2nw: the same on the detector built without its wire planes (include_wires=False)
The source photons depend only on (WORKLOAD, SEED), so both backends see the same photons for the same seed.
"""
import os, sys, time
import numpy as np
workload, seed, N, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
from chroma.backend import backend_name
from chroma.event import Photons
from chroma.sim import Simulation
from chroma_lar.geometry.config_loader import build_detector_from_config

rng = np.random.default_rng(1000 + seed)
cos = 2 * rng.random(N) - 1
phi = rng.uniform(0, 2 * np.pi, N)
sin = np.sqrt(1 - cos * cos)
d = np.stack([sin * np.cos(phi), sin * np.sin(phi), cos], 1)
wires = not workload.endswith("nw")
if workload.startswith("W1"):
    pos = np.tile(np.array([-450.0, 60.0, -120.0]), (N, 1))
    pol = np.zeros_like(d); pol[:, 0] = 1
else:
    lo, hi = np.array([-2310.0, -2160.0, -2160.0]), np.array([0.0, 2160.0, 2160.0])
    pos = lo + (hi - lo) * rng.random((N, 3))
    pol = np.cross(d, rng.normal(size=(N, 3)))
    pol /= np.linalg.norm(pol, axis=1)[:, None]
f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)
photons = Photons(f32(pos), f32(d), f32(pol), wavelengths=np.full(N, 128.0, np.float32), t=np.zeros(N, np.float32))
g = build_detector_from_config("detector_config_reflect_reflect3wires", **({} if wires else {"include_wires": False}))
t0 = time.time()
sim = Simulation(g, seed=seed)
ev = next(sim.simulate([photons], keep_photons_end=True, keep_flat_hits=True, keep_hits=False, max_steps=1000,
                       use_weights=True))
elapsed = time.time() - t0
pe, fh = ev.photons_end, ev.flat_hits
flags = np.asarray(pe.flags).astype(np.uint32)
detected = (flags & 4) != 0
# triangle -> channel from the hits (every weighted detection ends its photon: one hit per detected photon)
tri_channel = np.full(int(max(pe.last_hit_triangles.max(), fh.last_hit_triangles.max())) + 1, -1, np.int32)
tri_channel[fh.last_hit_triangles] = fh.channel
channel = np.where(detected, tri_channel[np.maximum(pe.last_hit_triangles, 0)], -1)
check = (int(detected.sum()) == len(fh.weights), float(pe.weights[detected].astype(np.float64).sum()),
         float(np.asarray(fh.weights, np.float64).sum()))
np.savez(out, flags=flags, weights=np.asarray(pe.weights, np.float32), t=np.asarray(pe.t, np.float32),
         channel=channel.astype(np.int16), last=np.asarray(pe.last_hit_triangles, np.int32), elapsed=elapsed,
         backend=backend_name(), env=" ".join("%s=%s" % (k, v) for k, v in sorted(os.environ.items())
                                              if k.startswith("CHROMA_TRITON")))
print("%s %s seed %d N %d: %.1fs detected %d, weight/photon %.6f, hits==detected %s (sum %.3f vs %.3f) [%s]" % (
    backend_name(), workload, seed, N, elapsed, detected.sum(), check[1] / N, check[0], check[1], check[2],
    " ".join("%s=%s" % (k, v) for k, v in sorted(os.environ.items()) if k.startswith("CHROMA_TRITON"))), flush=True)
