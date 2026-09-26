"""Ray-by-ray comparison of the production geometry (SAH trees, instances, analytic boxes, production wires)
with CUDA Chroma's (exact mode's W BVH and FP32 wires, recorded in the weighted tape's scene), on the rays of
real photon histories (production tracking of weighted W1 and W2 photons on reflect3wires).

usage: geometry_rays.py TAPE_DIR NPHOTONS OUT.npz  (TAPE_DIR: a CUDA Chroma tape of reflect3wires, for its scene)
"""
import sys, time
import numpy as np
import torch
import triton
from chroma.sim import Simulation
from chroma_lar.geometry.config_loader import build_detector_from_config
from trichroma.engine.api import DevicePhotons, TERMINAL
from trichroma.engine import exact_mode as XM
from trichroma.tape.format import Tape

tape_dir, nph, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dev = torch.device("cuda")
g = build_detector_from_config("detector_config_reflect_reflect3wires")
sim = Simulation(g, seed=11)
eng = sim.engine


def source(kind, n, seed):
    rng = np.random.default_rng(seed)
    cos = 2 * rng.random(n) - 1; phi = rng.uniform(0, 2 * np.pi, n); sin = np.sqrt(1 - cos * cos)
    d = np.stack([sin * np.cos(phi), sin * np.sin(phi), cos], 1)
    if kind == "W1":
        pos = np.tile(np.array([-450.0, 60.0, -120.0]), (n, 1)); pol = np.zeros_like(d); pol[:, 0] = 1
    else:
        lo, hi = np.array([-2150.0, -2150.0, -2150.0]), np.array([-10.0, 2150.0, 2150.0])
        pos = lo + (hi - lo) * rng.random((n, 3))
        pol = np.cross(d, rng.normal(size=(n, 3))); pol /= np.linalg.norm(pol, axis=1)[:, None]
    ph = DevicePhotons.empty(n, dev)
    ph.pos.copy_(torch.tensor(pos, dtype=torch.float32)); ph.dir.copy_(torch.tensor(d, dtype=torch.float32))
    ph.pol.copy_(torch.tensor(pol, dtype=torch.float32)); ph.wavelengths.fill_(128.0); ph.t.zero_()
    ph.ids.copy_(torch.arange(n, dtype=torch.int64) + seed * 10**9)
    return ph


pos, dirs, last, kinds = [], [], [], []
for k, kind in enumerate(("W1", "W2")):
    track = eng.propagate(source(kind, nph, 100 + k), max_steps=1000, use_weights=True, track=True)
    for rows, st in track:
        live = (st.flags & TERMINAL) == 0
        pos.append(st.pos[live]); dirs.append(st.dir[live]); last.append(st.last_hit_triangles[live])
        kinds.append(torch.full((int(live.sum()),), k, dtype=torch.int8, device=dev))
pos, dirs, last, kinds = (torch.cat(x).contiguous() for x in (pos, dirs, last, kinds))
n = len(last)
print("rays: %d (from %d W1 + %d W2 photon histories)" % (n, nph, nph), flush=True)

# production nearest boundary (whatever wires this process selected: CHROMA_TRITON_LEGACY_WIRES)
t0 = time.time()
tp, trip, np_, codes = [], [], [], []
for a in range(0, n, 1 << 20):
    r = eng.query(pos[a:a + (1 << 20)], dirs[a:a + (1 << 20)], last[a:a + (1 << 20)])
    tp.append(r[0]); trip.append(r[1]); np_.append(r[2]); codes.append(r[3])
tp, trip, np_, codes = torch.cat(tp), torch.cat(trip), torch.cat(np_), torch.cat(codes)
torch.cuda.synchronize(); tprod = time.time() - t0

# CUDA Chroma's geometry: the exact mode's fill_state geometry on the recorded scene
s = XM.ExactScene(Tape(tape_dir).scene(0), dev)
t0 = time.time()
te, trie, ne, m1e, m2e, se, fe = [], [], [], [], [], [], []
C = 1 << 20
stack = torch.empty(C * s.stack, dtype=torch.int32, device=dev)
for a in range(0, n, C):
    m = min(C, n - a)
    rows = torch.arange(m, dtype=torch.int32, device=dev); cnt = torch.tensor([m], dtype=torch.int32, device=dev)
    o = dict(tri=torch.empty(m, dtype=torch.int32, device=dev), dist=torch.empty(m, dtype=torch.float32, device=dev),
             surf=torch.empty(m, dtype=torch.int32, device=dev), m1=torch.empty(m, dtype=torch.int32, device=dev),
             m2=torch.empty(m, dtype=torch.int32, device=dev), nrm=torch.empty(m * 3, dtype=torch.float32, device=dev),
             flag=torch.empty(m, dtype=torch.int32, device=dev))
    XM.exact_geometry_kernel[(triton.cdiv(m, XM.BLOCK),)](
        rows, cnt, m, pos[a:a + m].contiguous(), dirs[a:a + m].contiguous(), last[a:a + m].contiguous(),
        s.nodes, s.vertices, s.triangles, s.material_codes, s.planes, s.nplanes, stack,
        o["tri"], o["dist"], o["surf"], o["m1"], o["m2"], o["nrm"], o["flag"],
        s.world[0], s.world[1], s.world[2], s.scale, STACK=s.stack, BLOCK=XM.BLOCK, num_warps=XM.NUM_WARPS)
    te.append(o["dist"]); trie.append(o["tri"]); ne.append(o["nrm"].view(m, 3)); m1e.append(o["m1"]); m2e.append(o["m2"])
    se.append(o["surf"]); fe.append(o["flag"])
te, trie, ne, m1e, m2e, se, fe = (torch.cat(x) for x in (te, trie, ne, m1e, m2e, se, fe))
torch.cuda.synchronize(); texact = time.time() - t0
print("query time: production %.2f s, exact (W) %.2f s" % (tprod, texact))

h = lambda x: x.cpu().numpy()
facing = (np_ * -dirs).sum(1) > 0
m1p = torch.where(facing, codes[:, 1], codes[:, 0]); m2p = torch.where(facing, codes[:, 0], codes[:, 1])
np.savez(out, pos=h(pos), dir=h(dirs), last=h(last), kind=h(kinds), t_prod=h(tp), tri_prod=h(trip), n_prod=h(np_),
         m1_prod=h(m1p), m2_prod=h(m2p), s_prod=h(codes[:, 2]), t_w=h(te), tri_w=h(trie), n_w=h(ne), m1_w=h(m1e),
         m2_w=h(m2e), s_w=h(se), flag_w=h(fe))
print("saved", out)
