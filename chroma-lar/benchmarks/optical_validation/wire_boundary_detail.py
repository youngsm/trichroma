"""Replay the locally generated trusted wire-boundary diagnostic artifact."""
import json
from pathlib import Path
import pickle
import numpy as np
import torch
from chroma.triton.spectral import SpectralSimulation
from chroma.triton.bvh_kernels import nearest_hit_local
with Path(__file__).with_name('wire_boundary_input.pkl').open('rb') as f:
    scene,photons,bad,final=pickle.load(f)
sim=SpectralSimulation(scene)
p=photons[bad]
state=sim.simulate(p,max_steps=1,seed=891).final_state
v=scene.host.vertices[scene.host.triangles].astype(np.float64)
e1=v[:,1]-v[:,0];e2=v[:,2]-v[:,0]
def intersect(o,d,skip=-1):
    h=np.cross(d,e2);det=np.sum(e1*h,axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        r=1/det;s=o-v[:,0];u=r*np.sum(s*h,axis=1)
        q=np.cross(s,e1);w=r*np.sum(d*q,axis=1);t=r*np.sum(e2*q,axis=1)
    with np.errstate(invalid='ignore'):
        ok=(abs(det)>1e-12)&(u>=-1e-10)&(w>=-1e-10)&(u+w<=1+1e-10)&(t>1e-10)
    if skip>=0:ok[skip]=False
    ix=np.flatnonzero(ok);ix=ix[np.argsort(t[ix])]
    return [(int(i),float(t[i]),float(u[i]),float(w[i])) for i in ix[:4]]
records=[]
for i in [0,1,2,*np.flatnonzero(state['flags']==2)[:2]]:
    n=nearest_hit_local(sim._device.bvh,torch.from_numpy(state['pos'][i:i+1]).cuda(),torch.from_numpy(state['direction'][i:i+1]).cuda(),last_hit=torch.from_numpy(state['last_hit'][i:i+1]).cuda())
    print('photon',int(bad[i]),'input',p.pos[i],p.dir[i],'first GPU',state['last_hit'][i],state['pos'][i],'first64',intersect(p.pos[i].astype(float),p.dir[i].astype(float)),flush=True)
    print('next GPU',n.triangle_ids.cpu().numpy(),n.distances.cpu().numpy(),'next64',intersect(state['pos'][i].astype(float),state['direction'][i].astype(float),int(state['last_hit'][i])),flush=True)

    records.append({"photon_index":int(bad[i]),"input_position":p.pos[i].tolist(),"input_direction":p.dir[i].tolist(),
        "first_step_flags":int(state['flags'][i]),"first_step_position":state['pos'][i].tolist(),
        "first_step_direction":state['direction'][i].tolist(),"first_step_last_hit":int(state['last_hit'][i]),
        "first_step_float64_intersections":intersect(p.pos[i].astype(float),p.dir[i].astype(float)),
        "next_gpu_triangle":int(n.triangle_ids.cpu().numpy()[0]),"next_gpu_distance":float(n.distances.cpu().numpy()[0]),
        "next_float64_intersections":intersect(state['pos'][i].astype(float),state['direction'][i].astype(float),int(state['last_hit'][i]))})
Path(__file__).with_name('wire_boundary_detail.json').write_text(json.dumps({
    "wire_angle_degrees":-60,"incidence_degrees":85,"input_photons":len(photons.pos),
    "bulk_absorbed":len(bad),"absorbed_before_reflection":int(np.count_nonzero(state['flags']==2)),
    "absorbed_after_first_reflection":int(np.count_nonzero(state['flags']!=2)),"selected_examples":records},indent=2)+'\n')
