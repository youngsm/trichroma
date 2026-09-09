"""Isolate geometric wire shadowing in original CUDA mesh/analytic modes."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--count",type=int,default=100000)
    parser.add_argument("--reflect",action="store_true",help="lossless reflective wires: all photons should eventually reach the outer sensor")
    parser.add_argument("--spectral",action="store_true",help="probe the spectral Triton mesh path")
    args=parser.parse_args()
    from chroma.detector import Detector
    from chroma.event import Photons
    from chroma.geometry import Solid
    from chroma.make import box
    from chroma.triton.bvh import build_packed_bvh
    from chroma_lar.geometry.wireplane import make_wire_plane,make_wireplane,attach_wireplanes
    from optical_comparison_cases import material,surface,GRID
    if args.spectral:
        from chroma.triton.spectral import SpectralSimulation
        context=None
    else:
        from chroma import gpu
        from pycuda import gpuarray as ga
        Path(os.environ.get("PYCUDA_CACHE_DIR","/tmp/chroma-wire-probe")).mkdir(parents=True,exist_ok=True)
        context=gpu.create_cuda_context()
    records=[]
    try:
        for wire_angle in (90,60,-60):
            angle=np.radians(wire_angle)
            for incidence in (0,60,85):
                theta=np.radians(incidence)
                rng=np.random.default_rng(417)
                pos=np.column_stack((np.full(args.count,-1.),rng.uniform(-24,24,(args.count,2))))
                photons=Photons(pos,np.tile([np.cos(theta),np.sin(theta),0],(args.count,1)),
                    np.tile([0,0,1],(args.count,1)),np.full(args.count,450.))
                masks={}
                outcomes={}
                for kind in (("mesh",) if args.spectral else ("mesh","analytic")):
                    lar,steel=material("lar"),material("steel")
                    steel.set("absorption_length",0.)
                    steel.set("scattering_length",0.)
                    black=surface("wire",**({"reflect_specular":1} if args.reflect else {"absorb":1}))
                    sensor=surface("sensor",detect=1)
                    detector=Detector(lar)
                    detector.add_pmt(Solid(box(40,100,100),lar,lar,surface=sensor),displacement=np.zeros(3))
                    if kind=="mesh":
                        for mesh in make_wire_plane(100,100,3,angle,.15,nsteps=32):
                            detector.add_solid(Solid(mesh,steel,lar,surface=black))
                    else:
                        width=100*(abs(np.cos(angle))+abs(np.sin(angle)))
                        attach_wireplanes(detector,[make_wireplane(center=(0,0,0),u_dir=(0,np.cos(angle),np.sin(angle)),
                            plane_normal=(1,0,0),pitch=3,radius=.075,length=width,width=width,
                            surface=black,material_inner=steel,material_outer=lar)])
                    detector.flatten()
                    if kind=="analytic":
                        # This isolated plane has no cathode sharing its
                        # surface/material, unlike the production detector.
                        detector.unique_surfaces.append(black)
                        detector.unique_materials.append(steel)
                    if args.spectral:
                        result=SpectralSimulation(detector,wavelengths=GRID).simulate(photons,seed=891,max_steps=100)
                        final=SimpleNamespace(flags=result.final_state["flags"],dir=result.final_state["direction"])
                    else:
                        bvh=build_packed_bvh(detector.mesh.vertices,detector.mesh.triangles)
                        detector.bvh=SimpleNamespace(nodes=np.array(bvh.nodes).view(ga.vec.uint4).reshape(-1),
                            world_coords=SimpleNamespace(world_origin=bvh.world_origin,world_scale=bvh.world_scale))
                        device=gpu.GPUGeometry(detector,wavelengths=GRID)
                        state=gpu.GPUPhotons(photons)
                        random=gpu.get_rng_states(128*128,seed=891)
                        state.propagate(device,random,nthreads_per_block=128,max_blocks=128,max_steps=100)
                        final=state.get()
                    masks[kind]=(final.flags&(64 if args.reflect else 10))!=0
                    outcomes[kind]={"bulk_absorbed":int(np.count_nonzero(final.flags&2)),
                        "detected":int(np.count_nonzero(final.flags&4)),
                        "nonfinite":int(np.count_nonzero(~np.isfinite(final.dir).all(axis=1)))}
                    if not args.spectral:
                        del state,device,random
                expected=.05*np.sqrt(1+(np.tan(theta)*np.sin(angle))**2)
                record={"wire_angle_deg":wire_angle,"incidence_deg":incidence,"count":args.count,
                        "reflective":args.reflect,"outcomes":outcomes,"backend":"triton" if args.spectral else "cuda",
                        "expected_cylinder_shadow":float(expected),
                        "mesh_shadow":float(masks["mesh"].mean()),
                        "analytic_shadow":float(masks["analytic"].mean()) if "analytic" in masks else None,
                        "same_ray_shadow_disagreements":int(np.count_nonzero(masks["mesh"]!=masks["analytic"])) if "analytic" in masks else None}
                records.append(record)
                print(json.dumps(record),flush=True)
                lossless_passed=all(v["bulk_absorbed"]==0 and v["detected"]==r["count"]
                    for r in records for v in r["outcomes"].values()) if args.reflect else None
                Path(args.output).write_text(json.dumps({"cases":records,
                    "lossless_invariant_passed":lossless_passed,
                    "lossless_expected":"Every input photon reaches the enclosing sensor; zero bulk absorption." if args.reflect else None},indent=2)+"\n")
    finally:
        if context is not None:
            context.pop()
    if args.reflect and not lossless_passed:
        raise SystemExit(1)


if __name__=="__main__":
    main()
