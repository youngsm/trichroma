"""Local, synchronized performance measurements for all three optical paths.

Run each backend in a separate process with no other GPU jobs from this
validation running. Every size has a discarded warm-up and repeated measured
runs. Disk I/O and detector setup are outside event latency. Setup/warm-up are
reported separately; on-disk compiler caches are retained and disclosed.
"""
import argparse
import dataclasses
import gc
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np


def cuda_geometry(scene):
    """Reconstruct Chroma's upload interface from the exact shared host scene."""
    from chroma.geometry import Material, Surface
    from pycuda import gpuarray as ga
    h=scene.host
    grid=h.optics.wavelength_grid.values
    materials=[]
    for row,name in enumerate(h.optics.materials.names):
        m=Material(name)
        for field in ("refractive_index","absorption_length","scattering_length"):
            m.set(field,getattr(h.optics.materials,field)[row],grid)
        materials.append(m)
    surfaces=[]
    for row,name in enumerate(h.optics.surfaces.names):
        if not h.optics.surfaces.present[row]:
            surfaces.append(None)
            continue
        s=Surface(name,model=int(h.optics.surfaces.model[row]))
        for field in ("detect","absorb","reflect_diffuse","reflect_specular","reemit","reemission_cdf"):
            s.set(field,getattr(h.optics.surfaces,field)[row],grid)
        surfaces.append(s)
    return SimpleNamespace(mesh=SimpleNamespace(vertices=h.vertices,triangles=h.triangles),
        unique_materials=materials,unique_surfaces=surfaces,material1_index=h.material1_index,
        material2_index=h.material2_index,surface_index=h.surface_index,solid_id=h.solid_id,
        colors=np.zeros(h.triangle_count,np.uint32),wireplanes=[],
        bvh=SimpleNamespace(nodes=np.array(scene.bvh.nodes).view(ga.vec.uint4).reshape(-1),
            world_coords=SimpleNamespace(world_origin=scene.bvh.world_origin,world_scale=scene.bvh.world_scale)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=["cuda","spectral","optimized","full"],required=True)
    parser.add_argument("--scene",default="chroma-lar/benchmarks/optical_validation/original_cuda/detector.input.pkl")
    parser.add_argument("--calibration",default="chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration.json")
    parser.add_argument("--counts",nargs="+",type=int,default=[1000,10000,100000])
    parser.add_argument("--repeats",type=int,default=3)
    parser.add_argument("--max-steps",type=int,default=2048)
    parser.add_argument("--tile-size",type=int,default=65536)
    parser.add_argument("--cuda-threads",type=int,default=512)
    parser.add_argument("--cuda-blocks",type=int,default=1024)
    parser.add_argument("--cuda-native",action="store_true",help="use the original detector's analytic wires for production CUDA throughput")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    from validate_triton_backend import _source_photons
    setup=time.perf_counter()
    context=None
    if args.backend in ("cuda","spectral"):
        with Path(args.scene).open("rb") as stream:
            scene,_,_,_=pickle.load(stream)
    if args.backend=="cuda":
        from chroma import gpu
        import pycuda.driver as cuda
        if os.environ.get("PYCUDA_CACHE_DIR"):
            Path(os.environ["PYCUDA_CACHE_DIR"]).mkdir(parents=True,exist_ok=True)
        context=gpu.create_cuda_context()
        if args.cuda_native:
            from chroma_lar.geometry.config_loader import build_detector_from_config
            from chroma.triton.bvh import build_packed_bvh
            from pycuda import gpuarray as ga
            native=build_detector_from_config("detector_config_reflect_reflect3wires",analytic_wires=True,flatten=True)
            bvh=build_packed_bvh(native.mesh.vertices,native.mesh.triangles)
            native.bvh=SimpleNamespace(nodes=np.array(bvh.nodes).view(ga.vec.uint4).reshape(-1),
                world_coords=SimpleNamespace(world_origin=bvh.world_origin,world_scale=bvh.world_scale))
            sim=gpu.GPUGeometry(native,wavelengths=scene.host.optics.wavelength_grid.values)
        else:
            sim=gpu.GPUGeometry(cuda_geometry(scene),wavelengths=scene.host.optics.wavelength_grid.values)
        # Match Chroma Simulation's persistent XORWOW state and default launch
        # configuration. Initialization belongs to setup, not every event.
        random=gpu.get_rng_states(args.cuda_threads*args.cuda_blocks,seed=12345)
        sync=context.synchronize
        device=context.get_device().name()
    else:
        import torch,triton
        sync=torch.cuda.synchronize
        device=torch.cuda.get_device_name()
        if args.backend=="optimized":
            from chroma_lar.triton_backend import Reflect3WiresTritonSimulation
            sim=Reflect3WiresTritonSimulation()
        elif args.backend=="spectral":
            from chroma.triton.spectral import SpectralSimulation
            sim=SpectralSimulation(scene,tile_size=args.tile_size)
        else:
            from chroma_lar.optical_calibration import OpticalCalibration
            from chroma.triton.spectral import SpectralSimulation
            from chroma.triton.optical_response import digitize
            calibration=OpticalCalibration.load(args.calibration)
            geometry=calibration.build_detector("detector_config_reflect_reflect3wires")
            sim=SpectralSimulation(geometry,wavelengths=calibration.wavelengths,tile_size=args.tile_size)
    sync()
    setup_seconds=time.perf_counter()-setup
    report={"backend":args.backend,"device":device,"setup_seconds":setup_seconds,"counts":[],
            "center_mm":[-1000,0,0],"voxel_size_mm":30,"max_steps":args.max_steps,
            "tile_size":args.tile_size if args.backend in ("spectral","full") else None,
            "disk_compiler_cache":"retained; setup and warm-ups are not cold-compile measurements",
            "scope":"source generation through host-readable output; no file writes or detector construction",
            "output_contract":"full photon states" if args.backend in ("cuda","spectral") else "compact optical hits" if args.backend=="optimized" else "full optical states, PE response and digitized waveforms",
            "physics":"450nm legacy optical tables" if args.backend!="full" else "synthetic VUV spectrum, TPB spectrum/delay, group velocity, TTS, charge and electronics",
            "transport_stage":"synchronized wall time including GPU kernels and host scheduling, excludes source, upload and download"}
    if args.backend in ("cuda","spectral"):
        report.update(triangles=scene.host.triangle_count,channels=scene.host.channel_count,scene_fingerprint=scene.fingerprint)
    elif args.backend=="full":
        report.update(triangles=sim.scene.host.triangle_count,channels=sim.scene.host.channel_count,scene_fingerprint=sim.scene.fingerprint)
    else:
        report["geometry_scope"]="existing instance/wire production specialization; source law matched, no VUV/WLS support"
    if args.backend=="cuda":
        report.update(cuda_threads=args.cuda_threads,cuda_blocks=args.cuda_blocks,
                      transport_rng="persistent XORWOW, setup seed 12345, advanced across warm-up and measured runs")
        if args.cuda_native:
            report.update(geometry_scope="original production detector with analytic wires",triangles=len(native.mesh.triangles),
                          scene_fingerprint=None)
    def run(count,seed):
        gc.collect()
        sync()
        start=time.perf_counter()
        if args.backend=="optimized":
            result=sim.simulate(count,(-1000,0,0),voxel_size=30,seed=seed,max_steps=args.max_steps)
            times,channels=result.flat_hits.to_numpy()
            sync()
            elapsed=time.perf_counter()-start
            return {"event_seconds":elapsed,"detected":len(times),"photons":count,
                    "photons_per_second":count/elapsed}
        if args.backend=="full":
            # Match requested photon counts exactly to separate performance
            # from Poisson count fluctuations; Poisson yield is tested elsewhere.
            positions=np.random.default_rng(seed).uniform(-15,15,(count,3))+[-1000,0,0]
            photons=calibration.source.photons(positions,event_indices=11,seed=seed)
        else:
            photons=_source_photons(count,(-1000,0,0),30,seed)
        sourced=time.perf_counter()
        if args.backend=="cuda":
            state=gpu.GPUPhotons(photons)
            sync(); uploaded=time.perf_counter()
            state.propagate(sim,random,nthreads_per_block=args.cuda_threads,max_blocks=args.cuda_blocks,max_steps=args.max_steps)
            sync(); propagated=time.perf_counter()
            final=state.get()
            detected=int(np.count_nonzero(final.flags&4))
            unfinished=int(np.count_nonzero((final.flags&(1|2|4|8|32768))==0))
            sync(); completed=time.perf_counter()
            record={"upload_allocation_seconds":uploaded-sourced,"transport_seconds":propagated-uploaded,
                    "download_result_seconds":completed-propagated,"detected":detected,"step_limit":unfinished}
        else:
            stages=[]
            result=sim.simulate(photons,seed=seed,max_steps=args.max_steps,timings=stages)
            record={key:sum(stage[key] for stage in stages) for key in
                    ("upload_allocation_seconds","transport_seconds","download_result_seconds")}
            record.update(detected=len(result.hits),step_limit=result.step_limit_count,tiles=len(stages))
            if args.backend=="full":
                began_response=time.perf_counter()
                pe=calibration.response.apply(result.hits,seed=seed)
                responded=time.perf_counter()
                wf=digitize(pe,event_indices=[11,99],channel_count=sim.scene.host.channel_count,
                            seed=seed,**calibration.digitizer)
                record.update(pmt_response_seconds=responded-began_response,
                              digitization_seconds=time.perf_counter()-responded,photoelectrons=len(pe.times),
                              waveform_samples=int(wf.samples.size))
            sync(); completed=time.perf_counter()
        elapsed=completed-start
        record.update(source_seconds=sourced-start,event_seconds=elapsed,photons=count,photons_per_second=count/elapsed)
        if record["step_limit"]:
            raise RuntimeError(f"timed run truncated {record['step_limit']} photons")
        return record
    try:
        for count in args.counts:
            warm=run(count,123)
            measured=[run(count,seed) for seed in (901+np.arange(args.repeats)*1009)]
            keys=[key for key in measured[0] if key.endswith("seconds") or key=="photons_per_second"]
            entry={"photons":count,"warmup":warm,"runs":measured,
                   "median":{key:float(np.median([r[key] for r in measured])) for key in keys},
                   "event_seconds_min":min(r["event_seconds"] for r in measured),
                   "event_seconds_max":max(r["event_seconds"] for r in measured)}
            report["counts"].append(entry)
            Path(args.output).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
            print(json.dumps({"backend":args.backend,"photons":count,"median":entry["median"]}),flush=True)
    finally:
        if context is not None:
            context.pop()


if __name__=="__main__":
    main()
