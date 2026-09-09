"""Exercise all optical stages in the repository's full, meshed detector.

The generated bundle is explicitly synthetic. Legacy wall/wire optics are
copied only as software-test inputs; LAr, TPB, glass and PMT values below are
controlled test parameters, not a detector calibration.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from chroma.geometry import Surface
from chroma_lar.geometry.config_loader import build_detector_from_config
from chroma_lar.optical_calibration import OpticalCalibration
from chroma_lar.optical_simulation import OpticalSimulation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--energy-mev",type=float,default=5.)
    args=parser.parse_args()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    config_name="detector_config_reflect_reflect3wires"
    coat=Surface("validation_tpb",model=2)
    geometry=build_detector_from_config(config_name,analytic_wires=False,flatten=False,pmt_coating_surface=coat)
    raw_materials={m.name:m for s in geometry.solids for m in np.r_[s.material1,s.material2] if m is not None}
    raw_surfaces={s.name:s for solid in geometry.solids for s in solid.surface if s is not None and s is not coat}
    prototype=json.loads((Path(__file__).parents[1]/"examples/synthetic_optical_calibration.json").read_text())
    grid=np.arange(120,501,dtype=float)
    prototype["provenance"]="Synthetic full-detector software validation; not measured. Wall and wire tables copied from repository configuration."
    prototype["materials"]={name:{field:np.column_stack((grid,np.interp(grid,getattr(m,field)[:,0],getattr(m,field)[:,1]))).tolist()
        for field in ("refractive_index","absorption_length","scattering_length")} for name,m in raw_materials.items()}
    prototype["surfaces"]={name:{field:np.column_stack((grid,np.interp(grid,getattr(s,field)[:,0],getattr(s,field)[:,1]))).tolist()
        for field in ("detect","absorb","reflect_diffuse","reflect_specular")} for name,s in raw_surfaces.items()}
    # Resolve roles from the detector configuration itself.
    from chroma_lar.config.detector_config_reflect_reflect3wires import get_config
    config=get_config()
    lar,glass,sensor=(config["target_material"].name,config["pmt_glass_material"].name,config["pmt_photocathode_surface"].name)
    prototype["roles"]={"lar":lar,"glass":glass,"photocathode":sensor,"tpb":"validation_tpb"}
    prototype["materials"][lar]={"refractive_index":[[120,1.4],[160,1.3],[400,1.23],[500,1.23]],
        "group_velocity":[[120,135],[160,145],[400,215],[500,220]],"absorption_length":100000,"scattering_length":950}
    prototype["materials"][glass]={"refractive_index":1.52,
        "absorption_length":[[120,.001],[200,.001],[201,1500],[500,1500]],"scattering_length":1e30}
    prototype["surfaces"][sensor]={"detect":[[120,0],[200,0],[201,.25],[500,.25]],
                                   "absorb":[[120,1],[200,1],[201,.75],[500,.75]]}
    demo=json.loads((Path(__file__).parents[1]/"examples/synthetic_optical_calibration.json").read_text())
    prototype["surfaces"]["validation_tpb"]=demo["surfaces"]["demo_tpb"]
    prototype["pmt_response"]["collection_efficiency"]=.9
    prototype["digitizer"]["noise_rms"]=0
    calibration_path=output/"full_detector_synthetic_calibration.json"
    calibration_path.write_text(json.dumps(prototype,indent=2)+"\n")
    del geometry,raw_materials,raw_surfaces
    calibration=OpticalCalibration.load(calibration_path)
    geometry=calibration.build_detector(config_name)
    simulation=OpticalSimulation(geometry,calibration,backend="triton")
    # x=0 is inside the steel cathode. Place the deposition in liquid argon.
    result=simulation.simulate_depositions([[-1000,0,0],[-1000,0,0]],[args.energy_mev,0],event_indices=[11,99],seed=1729,max_steps=2048)
    s=result.transport.final_state
    print(json.dumps(result.metadata),flush=True)
    assert result.transport.step_limit_count==0
    assert np.all(result.transport.hits.wavelengths>200)
    assert len(result.photoelectrons.times)>20
    assert np.all((s["flags"][(s["flags"]&4)!=0]&128)!=0)
    np.testing.assert_array_equal(result.waveforms.event_indices,[11,99])
    np.testing.assert_array_equal(result.waveforms.samples[1],prototype["digitizer"]["baseline"])
    assert np.all(result.photoelectrons.event_indices==11)
    result.save(output/"full_detector_vuv.npz")
    summary={**result.metadata,"triangle_count":simulation.transport.scene.host.triangle_count,
             "channels":simulation.transport.scene.host.channel_count,
             "channels_with_pe":int(len(np.unique(result.photoelectrons.channels))),
             "reemitted_photons":int(np.count_nonzero(s["flags"]&128)),
             "all_detected_photons_reemitted":True,"zero_energy_event_preserved":True,
             "optical_time_quantiles_ns":np.quantile(result.transport.hits.times,[.01,.1,.5,.9,.99]).tolist(),
             "waveform_shape":list(result.waveforms.samples.shape)}
    (output/"full_detector_pipeline.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
