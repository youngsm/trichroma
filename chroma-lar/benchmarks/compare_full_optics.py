"""Quantitative optical validation on a local CUDA GPU.

Analytic checks use six standard errors (binomial) or DKW alpha=1e-6
(distribution tests). CUDA ensembles use independent RNGs and a two-sample
DKW bound; CPU/Triton runs use common Philox counters. No measured detector
calibration is asserted by this synthetic/software comparison.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import time
import numpy as np

from optical_comparison_cases import CASES, GRID, make_case
from chroma.triton.spectral import SpectralSimulation
from chroma.triton.optical_response import OpticalHits, PMTResponse, TabulatedCDF, digitize, uniform
from chroma_lar.generator.scintillation import ScintillationSource


ALPHA = 1.e-6


def cdf_distance(samples, cdf):
    x = np.sort(np.asarray(samples, float))
    f = cdf(x)
    return float(max(np.max(np.arange(1,len(x)+1)/len(x)-f), np.max(f-np.arange(len(x))/len(x))))


def two_cdf_distance(a, b, tolerance=0.):
    a,b = np.sort(a),np.sort(b)
    if not len(a) or not len(b):
        return float(bool(len(a) or len(b)))
    # Allow a stated numerical tolerance in the coordinate being compared.
    d1 = np.arange(1,len(a)+1)/len(a)-np.searchsorted(b,a+tolerance,side="right")/len(b)
    d2 = np.arange(1,len(b)+1)/len(b)-np.searchsorted(a,b+tolerance,side="right")/len(a)
    return float(max(0,np.max(d1),np.max(d2)))


class Report:
    def __init__(self, path):
        import torch, triton
        self.path = Path(path)
        self.data = {"device":torch.cuda.get_device_name(), "torch":torch.__version__, "triton":triton.__version__,
                     "rng":"philox4x32-10-id64-stream32-seed64-v1", "dkw_alpha_per_check":ALPHA,
                     "checks":[], "cases":[], "physical_calibration":"not validated; measured inputs unavailable"}

    def check(self, name, value, limit, **detail):
        entry = {"name":name,"value":float(value),"limit":float(limit),"passed":bool(value <= limit), **detail}
        self.data["checks"].append(entry)
        if not entry["passed"]:
            print("FAILED "+json.dumps(entry),flush=True)

    def fraction(self, name, selected, expected):
        n = len(selected)
        actual = float(np.mean(selected))
        self.check(name, abs(actual-expected), 6*np.sqrt(expected*(1-expected)/n)+3/n,
                   observed=actual, expected=float(expected), count=n)

    def cdf(self, name, values, law):
        self.check(name, cdf_distance(values,law), np.sqrt(np.log(2/ALPHA)/(2*len(values))), count=len(values))

    def save(self):
        self.data["passed"] = all(c["passed"] for c in self.data["checks"])
        self.data["check_count"] = len(self.data["checks"])
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text(json.dumps(self.data,indent=2,sort_keys=True)+"\n")


def analytical_transport(report, name, s, expected):
    flags=s["flags"]
    prefix=name+".analytic"
    if name == "ballistic":
        report.check(prefix+".time_ns", np.max(np.abs(s["times"]-(7+100*1.4/299.792458))), 2e-6)
        report.fraction(prefix+".detection", (flags&4)!=0, 1)
    elif name == "absorption":
        for low, length in ((True,100),(False,1000)):
            report.fraction(prefix+f".survival_{length}mm", ((flags&4)!=0)[(s["wavelengths"]<300)==low], np.exp(-100/length))
    elif name == "competing_bulk":
        survival=np.exp(-1.5)
        for bit,p in ((4,survival),(2,(1-survival)/3),(16,2*(1-survival)/3)):
            report.fraction(prefix+f".flag{bit}", (flags&bit)!=0,p)
    elif name == "rayleigh":
        d=s["direction"][(flags&16)!=0]
        report.cdf(prefix+".polarized_cosine",d[:,0],lambda x:.5+.75*x-.25*x**3)
        report.check(prefix+".forward_second_moment",abs(np.mean(d[:,2]**2)-.4),.006)
    elif name == "default_surface":
        for bit,p in ((8,.2),(4,.3),(32,.25),(64,.15)):
            report.fraction(prefix+f".flag{bit}",(flags&bit)!=0,p)
    elif name == "diffuse":
        report.cdf(prefix+".lambert_cosine",-s["direction"][:,2],lambda x:x*x)
    elif name == "specular":
        report.check(prefix+".direction", np.max(np.abs(s["direction"]-[.6,0,-.8])),1e-6)
    elif name.startswith("fresnel") or name in ("tir","critical"):
        report.fraction(prefix+".reflectance",(flags&64)!=0,expected["reflectance"])
        transmitted=(flags&64)==0
        if np.any(transmitted):
            expected_sin=expected["n1"]/expected["n2"]*np.sin(np.radians(expected["angle_deg"]))
            report.check(prefix+".snell",np.max(np.abs(s["direction"][transmitted,0]-expected_sin)),2e-5)
    elif name in ("wls","wls_loss"):
        good=(flags&128)!=0
        report.fraction(prefix+".quantum_yield",good,1 if name=="wls" else .65)
        report.cdf(prefix+".wavelength",s["wavelengths"][good],lambda x:np.clip((x-410)/40,0,1))
        report.cdf(prefix+".isotropic_cosine",s["direction"][good,2],lambda x:(x+1)/2)


def response_checks(report,n):
    ids=np.arange(n)
    for seed in (0,1,2**32+5):
        report.cdf(f"rng.{seed}.uniform",uniform(ids,seed),lambda x:x)
    h=OpticalHits(np.full(n,7.),ids%3,ids,ids%5)
    response=PMTResponse(tts_sigma=[.5,2,5], transit_time=[20,30,40],collection_efficiency=[1,.75,.4],
                         charge_cdf=TabulatedCDF.from_pdf([0,1,2],[0,1,0]))
    pe=response.apply(h,seed=45)
    from scipy.special import ndtr
    for ch,sigma,offset,eff in ((0,.5,20,1),(1,2,30,.75),(2,5,40,.4)):
        mask=pe.channels==ch
        report.cdf(f"tts.channel{ch}",(pe.times[mask]-7-offset)/sigma,ndtr)
        present=np.isin(h.photon_ids[h.channels==ch],pe.photon_ids[mask])
        report.fraction(f"collection.channel{ch}",present,eff)
    report.cdf("charge.triangular",pe.charges,lambda x:np.where(x<1,.5*x*x,1-.5*(2-x)**2))
    src=ScintillationSource(TabulatedCDF.from_pdf([120,128,136],[0,1,0]),(6,1600),(.3,.7),1000,3)
    p=src.photons(np.zeros((n,3)),seed=193)
    report.cdf("lar.spectrum",p.wavelengths,lambda x:np.where(x<128,(x-120)**2/128,1-(136-x)**2/128))
    # Convolution of independent exponential rise and each exponential decay.
    def time_cdf(t):
        return sum(w*(1-(tau*np.exp(-t/tau)-3*np.exp(-t/3))/(tau-3)) for tau,w in ((6,.3),(1600,.7)))
    report.cdf("lar.rise_and_decay",p.times,time_cdf)
    report.cdf("lar.isotropy",p.direction[:,2],lambda x:(x+1)/2)
    dep=src.from_depositions(np.zeros((10000,3)),.01,event_indices=np.arange(10000),seed=91)
    counts=np.bincount(dep.event_indices,minlength=10000)
    report.check("lar.poisson_mean",abs(counts.mean()-10),6*np.sqrt(10/10000))
    report.check("lar.poisson_variance",abs(counts.var()-10),6*np.sqrt((10+2*100)/10000))
    # Compare finite-window, noninteger arrival waveform against a brute-force
    # sum at every sample, including pulses beginning before the window.
    rng=np.random.default_rng(194)
    small=OpticalHits(rng.uniform(-12,150,301),np.arange(301)%3,np.arange(301),np.arange(301)%4)
    pe=PMTResponse(gain=[1,2,3]).apply(small)
    events=[3,2,1,0,99]
    px,py=[0,1.3,4.2,17],[0,5,2,0]
    args=dict(event_indices=events,channel_count=3,start_ns=0,sample_period_ns=.7,sample_count=200,
              pulse_times_ns=px,pulse_adc_per_pe=py,baseline=7)
    wf=digitize(pe,**args)
    brute=np.full(wf.samples.shape,7.)
    for row,ev in enumerate(events):
        for ch in range(3):
            mask=(pe.event_indices==ev)&(pe.channels==ch)
            for t,q in zip(pe.times[mask],pe.charges[mask]):
                brute[row,ch]+=q*np.interp(wf.sample_times-t,px,py,left=0,right=0)
    report.check("waveform.direct_superposition",np.max(np.abs(wf.samples-brute)),1e-12)
    a=digitize(pe,noise_rms=.7,seed=314,**args)
    args["event_indices"]=events[::-1]
    b=digitize(pe,noise_rms=.7,seed=314,**args)
    report.check("waveform.event_order",np.max(np.abs(a.samples-b.samples[::-1])),0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--cuda-dir")
    parser.add_argument("--count",type=int,default=50000)
    parser.add_argument("--cases",nargs="+",default=CASES)
    parser.add_argument("--seeds",nargs="+",type=int,default=[11,29,83])
    parser.add_argument("--skip-response",action="store_true")
    args=parser.parse_args()
    report=Report(args.output)
    if not args.skip_response:
        response_checks(report,max(args.count,100000))
        report.save()
    for name in args.cases:
        canonical = Path(args.cuda_dir)/f"{name}.input.pkl" if args.cuda_dir else None
        if canonical is not None and canonical.exists():
            # Trusted artifacts produced locally by run_original_optics_comparison.
            with canonical.open("rb") as stream:
                detector,photons,steps,expected=pickle.load(stream)
            if len(photons)!=args.count:
                raise ValueError("exported source size differs from --count")
        else:
            detector,photons,steps,expected=make_case(name,args.count)
        start=time.monotonic()
        sim=SpectralSimulation(detector,wavelengths=GRID,backend="triton")
        host=sim.scene.host
        geometry_hash=hashlib.sha256(host.vertices.tobytes()+host.triangles.tobytes()).hexdigest()
        source_hash=hashlib.sha256(b"".join(getattr(photons,f).tobytes() for f in ("pos","dir","pol","wavelengths","t"))).hexdigest()
        case={"name":name,"triangles":host.triangle_count,"channels":host.channel_count,
              "scene_fingerprint":sim.scene.fingerprint,"geometry_sha256":geometry_hash,"source_sha256":source_hash,"seeds":[]}
        for seed in args.seeds:
            result=sim.simulate(photons,seed=seed,max_steps=steps)
            s=result.final_state
            prefix=f"{name}.{seed}"
            for field in ("pos","direction","polarization","times","wavelengths"):
                report.check(prefix+f".finite_{field}",np.count_nonzero(~np.isfinite(s[field])),0)
            report.check(prefix+".unit_direction",np.max(np.abs(np.linalg.norm(s["direction"],axis=1)-1)),1e-4)
            report.check(prefix+".unit_polarization",np.max(np.abs(np.linalg.norm(s["polarization"],axis=1)-1)),1e-4)
            report.check(prefix+".transversality",np.max(np.abs(np.sum(s["direction"]*s["polarization"],axis=1))),1e-4)
            if steps>1:
                report.check(prefix+".step_limit",result.step_limit_count,0)
            analytical_transport(report,name,s,expected)
            entry={"seed":seed,"count":args.count,"detected":len(result.hits),"step_limit":result.step_limit_count}
            if args.cuda_dir:
                with np.load(Path(args.cuda_dir)/f"{name}.{seed}.cuda.npz") as c:
                    meta=json.loads(str(c["metadata"]))
                    # Legacy Mesh duplicate removal stores indices as int64;
                    # scene compilation canonicalizes the identical indices
                    # to int32. Verify both byte representations explicitly.
                    index64_hash=hashlib.sha256(host.vertices.tobytes()+host.triangles.astype(np.int64).tobytes()).hexdigest()
                    if meta["geometry_sha256"] not in (geometry_hash,index64_hash) or meta["source_sha256"]!=source_hash:
                        raise RuntimeError(f"{prefix}: CUDA geometry/source inputs differ")
                    report.check(prefix+".cuda_nan_abort",np.count_nonzero(c["flags"]&(1<<15)),0)
                    for bit in (1,2,4,8,16,32,64,128):
                        a,b=(s["flags"]&bit)!=0,(c["flags"]&bit)!=0
                        pooled=(a.mean()+b.mean())/2
                        report.check(prefix+f".cuda_flag{bit}",abs(a.mean()-b.mean()),
                                     6*np.sqrt(2*pooled*(1-pooled)/len(a))+6/len(a),
                                     triton=float(a.mean()),cuda=float(b.mean()))
                    for field,tolerance in (("times",1e-4),("wavelengths",1e-4)):
                        a=s[field][(s["flags"]&4)!=0]; b=c[field][(c["flags"]&4)!=0]
                        if len(a) and len(b):
                            bound=np.sqrt(np.log(4/ALPHA)/(2*len(a)))+np.sqrt(np.log(4/ALPHA)/(2*len(b)))
                            report.check(prefix+f".cuda_{field}_cdf",two_cdf_distance(a,b,tolerance),bound,
                                         coordinate_tolerance=tolerance,triton_count=len(a),cuda_count=len(b))
                    if host.channel_count > 1:
                        from scipy.stats import chi2
                        counts=[]
                        for state in (s,c):
                            detected=(state["flags"]&4)!=0
                            channel_counts=np.bincount(state["channels"][detected],minlength=host.channel_count)
                            counts.append(np.r_[channel_counts,len(detected)-detected.sum()])
                        a,b=counts
                        # Pool sparse channels into a single category before
                        # applying the two-sample multinomial Pearson test.
                        populated=(a+b)>=10
                        a=np.r_[a[populated],a[~populated].sum()]
                        b=np.r_[b[populated],b[~populated].sum()]
                        keep=(a+b)>0
                        a,b=a[keep],b[keep]
                        if len(a)>1:
                            statistic=np.sum((a-b)**2/(a+b))
                            report.check(prefix+".cuda_channel_chi2",statistic,chi2.ppf(1-ALPHA,len(a)-1),
                                         categories=len(a),p_value=float(chi2.sf(statistic,len(a)-1)))
                    entry["cuda_detected"]=int(np.count_nonzero(c["flags"]&4))
                    nontransverse=np.abs(np.sum(c["direction"]*c["polarization"],axis=1))
                    entry["cuda_nonfinite_frames"]=int(np.count_nonzero(~np.isfinite(nontransverse)))
                    entry["cuda_max_nontransverse"]=float(np.nanmax(nontransverse))
            case["seeds"].append(entry)
            print(json.dumps({"case":name,**entry}),flush=True)
        # Independent brute-force CPU intersections on the same compiled scene.
        # Limit mesh*ray work for the full detector; larger GPU ensembles above
        # are compared to original CUDA instead.
        if name != "detector":
            count=512 if name=="pmt" else 2048
            sub=photons[:count]
            cpu=SpectralSimulation(sim.scene,backend="reference",tile_size=256).simulate(sub,seed=11,max_steps=steps)
            gpu=sim.simulate(sub,seed=11,max_steps=steps)
            disagree=cpu.final_state["flags"]!=gpu.final_state["flags"]
            report.check(name+".cpu_gpu_history_fraction",np.mean(disagree),.005)
            same=~disagree
            if np.any(same):
                report.check(name+".cpu_gpu_time_p99_ns",np.quantile(np.abs(cpu.final_state["times"][same]-gpu.final_state["times"][same]),.99),.003)
            case["cpu_gpu_count"]=len(sub)
            case["cpu_gpu_history_mismatches"]=int(disagree.sum())
        case["elapsed_seconds"]=time.monotonic()-start
        report.data["cases"].append(case)
        report.save()
    report.save()
    print(json.dumps({"passed":report.data["passed"],"checks":report.data["check_count"],"report":str(report.path)}))
    raise SystemExit(0 if report.data["passed"] else 1)


if __name__ == "__main__":
    main()
