"""Shared, deterministic inputs for independent CUDA/Triton comparisons."""
from pathlib import Path
import numpy as np
from chroma.detector import Detector
from chroma.event import Photons
from chroma.geometry import Material, Mesh, Solid, Surface
from chroma.make import box

GRID = np.arange(120, 501, dtype=np.float32)
CASES = ["ballistic", "absorption", "competing_bulk", "rayleigh", "default_surface", "diffuse", "specular",
         "fresnel_s_0", "fresnel_p_30", "fresnel_s_60", "fresnel_p_brewster", "fresnel_s_grazing",
         "tir", "critical", "wls", "wls_loss", "pmt", "detector"]


def material(name, n=1.):
    m = Material(name)
    m.set("refractive_index", n)
    m.set("absorption_length", 1.e30)
    m.set("scattering_length", 1.e30)
    return m


def surface(name, **properties):
    s = Surface(name, model=properties.pop("model", 0))
    for key, value in properties.items():
        s.set(key, value)
    return s


def plane():
    return Mesh([[-1e6,-1e6,0], [1e6,-1e6,0], [1e6,1e6,0], [-1e6,1e6,0]], [[0,1,2], [0,2,3]])


def make_case(name, count):
    medium = material("medium")
    detector = Detector(medium)
    detector.add_pmt(Solid(box(200,200,200), medium, medium, surface=surface("sensor", detect=1)),
                     displacement=np.zeros(3))
    pos = np.tile([.23,-.17,0.], (count,1))
    direction = np.tile([0.,0.,1.], (count,1))
    polarization = np.tile([1.,0.,0.], (count,1))
    wavelengths = np.full(count, 128.)
    steps, expectations = 256, {}
    if name == "ballistic":
        medium.set("refractive_index", 1.4)
    elif name == "absorption":
        medium.set("absorption_length", np.where(GRID < 300, 100, 1000), GRID)
        wavelengths[count//2:] = 450
    elif name in ("competing_bulk", "rayleigh"):
        steps = 1
        medium.set("scattering_length", 100 if name == "competing_bulk" else 10)
        if name == "competing_bulk":
            medium.set("absorption_length", 200)
    elif name in ("default_surface", "diffuse", "specular"):
        steps = 1
        pos[:,2] = -10
        props = {"absorb":.2, "detect":.3, "reflect_diffuse":.25, "reflect_specular":.15}
        if name != "default_surface":
            props = {"reflect_diffuse" if name == "diffuse" else "reflect_specular":1.}
        detector.add_pmt(Solid(plane(), medium, medium, surface=surface(name, **props)), displacement=np.zeros(3))
        if name == "specular":
            direction[:] = [.6,0,.8]
            polarization[:] = [.8,0,-.6]
    elif name.startswith("fresnel") or name in ("tir", "critical"):
        steps = 1
        angle = {"fresnel_s_0":0, "fresnel_p_30":30, "fresnel_s_60":60,
                 "fresnel_p_brewster":np.degrees(np.arctan(1.5)), "fresnel_s_grazing":89,
                 "tir":50, "critical":np.degrees(np.arcsin(1/1.5))}[name]
        n1,n2 = (1.5,1.) if name in ("tir", "critical") else (1.,1.5)
        a = np.radians(angle)
        detector = Detector(medium)
        detector.add_solid(Solid(plane(), material("from", n1), material("to", n2)))
        pos[:,2] = -10
        direction[:] = [np.sin(a),0,np.cos(a)]
        is_p = "_p_" in name
        polarization[:] = [np.cos(a),0,-np.sin(a)] if is_p else [0,1,0]
        # Calculate from actual float32 direction used by both backends.
        ci = float(np.float32(np.cos(a)))
        st2 = (n1/n2)**2*(1-ci*ci)
        ct = np.sqrt(max(0,1-st2))
        r = 1. if st2 >= 1 else (((n2*ci-n1*ct)/(n2*ci+n1*ct))**2 if is_p
                                 else ((n1*ci-n2*ct)/(n1*ci+n2*ct))**2)
        expectations = {"reflectance":r, "n1":n1, "n2":n2, "angle_deg":angle}
    elif name in ("wls", "wls_loss"):
        pos[:,2] = -10
        coat = surface("coat", model=2, absorb=1, reemit=1 if name == "wls" else .65)
        coat.set("reemission_cdf", np.clip((GRID-410)/40,0,1), GRID)
        detector.add_solid(Solid(plane(), medium, medium, surface=coat))
    elif name == "pmt":
        from chroma_lar.geometry.pmt import build_r5912_pmt
        medium.set("refractive_index", 1.23)
        glass, vacuum = material("glass", 1.52), material("vacuum")
        coat = surface("coat", model=2, reemit=.8)
        coat.set("absorb", np.where(GRID < 200, 1., 0.), GRID)
        coat.set("reemission_cdf", np.clip((GRID-410)/40,0,1), GRID)
        pmt = build_r5912_pmt(outer_material=medium, glass=glass, vacuum=vacuum,
            photocathode_surface=surface("photocathode", detect=.3, absorb=.7),
            back_surface=surface("black", absorb=1), coating_surface=coat, nzsteps=12, nsteps=24)
        detector = Detector(medium)
        detector.add_pmt(pmt, displacement=np.zeros(3))
        detector.add_solid(Solid(box(1000,1000,1000), medium, medium, surface=surface("black", absorb=1)))
        rng = np.random.default_rng(941)
        radius = 35*np.sqrt(rng.random(count))
        angle = rng.uniform(0,2*np.pi,count)
        pos = np.column_stack((radius*np.cos(angle), np.full(count,50.), radius*np.sin(angle)))
        direction[:] = [0,-1,0]
    elif name == "detector":
        from chroma_lar.geometry.config_loader import build_detector_from_config
        detector = build_detector_from_config("detector_config_reflect_reflect3wires", analytic_wires=False, flatten=False)
        rng = np.random.default_rng(1981)
        pos = rng.uniform(-50,50,(count,3))
        direction = rng.normal(size=(count,3))
        direction /= np.linalg.norm(direction,axis=1)[:,None]
        polarization = np.cross(direction, rng.normal(size=(count,3)))
        polarization /= np.linalg.norm(polarization,axis=1)[:,None]
        wavelengths[:] = 450
        steps = 2048
    else:
        raise ValueError(name)
    photons = Photons(pos, direction, polarization, wavelengths, t=np.full(count,7.))
    return detector, photons, steps, expectations
