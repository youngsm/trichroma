import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from chroma_lar.optical_calibration import OpticalCalibration
from chroma_lar.optical_simulation import OpticalSimulation

ROOT = Path(__file__).parents[1]


def calibration():
    return OpticalCalibration.load(ROOT/"examples/synthetic_optical_calibration.json")


def detector(c):
    spec = importlib.util.spec_from_file_location("run_optical_simulation", ROOT/"benchmarks/run_optical_simulation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.demo_detector(c)


@pytest.mark.parametrize("backend", ["reference", "triton"])
def test_complete_pipeline_and_zero_light_event(tmp_path, backend):
    if backend == "triton":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA GPU required")
    c = calibration()
    simulation = OpticalSimulation(detector(c), c, backend=backend)
    result = simulation.simulate_depositions([[0,0,0], [0,0,0]], [.015, 0.], event_indices=[3,7], seed=8)
    assert result.metadata["photons"] > 0
    assert result.metadata["detected_photons"] > 10
    assert result.metadata["calibration_status"] == "synthetic"
    assert result.transport.step_limit_count == 0
    assert result.waveforms.samples.shape == (2, 1, 16000)
    np.testing.assert_array_equal(result.waveforms.event_indices, [3,7])
    assert np.all(result.transport.hits.wavelengths > 200)
    path = tmp_path/"result.npz"
    result.save(path)
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata_json"]))
        assert metadata["calibration_fingerprint"] == c.fingerprint
        np.testing.assert_array_equal(saved["waveforms_adc"], result.waveforms.samples)


def test_calibration_file_hashes_coverage_and_missing_optics(tmp_path):
    config = json.loads((ROOT/"examples/synthetic_optical_calibration.json").read_text())
    config["scintillation"]["spectrum"] = {"file": "spectrum.csv", "kind": "pdf"}
    spectrum = tmp_path/"spectrum.csv"
    spectrum.write_text("120,0\n128,1\n136,0\n")
    path = tmp_path/"calibration.json"
    path.write_text(json.dumps(config))
    c1 = OpticalCalibration.load(path)
    spectrum.write_text("120,0\n128,2\n136,0\n")
    c2 = OpticalCalibration.load(path)
    assert c1.fingerprint != c2.fingerprint
    assert c1.files["spectrum.csv"] != c2.files["spectrum.csv"]
    d = detector(c2)
    del c2.materials["demo_medium"]
    with pytest.raises(ValueError, match="missing geometry optics"):
        c2.apply_to_geometry(d)
    config["materials"]["demo_medium"]["absorption_length"] = [[125, 100], [500, 100]]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="cover"):
        OpticalCalibration.load(path)


def test_measurement_comparison_detects_timing_and_yield_changes():
    spec = importlib.util.spec_from_file_location("compare_optical_measurements", ROOT/"benchmarks/compare_optical_measurements.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.compare_samples(np.arange(20), np.zeros(20, int), 100,
                                     np.arange(10)+50, np.zeros(10, int), 100)
    assert report["simulation_pe_per_emitted_photon"] == .2
    assert report["observed_pe_per_emitted_photon"] == .1
    assert report["channels"]["0"]["time_ks_distance"] == 1


def test_calibration_replacements_survive_flattening():
    from chroma.geometry import Material, Solid, Surface
    from chroma.detector import Detector
    from chroma.make import box
    from chroma.triton.spectral import SpectralScene
    c=calibration()
    old=Material("demo_medium")
    old.set("refractive_index", 7.)
    old_surface=Surface("demo_sensor")
    old_surface.set("absorb",1.)
    d=Detector(old)
    d.add_pmt(Solid(box(200,200,200),old,old,surface=old_surface),displacement=np.zeros(3))
    c.apply_to_geometry(d)
    scene=SpectralScene.compile(d,wavelengths=c.wavelengths)
    assert d.unique_materials == [c.materials["demo_medium"]]
    assert d.unique_surfaces == [c.surfaces["demo_sensor"]]
    np.testing.assert_allclose(scene.host.optics.materials.refractive_index,1.4)
    np.testing.assert_allclose(scene.host.optics.surfaces.detect,.25)


def test_explicit_events_cannot_drop_undetected_input():
    c=calibration()
    simulation=OpticalSimulation(detector(c),c,backend="reference")
    photons=c.source.photons([[0,0,0]],event_indices=[7])
    with pytest.raises(ValueError,match="every input photon event"):
        simulation.simulate_photons(photons,event_indices=[8])
