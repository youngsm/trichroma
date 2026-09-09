"""Run the validated optical regression suite from this checkout on a local GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "chroma-lite/test/test_optical_response.py",
    "chroma-lite/test/test_spectral_transport.py",
    "chroma-lite/test/test_spectral_accuracy.py",
    "chroma-lar/test/test_scintillation_source.py",
    "chroma-lar/test/test_pmt_coating.py",
    "chroma-lar/test/test_optical_pipeline.py",
    "chroma-lite/test/test_triton_optics.py",
    "chroma-lite/test/test_triton_scene.py",
    "chroma-lite/test/test_triton_runtime.py",
    "chroma-lite/test/test_triton_bvh.py",
    "chroma-lite/test/test_triton_physics.py",
    "chroma-lite/test/test_triton_transport.py",
    "chroma-lar/test/test_triton_scene_compiler.py",
    "chroma-lar/test/test_triton_scene_instances.py",
    "chroma-lar/test/test_triton_scene_intersect.py",
    "chroma-lar/test/test_triton_backend.py::test_small_end_to_end_run_has_valid_flat_hits",
    "chroma-lar/test/test_fast_spectral.py",
    "chroma-lar/test/test_triton_device_geometry.py",
    "chroma-lar/test/test_spectral_settings.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/trichroma-checks"))
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        parser.error("a local CUDA GPU is required; refusing to report a skipped GPU suite as passing")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(ROOT / p) for p in ("chroma-lite", "chroma-lar"))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = "1"
    command = [sys.executable, "-m", "pytest", "-q", *TESTS, f"--junitxml={output / 'tests.xml'}"]
    started = time.perf_counter()
    with (output / "tests.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    counts = None
    returncode = result.returncode
    if returncode == 0:
        suites = ET.parse(output / "tests.xml").getroot().findall("testsuite")
        counts = {key: sum(int(s.attrib[key]) for s in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        if not counts["tests"] or any(counts[key] for key in ("failures", "errors", "skipped")):
            returncode = 1
    report = {"checkout": str(ROOT), "python": sys.executable, "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "command": command, "returncode": result.returncode,
              "validation_returncode": returncode, "tests": counts,
              "elapsed_seconds": time.perf_counter() - started}
    (output / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    print((output / "tests.log").read_text())
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
