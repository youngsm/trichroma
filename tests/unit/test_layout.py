"""The engine lives at chroma.triton; nothing imports the old trichroma package."""

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_no_trichroma_imports_left():
    pattern = re.compile(r"^\s*(from|import)\s+trichroma\b", re.M)
    offenders = []
    for base in ("chroma", "tests", "benchmarks"):
        for path in (ROOT / base).rglob("*.py"):
            if pattern.search(path.read_text()):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_engine_modules_import():
    import chroma.triton.options
    import chroma.triton.sources
    import chroma.triton.utils

    assert chroma.triton.utils.two_proportion_z(10, 100, 10, 100) == 0.0


def test_verify_module_runs_as_main():
    env = dict(os.environ, CHROMA_BACKEND="triton")
    result = subprocess.run([sys.executable, "-m", "chroma.triton.tape.verify", "--help"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "python -m chroma.triton.tape.verify" in result.stdout


def test_old_cuda_suite_is_opt_in_even_with_pycuda(tmp_path):
    (tmp_path / "pycuda").mkdir()
    (tmp_path / "pycuda" / "__init__.py").write_text("")
    env = {k: v for k, v in os.environ.items() if k != "CHROMA_CUDA_SUITE"}
    env["PYTHONPATH"] = str(tmp_path) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/cuda_backend"],
                            capture_output=True, text=True, env=env, cwd=ROOT)
    assert result.returncode == 5, result.stdout[-2000:]  # 5: no tests collected
