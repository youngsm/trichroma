"""Backend selection (no GPU needed)."""

import importlib.util

import pytest

from chroma.backend import backend_name, default_backend, tape_mode


def test_default_backend_is_triton_even_with_pycuda(monkeypatch):
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args):
        return object() if name == "pycuda" else real_find_spec(name, *args)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    default_backend.cache_clear()
    try:
        assert default_backend() == "triton"
        assert backend_name({}) == "triton"
        assert backend_name({"CHROMA_BACKEND": ""}) == "triton"
    finally:
        default_backend.cache_clear()


def test_cuda_backend_without_pycuda_says_which_extra():
    import os
    import subprocess
    import sys

    if importlib.util.find_spec("pycuda") is not None:
        pytest.skip("PyCUDA is installed")
    env = dict(os.environ, CHROMA_BACKEND="cuda")
    result = subprocess.run([sys.executable, "-c", "from chroma.sim import Simulation"],
                            capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "trichroma[cuda]" in result.stderr


def test_explicit_backend_and_validation():
    assert backend_name({"CHROMA_BACKEND": "triton"}) == "triton"
    assert backend_name({"CHROMA_BACKEND": " CUDA "}) == "cuda"
    with pytest.raises(ValueError):
        backend_name({"CHROMA_BACKEND": "opencl"})


def test_tape_mode_parsing():
    assert not tape_mode({}).enabled
    assert tape_mode({"CHROMA_TRITON_TAPE": "replay:/tmp/t"}).directory == "/tmp/t"
    sorted_mode = tape_mode({"CHROMA_TRITON_TAPE": "record-sorted:/tmp/t"})
    assert (sorted_mode.mode, sorted_mode.directory, sorted_mode.sorted) == ("record", "/tmp/t", True)
    assert not tape_mode({"CHROMA_TRITON_TAPE": "record:/tmp/t"}).sorted
    with pytest.raises(ValueError):
        tape_mode({"CHROMA_TRITON_TAPE": "replay"})
