"""Backend selection (no GPU needed)."""

import importlib.util

import pytest

from chroma.backend import backend_name, default_backend, tape_mode


def test_default_backend_follows_pycuda():
    expected = "cuda" if importlib.util.find_spec("pycuda") is not None else "triton"
    assert default_backend() == expected
    assert backend_name({}) == expected
    assert backend_name({"CHROMA_BACKEND": ""}) == expected


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
