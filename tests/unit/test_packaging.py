"""The trichroma distribution provides chroma and chroma.triton from one tree."""

import os
import subprocess
import sys
from importlib import metadata


def test_distribution_is_trichroma():
    dist = metadata.distribution("trichroma")
    assert dist.version == "0.6.0"
    requires = " ".join(dist.requires or [])
    for name in ("numpy", "scipy", "torch", "triton", "setuptools"):
        assert name in requires
    assert 'extra == "cuda"' in requires


def test_torch_and_triton_are_bounded_to_the_validated_stack():
    from packaging.requirements import Requirement

    specs = {r.name: r.specifier for r in map(Requirement, metadata.distribution("trichroma").requires or [])}
    assert "2.5.1" in specs["torch"] and "2.6.0" not in specs["torch"]
    assert "3.1.0" in specs["triton"] and "3.2.0" not in specs["triton"]


def test_chroma_and_engine_come_from_the_same_tree():
    import chroma
    import chroma.triton

    chroma_dir = os.path.dirname(os.path.abspath(chroma.__file__))
    assert os.path.dirname(os.path.dirname(os.path.abspath(chroma.triton.__file__))) == chroma_dir
    for old in ("Chroma",):
        try:
            metadata.distribution(old)
        except metadata.PackageNotFoundError:
            continue
        raise AssertionError("the old %r distribution is still installed: uninstall it" % old)


def test_import_chroma_imports_no_torch():
    code = ("import sys, chroma\n"
            "heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'triton', 'pycuda'))\n"
            "assert not heavy, heavy\n")
    env = {k: v for k, v in os.environ.items() if k != "CHROMA_BACKEND"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_cuda_kernels_are_package_data():
    import chroma

    cuda_dir = os.path.join(os.path.dirname(chroma.__file__), "cuda")
    assert any(name.endswith(".cu") for name in os.listdir(cuda_dir))
