"""Independent regressions for errors found during the full optics audit."""
import numpy as np
import pytest

from chroma.triton.optical_response import TabulatedCDF, uniform, _philox4x32


@pytest.mark.parametrize("counter,key,expected", [
    ([0]*4, [0]*2, [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]),
    ([0xffffffff]*4, [0xffffffff]*2, [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]),
    ([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0],
     [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]),
])
def test_philox_random123_known_answer_vectors(counter, key, expected):
    # Primary reference: DEShawResearch/random123 tests/kat_vectors.
    np.testing.assert_array_equal(_philox4x32(counter, key), expected)


def test_random_streams_do_not_alias_seeds_or_large_ids():
    ids = np.arange(8192, dtype=np.int64)
    a, b = uniform(ids, 0), uniform(ids, 1)
    assert not np.array_equal(a, b[ids ^ 1])  # Failed with the original hash.
    assert not np.array_equal(a, uniform(ids, 1 << 32))
    assert abs(np.corrcoef(a, b)[0, 1]) < .06
    # These IDs collided in every stream under the original high/low XOR.
    first = [uniform([0], 9, s)[0] for s in range(20)]
    second = [uniform([(1 << 32) | 0x85ebca6b], 9, s)[0] for s in range(20)]
    assert not np.array_equal(first, second)
    assert np.all((a > 0) & (a < 1))


def test_pdf_inverse_is_exact_for_slopes_and_plateaus():
    triangle = TabulatedCDF.from_pdf([120, 128, 136], [0, 1, 0])
    u = np.linspace(0, 1, 10001, endpoint=False)
    expected = np.where(u < .5, 120 + np.sqrt(128*u), 136 - np.sqrt(128*(1-u)))
    np.testing.assert_allclose(triangle.sample(u), expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(triangle.evaluate(triangle.sample(u)), u, atol=2e-15)
    np.testing.assert_allclose(triangle.evaluate([119, 124, 128, 132, 140]), [0, .125, .5, .875, 1])
    rising = TabulatedCDF.from_pdf([0, 2], [0, 1])
    np.testing.assert_allclose(rising.sample(u), 2*np.sqrt(u), atol=1e-14)
    flat = TabulatedCDF.from_pdf([0, 1, 2, 3, 4], [1, 0, 0, 0, 1])
    assert flat.sample(.5) == 3
    np.testing.assert_allclose(flat.evaluate(flat.sample(u)), u, atol=2e-15)


def test_gpu_philox_matches_cpu_for_full_width_seeds_and_ids():
    torch = pytest.importorskip("torch")
    triton = pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("local GPU required")
    import triton.language as tl
    from chroma.triton.spectral_kernels import random_uniform
    globals().update(tl=tl, random_uniform=random_uniform)

    @triton.jit
    def kernel(ids, output, count, seed, stream, BLOCK: tl.constexpr):
        i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
        pid = tl.load(ids+i, i < count, other=0)
        tl.store(output+i, random_uniform(pid, seed, stream), i < count)

    ids = np.r_[np.arange(257), 2**32+np.arange(259), 2**62+np.arange(261)]
    device_ids = torch.tensor(ids, device="cuda")
    output = torch.empty(len(ids), device="cuda")
    for seed in (0, 1, 2**32, 2**63+91, 2**64-1):
        for stream in (0, 31, 0x10000006, 0x20000003):
            kernel[(triton.cdiv(len(ids), 128),)](device_ids, output, len(ids), seed, stream, BLOCK=128)
            np.testing.assert_array_equal(output.cpu().numpy(), uniform(ids, seed, stream))


def test_wls_pdf_delay_cpu_gpu_shape():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("local GPU required")
    from .test_spectral_transport import coated_detector, source, run
    d = coated_detector()
    coat = d.solids[-1].surface[0]
    coat.reemission_time_cdf = TabulatedCDF.from_pdf([0, 20], [0, 1])
    p = source(8192, z=-10)
    for backend in ("reference", "triton"):
        result = run(d, p, backend=backend, seed=123)
        state = result.final_state
        distance = np.linalg.norm(state["pos"]-[.23, -.17, 0], axis=1)
        dt = state["times"]-7.1-distance/200
        assert abs(dt.mean()-40/3) < .2
        assert abs(np.mean(dt <= 10)-.25) < .015
        expected = 20*np.sqrt(uniform(np.arange(len(dt)), 123, 9))
        np.testing.assert_allclose(dt, expected, atol=2e-5)


def test_unsupported_or_ambiguous_wls_tables_fail_before_transport():
    from .test_spectral_transport import coated_detector, GRID
    from chroma.triton.spectral import SpectralScene
    d=coated_detector()
    coat=d.solids[-1].surface[0]
    coat.set("detect",.1)
    with pytest.raises(ValueError,match="cannot detect"):
        SpectralScene.compile(d,wavelengths=GRID)
    coat.set("detect",0)
    coat.set("reemission_cdf",np.clip((GRID-410)/40,0,1)*(1-5e-7),GRID)
    with pytest.raises(ValueError,match="endpoints"):
        SpectralScene.compile(d,wavelengths=GRID)


def test_nonfinite_transport_result_cannot_silently_escape():
    from chroma.triton.spectral import _finish
    with pytest.raises(RuntimeError,match="nonfinite"):
        _finish({"pos":np.array([[np.inf,0,0]])},1,"test")


def test_timing_instrumentation_preserves_photons():
    torch=pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("local GPU required")
    from .test_spectral_transport import coated_detector, source, GRID
    from chroma.triton.spectral import SpectralSimulation
    sim=SpectralSimulation(coated_detector(),wavelengths=GRID,tile_size=64)
    p=source(137,z=-10)
    expected=sim.simulate(p,seed=123)
    stages=[]
    actual=sim.simulate(p,seed=123,timings=stages)
    for key in expected.final_state:
        np.testing.assert_array_equal(actual.final_state[key],expected.final_state[key])
    assert len(stages)==3
    assert sum(stage["photons"] for stage in stages)==137
    for stage in stages:
        assert all(stage[key]>=0 for key in ("upload_allocation_seconds","transport_seconds","download_result_seconds"))
