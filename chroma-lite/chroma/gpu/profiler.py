import os
import atexit
import threading
from typing import Any, Dict

import pycuda.driver as cuda
from chroma.log import logger
import numpy as np


class _KernelStats:
    __slots__ = (
        "name",
        "calls",
        "total_ms",
        "min_ms",
        "max_ms",
        "last_ms",
    )

    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.total_ms = 0.0
        self.min_ms = float("inf")
        self.max_ms = 0.0
        self.last_ms = 0.0

    def add(self, ms: float):
        self.calls += 1
        self.total_ms += ms
        self.last_ms = ms
        if ms < self.min_ms:
            self.min_ms = ms
        if ms > self.max_ms:
            self.max_ms = ms


class _Profiler:
    def __init__(self):
        self._enabled = False
        self._detailed = False
        self._lock = threading.Lock()
        self._stats: Dict[str, _KernelStats] = {}
        self._per_call: Dict[str, list] = {}
        self._patched = False

    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self, detailed: bool = False):
        with self._lock:
            self._enabled = True
            self._detailed = detailed
            if not self._patched:
                self._patch_get_function()
                self._patched = True

    def disable(self):
        with self._lock:
            self._enabled = False

    def reset(self):
        with self._lock:
            self._stats.clear()
            self._per_call.clear()

    def _patch_get_function(self):
        # Monkey-patch SourceModule.get_function to wrap kernels globally
        import pycuda.compiler as _compiler
        orig_get_function = _compiler.SourceModule.get_function

        def wrapped_get_function(mod, name, *args, **kwargs):
            f = orig_get_function(mod, name, *args, **kwargs)
            return self.wrap_function(f, name)

        # Avoid double patching
        if getattr(_compiler.SourceModule.get_function, "_chroma_profiled", False):
            return
        wrapped_get_function._chroma_profiled = True  # type: ignore[attr-defined]
        _compiler.SourceModule.get_function = wrapped_get_function  # type: ignore[assignment]

    def _get_stats(self, name: str) -> _KernelStats:
        st = self._stats.get(name)
        if st is None:
            st = _KernelStats(name)
            self._stats[name] = st
        return st

    def wrap_function(self, func, name: str):
        # If already wrapped, return as-is
        if getattr(func, "_is_chroma_kernel_wrapper", False):
            return func

        profiler = self

        class KernelWrapper:
            _is_chroma_kernel_wrapper = True

            def __init__(self, f, kname):
                self._f = f
                self._name = kname

            def __getattr__(self, item):
                return getattr(self._f, item)

            def __call__(self, *args, **kwargs):
                if not profiler.is_enabled():
                    return self._f(*args, **kwargs)

                # Determine stream if provided; default stream if not
                stream = kwargs.get("stream", None)
                start = cuda.Event()
                end = cuda.Event()
                if stream is not None:
                    start.record(stream)
                else:
                    start.record()

                # Launch kernel
                ret = self._f(*args, **kwargs)

                if stream is not None:
                    end.record(stream)
                else:
                    end.record()
                end.synchronize()
                ms = end.time_since(start)

                with profiler._lock:
                    st = profiler._get_stats(self._name)
                    st.add(ms)
                    if profiler._detailed:
                        profiler._per_call.setdefault(self._name, []).append(ms)
                        logger.debug(f"kernel {self._name} took {ms:.3f} ms")

                return ret

        return KernelWrapper(func, name)

    def stats(self) -> Dict[str, dict]:
        with self._lock:
            out = {}
            for k, st in self._stats.items():
                out[k] = {
                    "calls": st.calls,
                    "total_ms": st.total_ms,
                    "avg_ms": (st.total_ms / st.calls) if st.calls else 0.0,
                    "min_ms": (0.0 if st.min_ms == float("inf") else st.min_ms),
                    "max_ms": st.max_ms,
                    "last_ms": st.last_ms,
                }
            return out

    def report(self, sort_by: str = "total_ms", top: int = 0) -> str:
        stats = self.stats()
        items = [
            (name, s["calls"], s["total_ms"], s["avg_ms"], s["min_ms"], s["max_ms"], s["last_ms"])
            for name, s in stats.items()
        ]
        if sort_by == "calls":
            items.sort(key=lambda x: x[1], reverse=True)
        else:
            items.sort(key=lambda x: x[2], reverse=True)
        if top > 0:
            items = items[:top]

        lines = ["CUDA kernel profile (name | calls | total ms | avg ms | min | max | last):"]
        for (name, calls, total_ms, avg_ms, min_ms, max_ms, last_ms) in items:
            lines.append(f"{name} | {calls} | {total_ms:.3f} | {avg_ms:.3f} | {min_ms:.3f} | {max_ms:.3f} | {last_ms:.3f}")
        report = "\n".join(lines)
        logger.info(report)
        return report


profiler = _Profiler()


def enable(detailed: bool = False):
    profiler.enable(detailed=detailed)


def disable():
    profiler.disable()


def reset():
    profiler.reset()


def is_enabled() -> bool:
    return profiler.is_enabled()


def wrap_function(func, name: str):
    return profiler.wrap_function(func, name)


def stats():
    return profiler.stats()


def report(sort_by: str = "total_ms", top: int = 0) -> str:
    return profiler.report(sort_by=sort_by, top=top)


# -------- Device-side (inner-kernel) profiling helpers --------

_DEVICE_REGION_NAMES = {
    0: "intersect_mesh",
    1: "intersect_node",
    2: "intersect_triangle",
    3: "intersect_box",
}


def device_fetch(module=None):
    """Fetch device profiling counters (calls and cycles) into host arrays.

    Returns a dict: { name: {calls, cycles} }
    Requires kernels compiled with -DCHROMA_DEVICE_PROFILE=1.
    """
    if module is None:
        from chroma.gpu.tools import get_cu_module, cuda_options
        module = get_cu_module('propagate.cu', options=cuda_options)
    try:
        calls_ptr, calls_sz = module.get_global('chroma_prof_calls')
        cycles_ptr, cycles_sz = module.get_global('chroma_prof_cycles')
    except Exception as e:
        raise RuntimeError('Device profiling symbols not found. Compile with CHROMA_DEVICE_PROFILE=1') from e

    n = min(calls_sz // 8, cycles_sz // 8)
    calls = np.zeros(n, dtype=np.uint64)
    cycles = np.zeros(n, dtype=np.uint64)
    cuda.memcpy_dtoh(calls, calls_ptr)
    cuda.memcpy_dtoh(cycles, cycles_ptr)

    out = {}
    for i in range(n):
        name = _DEVICE_REGION_NAMES.get(i, f'region_{i}')
        out[name] = { 'calls': int(calls[i]), 'cycles': int(cycles[i]) }
    return out


def device_reset(module=None):
    """Zero device profiling counters (if compiled)."""
    if module is None:
        from chroma.gpu.tools import get_cu_module, cuda_options
        module = get_cu_module('propagate.cu', options=cuda_options)
    try:
        reset = module.get_function('chroma_prof_reset')
    except Exception as e:
        # Fallback: try host memset if reset kernel not available for any reason
        try:
            calls_ptr, calls_sz = module.get_global('chroma_prof_calls')
            cycles_ptr, cycles_sz = module.get_global('chroma_prof_cycles')
            cuda.memset_d8(calls_ptr, 0, calls_sz)
            cuda.memset_d8(cycles_ptr, 0, cycles_sz)
            return
        except Exception:
            raise RuntimeError('Device profiling reset not available; ensure CHROMA_DEVICE_PROFILE=1') from e
    reset(block=(64,1,1), grid=(1,1))


def device_report(module=None, clock_khz=None) -> str:
    """Format a report using device-side counters; converts cycles to ms.

    clock_khz: if None, uses current device clock rate.
    """
    stats = device_fetch(module)
    if clock_khz is None:
        dev = cuda.Context.get_device()
        attrs = dev.get_attributes()
        clock_khz = attrs.get(cuda.device_attribute.CLOCK_RATE, 0)  # in kHz
        if not clock_khz:
            clock_khz = 1000000  # fallback 1 GHz
    khz = float(clock_khz)

    lines = ["CUDA device profile (name | calls | total ms | avg us):"]
    for name, s in stats.items():
        calls = s['calls']
        cycles = s['cycles']
        total_ms = (cycles / (khz * 1000.0))
        avg_us = (total_ms * 1000.0 / calls) if calls else 0.0
        lines.append(f"{name} | {calls} | {total_ms:.3f} | {avg_us:.3f}")
    report = "\n".join(lines)
    logger.info(report)
    return report


# Auto-enable via environment variable
_auto = os.environ.get("CHROMA_CUDA_PROFILE", "").strip().lower()
_detailed = os.environ.get("CHROMA_CUDA_PROFILE_DETAIL", "").strip().lower()
_autoreport = os.environ.get("CHROMA_CUDA_PROFILE_AUTOREPORT", "").strip().lower()

if _auto in ("1", "true", "yes", "on"):
    enable(detailed=_detailed in ("1", "true", "yes", "on"))

    if _autoreport in ("1", "true", "yes", "on"):
        atexit.register(report)
