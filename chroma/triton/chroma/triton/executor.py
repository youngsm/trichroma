"""Executable asynchronous pipeline orchestration for Triton backends.

The executor is deliberately ignorant of photon-buffer layout.  Callers own
their ring buffers and provide one enqueue callback for each stage kind used by
an :class:`~chroma.triton.runtime.AsyncPipelinePlan`.  Every callback receives
the stage's ``buffer_slot`` and the driver's stream, which is enough to enqueue
an asynchronous copy or kernel launch against a triple-buffered workspace.

Importing this module is CPU-only.  :class:`TorchCudaPipelineDriver` imports
PyTorch and creates CUDA streams/events only when instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from .runtime import (
    AsyncPipelinePlan,
    PipelineStage,
    PipelineStageKind,
    RuntimeContractError,
)


StageCallback = Callable[[PipelineStage, Any], Any]


@runtime_checkable
class PipelineExecutionDriver(Protocol):
    """Stream/event operations required by :class:`AsyncPipelineExecutor`.

    ``invoke`` must enqueue the callback's work on ``stream`` without a
    device-wide synchronization.  Event waits and records must likewise be
    stream-local so independent H2D, compute, and D2H work can overlap.
    """

    def create_stream(self, name: str) -> Any:
        ...

    def create_event(self, stage: PipelineStage) -> Any:
        ...

    def wait_event(self, stream: Any, event: Any) -> None:
        ...

    def invoke(
        self, stream: Any, callback: StageCallback, stage: PipelineStage
    ) -> Any:
        ...

    def record_event(self, event: Any, stream: Any) -> None:
        ...

    def synchronize_event(self, event: Any) -> None:
        ...


@dataclass(frozen=True)
class PipelineExecution:
    """Completed callback values keyed by pipeline stage ID."""

    plan: AsyncPipelinePlan
    stage_values: Mapping[str, Any]
    synchronized_stage_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        values = dict(self.stage_values)
        expected = tuple(stage.stage_id for stage in self.plan.stages)
        if tuple(values) != expected:
            raise RuntimeContractError(
                "pipeline execution values must follow every planned stage"
            )
        synchronized = tuple(self.synchronized_stage_ids)
        expected_terminal = tuple(
            stage.stage_id for stage in self.plan.terminal_stages
        )
        if synchronized != expected_terminal:
            raise RuntimeContractError(
                "pipeline execution must synchronize every terminal stage"
            )
        object.__setattr__(self, "stage_values", MappingProxyType(values))
        object.__setattr__(self, "synchronized_stage_ids", synchronized)

    def value_for_tile(
        self, tile_index: int, kind: PipelineStageKind
    ) -> Any:
        """Return a callback value for one tile/stage kind."""

        stage_id = f"{PipelineStageKind(kind).value}:{int(tile_index)}"
        if stage_id not in self.stage_values:
            raise KeyError(stage_id)
        return self.stage_values[stage_id]


class AsyncPipelineExecutor:
    """Submit a pipeline DAG and wait only after every tile is enqueued.

    Submission walks the topologically ordered plan, inserts a stream-local
    wait for every dependency event, invokes the relevant user callback, and
    records completion on that stream.  Crucially, no event is synchronized
    until *all* stages for *all* tiles have been submitted.  This permits the
    CUDA work queues to overlap next-tile H2D, current-tile compute, and
    previous-tile D2H while preserving safe ring-buffer reuse.
    """

    def __init__(self, driver: PipelineExecutionDriver):
        if not isinstance(driver, PipelineExecutionDriver):
            raise RuntimeContractError(
                "driver does not implement the pipeline execution protocol"
            )
        self.driver = driver

    def execute(
        self,
        plan: AsyncPipelinePlan,
        *,
        compute: StageCallback,
        h2d: Optional[StageCallback] = None,
        d2h: Optional[StageCallback] = None,
    ) -> PipelineExecution:
        if not isinstance(plan, AsyncPipelinePlan):
            raise RuntimeContractError("plan must be an AsyncPipelinePlan")
        callbacks = {
            PipelineStageKind.HOST_TO_DEVICE: h2d,
            PipelineStageKind.COMPUTE: compute,
            PipelineStageKind.DEVICE_TO_HOST: d2h,
        }
        missing = sorted(
            {
                stage.kind.value
                for stage in plan.stages
                if not callable(callbacks[stage.kind])
            }
        )
        if missing:
            raise RuntimeContractError(
                "missing callbacks for pipeline stages: " + ", ".join(missing)
            )

        streams = {
            name: self.driver.create_stream(name)
            for name in dict.fromkeys(stage.stream_name for stage in plan.stages)
        }
        events: dict[str, Any] = {}
        stage_values: dict[str, Any] = {}

        # This loop contains no host/device synchronization.  Dependencies are
        # enqueued as stream waits, leaving CUDA free to overlap independent
        # work while Python continues submitting later tiles.
        for stage in plan.stages:
            stream = streams[stage.stream_name]
            event = self.driver.create_event(stage)
            for dependency in stage.depends_on:
                try:
                    dependency_event = events[dependency]
                except KeyError as exc:  # defensive for third-party plans
                    raise RuntimeContractError(
                        f"stage {stage.stage_id!r} has unavailable dependency "
                        f"{dependency!r}"
                    ) from exc
                self.driver.wait_event(stream, dependency_event)
            callback = callbacks[stage.kind]
            # Missing callbacks were rejected before creating any stream.
            assert callback is not None
            stage_values[stage.stage_id] = self.driver.invoke(
                stream, callback, stage
            )
            self.driver.record_event(event, stream)
            events[stage.stage_id] = event

        terminal_ids: list[str] = []
        for stage in plan.terminal_stages:
            self.driver.synchronize_event(events[stage.stage_id])
            terminal_ids.append(stage.stage_id)

        return PipelineExecution(plan, stage_values, tuple(terminal_ids))


class TorchCudaPipelineDriver:
    """Lazy PyTorch implementation using nonblocking CUDA streams/events.

    Callbacks run inside ``torch.cuda.stream(stream)`` and should enqueue only
    asynchronous operations (for copies, use pinned host memory together with
    ``non_blocking=True``).  Calling ``synchronize`` inside a callback defeats
    overlap and is intentionally outside the driver contract.
    """

    def __init__(self, device: Any = None, *, stream_priority: int = 0):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for the CUDA pipeline driver") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA device visible to PyTorch is required")
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("TorchCudaPipelineDriver requires a CUDA device")
        self._torch = torch
        self.device = device
        self.stream_priority = int(stream_priority)

    def create_stream(self, name: str) -> Any:
        del name
        with self._torch.cuda.device(self.device):
            return self._torch.cuda.Stream(
                device=self.device, priority=self.stream_priority
            )

    def create_event(self, stage: PipelineStage) -> Any:
        del stage
        with self._torch.cuda.device(self.device):
            return self._torch.cuda.Event(
                enable_timing=False, blocking=False, interprocess=False
            )

    def wait_event(self, stream: Any, event: Any) -> None:
        stream.wait_event(event)

    def invoke(
        self, stream: Any, callback: StageCallback, stage: PipelineStage
    ) -> Any:
        with self._torch.cuda.device(self.device), self._torch.cuda.stream(stream):
            return callback(stage, stream)

    def record_event(self, event: Any, stream: Any) -> None:
        event.record(stream)

    def synchronize_event(self, event: Any) -> None:
        event.synchronize()


__all__ = [
    "AsyncPipelineExecutor",
    "PipelineExecution",
    "PipelineExecutionDriver",
    "StageCallback",
    "TorchCudaPipelineDriver",
]
