from __future__ import annotations

from contextlib import ExitStack
from time import perf_counter
from typing import Iterable, Protocol

from .models import FramePacket, FrameResult, RunSummary, RuntimeMetrics, SourceMetadata


class FrameSource(Protocol):
    metadata: SourceMetadata

    def open(self) -> None: ...
    def __iter__(self) -> Iterable[FramePacket]: ...
    def close(self) -> None: ...


class VisionCore(Protocol):
    def process_frame(self, frame, frame_index: int = 0) -> FrameResult: ...


class FrameSink(Protocol):
    def open(self, metadata: SourceMetadata) -> None: ...
    def write(self, packet: FramePacket, result: FrameResult, metrics: RuntimeMetrics) -> bool: ...
    def close(self, summary: RunSummary | None = None) -> None: ...


def run(
    source: FrameSource,
    core: VisionCore,
    sinks: Iterable[FrameSink] = (),
    *,
    max_frames: int | None = None,
) -> RunSummary:
    """Run Source -> Core -> Sink orchestration with deterministic cleanup."""
    sink_list = list(sinks)
    summary: RunSummary | None = None
    fps_samples: list[float] = []
    count_events = []
    frames_processed = 0
    total_count = 0
    count_by_color: dict[str, int] = {}

    with ExitStack() as stack:
        stack.callback(source.close)
        source.open()
        metadata = source.metadata or SourceMetadata()
        for sink in sink_list:
            stack.callback(lambda current=sink: current.close(summary))
            sink.open(metadata)

        packets = iter(source)
        while max_frames is None or frames_processed < max_frames:
            try:
                packet = next(packets)
            except StopIteration:
                break

            started = perf_counter()
            result = core.process_frame(packet.frame, packet.frame_index)
            elapsed = perf_counter() - started
            if elapsed > 0:
                fps_samples.append(1.0 / elapsed)

            recent = fps_samples[-30:]
            metrics = RuntimeMetrics(
                processing_fps=sum(recent) / len(recent) if recent else 0.0,
            )
            frames_processed += 1
            total_count = result.total_count
            count_by_color = dict(result.count_by_color)
            count_events.extend(result.events)

            stop_requested = False
            for sink in sink_list:
                if sink.write(packet, result, metrics) is False:
                    stop_requested = True
            if stop_requested:
                break

        summary = RunSummary(
            frames_processed=frames_processed,
            average_processing_fps=(sum(fps_samples) / len(fps_samples)) if fps_samples else 0.0,
            total_count=total_count,
            count_by_color=count_by_color,
            count_events=count_events,
        )
        return summary
