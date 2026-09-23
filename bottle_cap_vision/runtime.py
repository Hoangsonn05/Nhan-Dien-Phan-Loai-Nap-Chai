from __future__ import annotations

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
    if (
        max_frames is not None
        and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0)
    ):
        raise ValueError("max_frames must be a positive integer")

    sink_list = list(sinks)
    summary: RunSummary | None = None
    processing_durations: list[float] = []
    count_events = []
    frames_processed = 0
    total_count = 0
    count_by_color: dict[str, int] = {}
    sinks_to_close: list[FrameSink] = []
    primary_error: BaseException | None = None

    try:
        source.open()
        metadata = source.metadata or SourceMetadata()
        for sink in sink_list:
            # Register before open so a partially initialized sink is cleaned up.
            sinks_to_close.append(sink)
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
            processing_durations.append(elapsed)
            if result.frame_index != packet.frame_index:
                raise RuntimeError(
                    "Core returned a frame index inconsistent with its source packet: "
                    f"packet={packet.frame_index}, result={result.frame_index}"
                )

            recent = processing_durations[-30:]
            recent_seconds = sum(recent)
            metrics = RuntimeMetrics(
                processing_fps=(len(recent) / recent_seconds) if recent_seconds > 0 else 0.0,
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

        processing_seconds = sum(processing_durations)
        summary = RunSummary(
            frames_processed=frames_processed,
            processing_seconds=processing_seconds,
            average_processing_fps=(frames_processed / processing_seconds)
            if processing_seconds > 0
            else 0.0,
            total_count=total_count,
            count_by_color=count_by_color,
            count_events=count_events,
        )
        return summary
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        for sink in reversed(sinks_to_close):
            try:
                sink.close(summary)
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            source.close()
        except BaseException as exc:
            cleanup_errors.append(exc)

        if cleanup_errors:
            if primary_error is not None:
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(
                        "Cleanup also failed: "
                        + "; ".join(f"{type(exc).__name__}: {exc}" for exc in cleanup_errors)
                    )
            else:
                first = cleanup_errors[0]
                if len(cleanup_errors) > 1 and hasattr(first, "add_note"):
                    first.add_note(
                        "Additional cleanup failures: "
                        + "; ".join(
                            f"{type(exc).__name__}: {exc}" for exc in cleanup_errors[1:]
                        )
                    )
                raise first
