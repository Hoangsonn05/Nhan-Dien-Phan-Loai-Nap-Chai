"""Interactive application launcher for the three BottleCapVision practical modes."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .camera_demo import run_camera
from .models import RunSummary
from .practical_demo import (
    DEFAULT_CONFIG,
    DEFAULT_INPUT_DIR,
    PracticalDemoError,
    run_image,
)
from .video_demo import run_video


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], Any]


def _print_menu(output: OutputFunction) -> None:
    output("========================================")
    output("BOTTLE CAP COLOR RECOGNITION")
    output("========================================")
    output("1. Test image folder")
    output("2. Realtime camera")
    output("3. Test video")
    output("0. Exit")


def _print_run_summary(summary: RunSummary, output: OutputFunction) -> None:
    output("Run summary")
    output(f"frames_processed: {summary.frames_processed}")
    output(f"average_processing_fps: {summary.average_processing_fps:.2f}")
    output(f"total_count: {summary.total_count}")
    output(f"count_by_color: {summary.count_by_color}")


def run_launcher(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    input_fn: InputFunction | None = None,
    output_fn: OutputFunction | None = None,
    image_runner: Callable[..., Any] | None = None,
    camera_runner: Callable[..., RunSummary] | None = None,
    video_runner: Callable[..., RunSummary] | None = None,
) -> int:
    """Run the menu loop and dispatch each selection to an existing practical API."""
    read_input = input_fn or input
    write_output = output_fn or print
    selected_image_runner = image_runner or run_image
    selected_camera_runner = camera_runner or run_camera
    selected_video_runner = video_runner or run_video

    try:
        while True:
            _print_menu(write_output)
            choice = read_input("Select mode: ").strip()

            if choice == "0":
                write_output("Goodbye.")
                return 0
            if choice not in {"1", "2", "3"}:
                write_output("Invalid selection. Please choose 0-3.")
                continue

            try:
                if choice == "1":
                    entered = read_input(f"Image folder [{DEFAULT_INPUT_DIR}]: ").strip()
                    input_dir = Path(entered) if entered else DEFAULT_INPUT_DIR
                    selected_image_runner(input_dir, config_path=config_path)
                elif choice == "2":
                    write_output("Camera realtime started. Q / Esc: Exit")
                    summary = selected_camera_runner(
                        camera_index=0,
                        config_path=config_path,
                    )
                    _print_run_summary(summary, write_output)
                else:
                    video_path = read_input("Video path: ").strip()
                    if not video_path:
                        write_output("ERROR: Video path is required.")
                        continue
                    write_output("Video practical started. Q / Esc: Exit")
                    summary = selected_video_runner(
                        Path(video_path),
                        config_path=config_path,
                    )
                    _print_run_summary(summary, write_output)
            except (PracticalDemoError, RuntimeError, ValueError) as exc:
                write_output(f"ERROR: {type(exc).__name__}: {exc}")
    except EOFError:
        write_output("Input closed. Exiting.")
        return 0
    except KeyboardInterrupt:
        write_output("\nCancelled. Exiting.")
        return 130
