from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import bottle_cap_vision.practical_demo as practical_demo
from bottle_cap_vision.models import RunSummary
from bottle_cap_vision.practical_demo import PracticalDemoError
from bottle_cap_vision.practical_launcher import run_launcher


def _inputs(*values):
    iterator = iter(values)
    return lambda _prompt: next(iterator)


def _summary() -> RunSummary:
    return RunSummary(3, 0.5, 6.0, 2, {"Red": 2}, [])


def test_image_composition_reuses_existing_processing_and_display(monkeypatch, tmp_path):
    report = SimpleNamespace(processed_images=[])
    calls = []
    monkeypatch.setattr(
        practical_demo,
        "process_images",
        lambda *args, **kwargs: (calls.append(("process", args, kwargs)) or report),
    )
    monkeypatch.setattr(
        practical_demo,
        "print_summary",
        lambda value: calls.append(("summary", value)),
    )
    monkeypatch.setattr(
        practical_demo,
        "show_results",
        lambda images: calls.append(("display", images)),
    )
    result = practical_demo.run_image(
        tmp_path / "images",
        output_dir=tmp_path / "output",
        config_path=tmp_path / "config.json",
    )

    assert result is report
    assert calls == [
        (
            "process",
            (tmp_path / "images", tmp_path / "output", tmp_path / "config.json"),
            {"recursive": False, "detector_factory": None},
        ),
        ("summary", report),
        ("display", []),
    ]


def test_routes_multiple_modes_in_one_process_with_shared_config(tmp_path):
    config = tmp_path / "config.json"
    image_dir = tmp_path / "images"
    video_path = tmp_path / "input.mp4"
    calls = []
    output = []

    result = run_launcher(
        config_path=config,
        input_fn=_inputs("2", "3", str(video_path), "1", str(image_dir), "0"),
        output_fn=output.append,
        image_runner=lambda path, **kwargs: calls.append(("image", path, kwargs)),
        camera_runner=lambda **kwargs: (calls.append(("camera", kwargs)) or _summary()),
        video_runner=lambda path, **kwargs: (calls.append(("video", path, kwargs)) or _summary()),
    )

    assert result == 0
    assert calls == [
        ("camera", {"camera_index": 0, "config_path": config}),
        ("video", video_path, {"config_path": config}),
        ("image", image_dir, {"config_path": config}),
    ]
    assert output.count("Run summary") == 2


def test_invalid_selections_do_not_dispatch_a_mode():
    calls = []
    output = []

    result = run_launcher(
        input_fn=_inputs("9", "abc", "", "0"),
        output_fn=output.append,
        image_runner=lambda *args, **kwargs: calls.append("image"),
        camera_runner=lambda *args, **kwargs: calls.append("camera"),
        video_runner=lambda *args, **kwargs: calls.append("video"),
    )

    assert result == 0
    assert calls == []
    assert output.count("Invalid selection. Please choose 0-3.") == 3


def test_blank_video_path_returns_to_menu_without_dispatch():
    calls = []
    output = []

    result = run_launcher(
        input_fn=_inputs("3", "", "0"),
        output_fn=output.append,
        video_runner=lambda *args, **kwargs: calls.append("video"),
    )

    assert result == 0
    assert calls == []
    assert "ERROR: Video path is required." in output


def test_repeated_camera_and_cross_mode_dispatch_are_isolated(tmp_path):
    video_path = tmp_path / "input.mp4"
    calls = []

    result = run_launcher(
        input_fn=_inputs("2", "2", "3", str(video_path), "2", "0"),
        output_fn=lambda _line: None,
        camera_runner=lambda **_kwargs: (calls.append("camera") or _summary()),
        video_runner=lambda *_args, **_kwargs: (calls.append("video") or _summary()),
    )

    assert result == 0
    assert calls == ["camera", "camera", "video", "camera"]


def test_known_mode_failure_is_reported_and_menu_can_exit():
    output = []

    def unavailable_camera(**_kwargs):
        raise PracticalDemoError("camera unavailable")

    result = run_launcher(
        input_fn=_inputs("2", "0"),
        output_fn=output.append,
        camera_runner=unavailable_camera,
    )

    assert result == 0
    assert "ERROR: PracticalDemoError: camera unavailable" in output


def test_unexpected_mode_failure_is_not_swallowed():
    def broken_camera(**_kwargs):
        raise LookupError("unexpected integration failure")

    with pytest.raises(LookupError, match="unexpected integration failure"):
        run_launcher(
            input_fn=_inputs("2"),
            output_fn=lambda _line: None,
            camera_runner=broken_camera,
        )


def test_ctrl_c_exits_cleanly_after_runner_cleanup():
    lifecycle = []
    output = []

    def interrupted_camera(**_kwargs):
        try:
            lifecycle.append("open")
            raise KeyboardInterrupt
        finally:
            lifecycle.append("close")

    result = run_launcher(
        input_fn=_inputs("2"),
        output_fn=output.append,
        camera_runner=interrupted_camera,
    )

    assert result == 130
    assert lifecycle == ["open", "close"]
    assert "\nCancelled. Exiting." in output
