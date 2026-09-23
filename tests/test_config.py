import json

import pytest

from bottle_cap_vision.config import AppConfig, DEFAULT_COLORS, config_to_dict, load_config, save_config


def test_default_config_contains_all_expected_colors():
    cfg = AppConfig.default()
    assert cfg.detector.backend == "hsv"
    assert [color.name for color in cfg.colors] == DEFAULT_COLORS
    assert all(color.hsv_ranges for color in cfg.colors)


def test_project_config_loads_all_expected_colors():
    cfg = load_config("configs/config.json")
    assert [color.name for color in cfg.colors] == DEFAULT_COLORS


def test_missing_colors_key_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"detection": {"min_area": 123}}), encoding="utf-8")
    with pytest.raises(ValueError, match="colors"):
        load_config(path)


def test_explicit_empty_colors_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"colors": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        load_config(path)


def test_invalid_hsv_ranges_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    payload = {
        "colors": [
            {
                "name": "Blue",
                "hsv_ranges": [{"lower": [180, 0, 0], "upper": [200, 255, 255]}],
            }
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="HSV|OpenCV|0..179"):
        load_config(path)


def test_invalid_tracking_range_and_enum_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    payload = {
        "tracking": {"max_distance": -5},
        "counting": {"orientation": "diagonal", "direction": "left"},
        "colors": [{"name": "Blue", "hsv_ranges": [{"lower": [0, 0, 0], "upper": [10, 255, 255]}]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_unknown_fields_are_rejected(tmp_path):
    path = tmp_path / "config.json"
    payload = {
        "tracking": {"max_distance": 80, "unexpected": True},
        "colors": [{"name": "Blue", "hsv_ranges": [{"lower": [0, 0, 0], "upper": [10, 255, 255]}]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected|unknown"):
        load_config(path)


def test_config_round_trip(tmp_path):
    path = tmp_path / "config.json"
    source = AppConfig.default()
    source.roi.enabled = True
    source.roi.x1 = 10
    source.roi.y1 = 20
    source.roi.x2 = 120
    source.roi.y2 = 140
    save_config(source, path)
    loaded = load_config(path)
    assert loaded.roi.enabled is True
    assert loaded.roi.x1 == 10
    assert loaded.roi.x2 == 120
    assert loaded.roi.y2 == 140
    assert [c.name for c in loaded.colors] == DEFAULT_COLORS


@pytest.mark.parametrize(
    "color_stabilization",
    [
        {"enabled": 1},
        {"window_size": 0},
        {"min_observations": 0},
        {"window_size": 3, "min_observations": 4},
        {"unexpected": True},
    ],
)
def test_invalid_color_stabilization_config_is_rejected(tmp_path, color_stabilization):
    path = tmp_path / "config.json"
    payload = {
        "color_stabilization": color_stabilization,
        "colors": [{"name": "Blue", "hsv_ranges": [{"lower": [0, 0, 0], "upper": [10, 255, 255]}]}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="color_stabilization|Unknown"):
        load_config(path)


def test_color_stabilization_defaults_and_round_trip(tmp_path):
    cfg = AppConfig.default()
    assert cfg.color_stabilization.enabled is True
    assert cfg.color_stabilization.window_size == 5
    assert cfg.color_stabilization.min_observations == 3

    path = tmp_path / "config.json"
    save_config(cfg, path)
    loaded = load_config(path)

    assert loaded.color_stabilization == cfg.color_stabilization


def test_learned_detector_config_requires_checkpoint_and_metadata(tmp_path):
    path = tmp_path / "config.json"
    payload = config_to_dict(AppConfig.default())
    payload["detector"] = {"backend": "learned"}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint"):
        load_config(path)


@pytest.mark.parametrize(
    "detector",
    [
        {"backend": "unknown"},
        {"backend": "learned", "checkpoint": "best.pt", "metadata": "meta.json", "confidence_threshold": -0.1},
        {"backend": "learned", "checkpoint": "best.pt", "metadata": "meta.json", "confidence_threshold": 1.1},
        {"backend": "learned", "checkpoint": "best.pt", "metadata": "meta.json", "device": "gpu"},
        {"backend": "hsv", "checkpoint": "best.pt"},
        {"backend": "hsv", "unexpected": True},
    ],
)
def test_invalid_detector_config_is_rejected(tmp_path, detector):
    path = tmp_path / "config.json"
    payload = config_to_dict(AppConfig.default())
    payload["detector"] = detector
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="detector|Detector"):
        load_config(path)


def test_learned_detector_config_round_trip(tmp_path):
    cfg = AppConfig.default()
    cfg.detector.backend = "learned"
    cfg.detector.checkpoint = "experiment/checkpoints/best.pt"
    cfg.detector.metadata = "experiment/checkpoints/checkpoint_metadata.json"
    cfg.detector.device = "cpu"
    cfg.detector.confidence_threshold = 0.4
    path = tmp_path / "config.json"

    save_config(cfg, path)

    assert load_config(path).detector == cfg.detector


def test_project_learned_smoke_config_is_loadable():
    cfg = load_config("configs/config.learned-p9-smoke.json")

    assert cfg.detector.backend == "learned"
    assert cfg.detector.checkpoint == "outputs/experiments/p9_smoke_001/checkpoints/best.pt"
    assert cfg.detector.metadata.endswith("checkpoint_metadata.json")
    assert cfg.detector.device == "cpu"
    assert cfg.detector.confidence_threshold == 0.25
    assert [color.name for color in cfg.colors] == DEFAULT_COLORS
