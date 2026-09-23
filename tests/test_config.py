import json

import pytest

from bottle_cap_vision.config import AppConfig, DEFAULT_COLORS, load_config, save_config


def test_default_config_contains_all_expected_colors():
    cfg = AppConfig.default()
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
