from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import List, Tuple


DEFAULT_COLORS = [
    "Blue", "Brown", "Green", "Lilac", "Orange",
    "Pink", "Purple", "Red", "White", "Yellow",
]


@dataclass(frozen=True)
class HSVRange:
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]


@dataclass
class ColorConfig:
    name: str
    hsv_ranges: List[HSVRange]


@dataclass
class ROIConfig:
    enabled: bool = False
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0


@dataclass
class DetectionConfig:
    min_area: int = 250
    max_area: int = 15000
    min_circularity: float = 0.35
    min_solidity: float = 0.55
    morphology_kernel: int = 5
    open_iterations: int = 1
    close_iterations: int = 2
    min_color_ratio: float = 0.35


@dataclass
class DetectorConfig:
    """Runtime backend selection; training options do not belong here."""

    backend: str = "hsv"
    checkpoint: str | None = None
    metadata: str | None = None
    device: str = "cpu"
    confidence_threshold: float = 0.25


@dataclass
class TrackingConfig:
    enabled: bool = True
    max_distance: float = 80.0
    max_missing_frames: int = 12
    min_confirmed_frames: int = 2
    max_history: int = 20


@dataclass
class ColorStabilizationConfig:
    enabled: bool = True
    window_size: int = 5
    min_observations: int = 3


@dataclass
class CountingConfig:
    enabled: bool = True
    orientation: str = "horizontal"
    line_position: float = 0.60
    direction: str = "positive"
    deadband: float = 5.0


@dataclass
class AppConfig:
    colors: List[ColorConfig] = field(default_factory=list)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    color_stabilization: ColorStabilizationConfig = field(default_factory=ColorStabilizationConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)

    @classmethod
    def default(cls) -> "AppConfig":
        # Starter HSV ranges only. Calibrate them on the final image/video set.
        ranges = {
            "Blue": [(90, 70, 50, 135, 255, 255)],
            "Brown": [(5, 60, 35, 25, 255, 210)],
            "Green": [(35, 60, 40, 85, 255, 255)],
            "Lilac": [(130, 35, 50, 165, 255, 255)],
            "Orange": [(5, 80, 60, 24, 255, 255)],
            "Pink": [(155, 35, 60, 179, 255, 255), (0, 35, 60, 8, 255, 255)],
            "Purple": [(120, 45, 45, 150, 255, 255)],
            "Red": [(0, 80, 50, 8, 255, 255), (170, 80, 50, 179, 255, 255)],
            "White": [(0, 0, 150, 179, 75, 255)],
            "Yellow": [(20, 70, 60, 39, 255, 255)],
        }
        colors = [
            ColorConfig(
                name=name,
                hsv_ranges=[HSVRange((a, b, c), (d, e, f)) for a, b, c, d, e, f in values],
            )
            for name, values in ranges.items()
        ]
        return cls(colors=colors)


def _ensure_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool, got {type(value).__name__}")
    return value


def _ensure_int(value: object, field_name: str, *, minimum: int | None = None, minimum_exclusive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an int, got {type(value).__name__}")
    if minimum is not None:
        if minimum_exclusive and value <= minimum:
            raise ValueError(f"{field_name} must be > {minimum}, got {value}")
        if not minimum_exclusive and value < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    return value


def _ensure_number(value: object, field_name: str, *, minimum: float | None = None, maximum: float | None = None, minimum_exclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {type(value).__name__}")
    number = float(value)
    if minimum is not None:
        if minimum_exclusive and number <= minimum:
            raise ValueError(f"{field_name} must be > {minimum}, got {number}")
        if not minimum_exclusive and number < minimum:
            raise ValueError(f"{field_name} must be >= {minimum}, got {number}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}, got {number}")
    return number


def _validate_hsv_triplet(value: object, field_name: str) -> Tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 HSV values")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{field_name} must contain numeric HSV values")
    h, s, v = (int(item) for item in value)
    if not (0 <= h <= 179 and 0 <= s <= 255 and 0 <= v <= 255):
        raise ValueError(f"{field_name} is outside OpenCV HSV ranges: {value} (expected H in [0,179], S/V in [0,255])")
    if h < 0 or h > 179 or s < 0 or s > 255 or v < 0 or v > 255:
        raise ValueError(f"{field_name} is outside OpenCV HSV ranges: {value}")
    return h, s, v


def _validate_hsv_range(raw_range: object, field_name: str) -> HSVRange:
    if not isinstance(raw_range, dict):
        raise ValueError(f"{field_name} must be an object with 'lower' and 'upper' keys")
    if "lower" not in raw_range or "upper" not in raw_range:
        raise ValueError(f"{field_name} must include both 'lower' and 'upper' HSV triplets")
    lower = _validate_hsv_triplet(raw_range["lower"], f"{field_name}.lower")
    upper = _validate_hsv_triplet(raw_range["upper"], f"{field_name}.upper")
    for idx, (lo, hi) in enumerate(zip(lower, upper)):
        if lo > hi:
            raise ValueError(f"{field_name} lower bound must be <= upper bound at component {idx}; got lower={lower}, upper={upper}")
    return HSVRange(lower, upper)


def _parse_colors(raw_colors: object) -> List[ColorConfig]:
    if not isinstance(raw_colors, list) or not raw_colors:
        raise ValueError("'colors' must be a non-empty list")

    parsed: List[ColorConfig] = []
    names_seen: set[str] = set()
    for item in raw_colors:
        if not isinstance(item, dict):
            raise ValueError("Each color entry must be an object")
        allowed = {"name", "hsv_ranges"}
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise ValueError(f"Unknown color field(s): {', '.join(unknown)}")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Each color entry requires a non-empty 'name'")
        if name in names_seen:
            raise ValueError(f"Duplicate color name: {name}")
        ranges = item.get("hsv_ranges")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError(f"Color '{name}' requires at least one HSV range")

        hsv_ranges: List[HSVRange] = []
        for idx, raw_range in enumerate(ranges):
            hsv_ranges.append(_validate_hsv_range(raw_range, f"colors[{len(parsed)}].hsv_ranges[{idx}]"))

        parsed.append(ColorConfig(name=name, hsv_ranges=hsv_ranges))
        names_seen.add(name)
    return parsed


def _validate_roi(raw_roi: object) -> ROIConfig:
    if raw_roi is None:
        return ROIConfig()
    if not isinstance(raw_roi, dict):
        raise ValueError("'roi' must be a JSON object")
    allowed = {"enabled", "x1", "y1", "x2", "y2"}
    unknown = sorted(set(raw_roi) - allowed)
    if unknown:
        raise ValueError(f"Unknown ROI field(s): {', '.join(unknown)}")
    cfg = ROIConfig()
    cfg.enabled = _ensure_bool(raw_roi.get("enabled", cfg.enabled), "roi.enabled")
    for field_name in ("x1", "y1", "x2", "y2"):
        setattr(cfg, field_name, _ensure_int(raw_roi.get(field_name, getattr(cfg, field_name)), f"roi.{field_name}"))
    if cfg.enabled and (cfg.x2 <= cfg.x1 or cfg.y2 <= cfg.y1):
        raise ValueError(f"roi requires x2 > x1 and y2 > y1 when enabled; got x1={cfg.x1}, x2={cfg.x2}, y1={cfg.y1}, y2={cfg.y2}")
    return cfg


def _validate_detection(raw_detection: object) -> DetectionConfig:
    if raw_detection is None:
        return DetectionConfig()
    if not isinstance(raw_detection, dict):
        raise ValueError("'detection' must be a JSON object")
    allowed = {
        "min_area", "max_area", "min_circularity", "min_solidity",
        "morphology_kernel", "open_iterations", "close_iterations", "min_color_ratio",
    }
    unknown = sorted(set(raw_detection) - allowed)
    if unknown:
        raise ValueError(f"Unknown detection field(s): {', '.join(unknown)}")
    cfg = DetectionConfig()
    cfg.min_area = _ensure_int(raw_detection.get("min_area", cfg.min_area), "detection.min_area", minimum=0, minimum_exclusive=True)
    cfg.max_area = _ensure_int(raw_detection.get("max_area", cfg.max_area), "detection.max_area", minimum=0, minimum_exclusive=True)
    if cfg.max_area < cfg.min_area:
        raise ValueError(f"detection.max_area must be >= detection.min_area; got min_area={cfg.min_area}, max_area={cfg.max_area}")
    cfg.min_circularity = _ensure_number(raw_detection.get("min_circularity", cfg.min_circularity), "detection.min_circularity", minimum=0.0, maximum=1.0)
    cfg.min_solidity = _ensure_number(raw_detection.get("min_solidity", cfg.min_solidity), "detection.min_solidity", minimum=0.0, maximum=1.0)
    cfg.morphology_kernel = _ensure_int(raw_detection.get("morphology_kernel", cfg.morphology_kernel), "detection.morphology_kernel", minimum=1, minimum_exclusive=False)
    if cfg.morphology_kernel % 2 == 0:
        raise ValueError(f"detection.morphology_kernel must be odd, got {cfg.morphology_kernel}")
    cfg.open_iterations = _ensure_int(raw_detection.get("open_iterations", cfg.open_iterations), "detection.open_iterations", minimum=0)
    cfg.close_iterations = _ensure_int(raw_detection.get("close_iterations", cfg.close_iterations), "detection.close_iterations", minimum=0)
    cfg.min_color_ratio = _ensure_number(raw_detection.get("min_color_ratio", cfg.min_color_ratio), "detection.min_color_ratio", minimum=0.0, maximum=1.0)
    return cfg


def _validate_detector(raw_detector: object) -> DetectorConfig:
    if raw_detector is None:
        return DetectorConfig()
    if not isinstance(raw_detector, dict):
        raise ValueError("'detector' must be a JSON object")
    allowed = {"backend", "checkpoint", "metadata", "device", "confidence_threshold"}
    unknown = sorted(set(raw_detector) - allowed)
    if unknown:
        raise ValueError(f"Unknown detector field(s): {', '.join(unknown)}")

    cfg = DetectorConfig()
    backend = raw_detector.get("backend", cfg.backend)
    if not isinstance(backend, str) or backend not in {"hsv", "learned"}:
        raise ValueError(f"detector.backend must be one of ['hsv', 'learned']; got {backend!r}")
    cfg.backend = backend

    for field_name in ("checkpoint", "metadata"):
        value = raw_detector.get(field_name)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"detector.{field_name} must be a non-empty path string")
        setattr(cfg, field_name, value)

    device = raw_detector.get("device", cfg.device)
    if not isinstance(device, str) or not re.fullmatch(r"cpu|auto|cuda(?::[0-9]+)?", device):
        raise ValueError("detector.device must be 'cpu', 'auto', 'cuda', or 'cuda:<index>'")
    cfg.device = device
    cfg.confidence_threshold = _ensure_number(
        raw_detector.get("confidence_threshold", cfg.confidence_threshold),
        "detector.confidence_threshold",
        minimum=0.0,
        maximum=1.0,
    )

    if backend == "learned":
        if cfg.checkpoint is None:
            raise ValueError("detector.checkpoint is required for learned backend")
        if cfg.metadata is None:
            raise ValueError("detector.metadata is required for learned backend")
    elif cfg.checkpoint is not None or cfg.metadata is not None:
        raise ValueError("detector checkpoint/metadata are only valid for learned backend")
    return cfg


def _validate_tracking(raw_tracking: object) -> TrackingConfig:
    if raw_tracking is None:
        return TrackingConfig()
    if not isinstance(raw_tracking, dict):
        raise ValueError("'tracking' must be a JSON object")
    allowed = {"enabled", "max_distance", "max_missing_frames", "min_confirmed_frames", "max_history"}
    unknown = sorted(set(raw_tracking) - allowed)
    if unknown:
        raise ValueError(f"Unknown tracking field(s): {', '.join(unknown)}")
    cfg = TrackingConfig()
    cfg.enabled = _ensure_bool(raw_tracking.get("enabled", cfg.enabled), "tracking.enabled")
    cfg.max_distance = _ensure_number(raw_tracking.get("max_distance", cfg.max_distance), "tracking.max_distance", minimum=0.0, minimum_exclusive=True)
    cfg.max_missing_frames = _ensure_int(raw_tracking.get("max_missing_frames", cfg.max_missing_frames), "tracking.max_missing_frames", minimum=0)
    cfg.min_confirmed_frames = _ensure_int(raw_tracking.get("min_confirmed_frames", cfg.min_confirmed_frames), "tracking.min_confirmed_frames", minimum=1)
    cfg.max_history = _ensure_int(raw_tracking.get("max_history", cfg.max_history), "tracking.max_history", minimum=1)
    return cfg


def _validate_color_stabilization(raw: object) -> ColorStabilizationConfig:
    if raw is None:
        return ColorStabilizationConfig()
    if not isinstance(raw, dict):
        raise ValueError("'color_stabilization' must be a JSON object")
    allowed = {"enabled", "window_size", "min_observations"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown color_stabilization field(s): {', '.join(unknown)}")
    cfg = ColorStabilizationConfig()
    cfg.enabled = _ensure_bool(raw.get("enabled", cfg.enabled), "color_stabilization.enabled")
    cfg.window_size = _ensure_int(
        raw.get("window_size", cfg.window_size),
        "color_stabilization.window_size",
        minimum=1,
    )
    cfg.min_observations = _ensure_int(
        raw.get("min_observations", cfg.min_observations),
        "color_stabilization.min_observations",
        minimum=1,
    )
    if cfg.min_observations > cfg.window_size:
        raise ValueError("color_stabilization.min_observations must not exceed window_size")
    return cfg


def _validate_counting(raw_counting: object) -> CountingConfig:
    if raw_counting is None:
        return CountingConfig()
    if not isinstance(raw_counting, dict):
        raise ValueError("'counting' must be a JSON object")
    allowed = {"enabled", "orientation", "line_position", "direction", "deadband"}
    unknown = sorted(set(raw_counting) - allowed)
    if unknown:
        raise ValueError(f"Unknown counting field(s): {', '.join(unknown)}")
    cfg = CountingConfig()
    cfg.enabled = _ensure_bool(raw_counting.get("enabled", cfg.enabled), "counting.enabled")
    orientation = str(raw_counting.get("orientation", cfg.orientation))
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError(f"counting.orientation must be one of ['horizontal', 'vertical']; got {orientation!r}")
    cfg.orientation = orientation
    cfg.line_position = _ensure_number(raw_counting.get("line_position", cfg.line_position), "counting.line_position", minimum=0.0, maximum=1.0)
    direction = str(raw_counting.get("direction", cfg.direction))
    if direction not in {"positive", "negative", "both"}:
        raise ValueError(f"counting.direction must be one of ['positive', 'negative', 'both']; got {direction!r}")
    cfg.direction = direction
    cfg.deadband = _ensure_number(raw_counting.get("deadband", cfg.deadband), "counting.deadband", minimum=0.0)
    return cfg


def config_to_dict(cfg: AppConfig) -> dict:
    return {
        "detector": cfg.detector.__dict__,
        "roi": cfg.roi.__dict__,
        "detection": cfg.detection.__dict__,
        "tracking": cfg.tracking.__dict__,
        "color_stabilization": cfg.color_stabilization.__dict__,
        "counting": cfg.counting.__dict__,
        "colors": [
            {
                "name": color.name,
                "hsv_ranges": [
                    {"lower": list(rng.lower), "upper": list(rng.upper)}
                    for rng in color.hsv_ranges
                ],
            }
            for color in cfg.colors
        ],
    }


def save_config(cfg: AppConfig, path: str | Path) -> None:
    if not cfg.colors:
        raise ValueError("Cannot save configuration with no colors")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(cfg), indent=2), encoding="utf-8")


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object")

    allowed_root_keys = {"detector", "roi", "detection", "tracking", "color_stabilization", "counting", "colors"}
    unknown_root = sorted(set(data) - allowed_root_keys)
    if unknown_root:
        raise ValueError(f"Unknown config field(s): {', '.join(unknown_root)}")
    if "colors" not in data:
        raise ValueError("Configuration requires a non-empty 'colors' list")

    cfg = AppConfig.default()
    cfg.detector = _validate_detector(data.get("detector"))
    cfg.roi = _validate_roi(data.get("roi", cfg.roi.__dict__))
    cfg.detection = _validate_detection(data.get("detection", cfg.detection.__dict__))
    cfg.tracking = _validate_tracking(data.get("tracking", cfg.tracking.__dict__))
    cfg.color_stabilization = _validate_color_stabilization(
        data.get("color_stabilization", cfg.color_stabilization.__dict__)
    )
    cfg.counting = _validate_counting(data.get("counting", cfg.counting.__dict__))
    cfg.colors = _parse_colors(data["colors"])
    return cfg
