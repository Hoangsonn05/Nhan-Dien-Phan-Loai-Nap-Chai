"""Trace learned detections through counting for the canonical conveyor-v2 fixture.

This is diagnostic-only tooling.  It deliberately observes the existing
detector, tracker, color stabilizer, and counter without changing their
configuration or behavior.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bottle_cap_vision.detector import create_detector
from bottle_cap_vision.models import Detection, Track
from bottle_cap_vision.pipeline import BottleCapPipeline
from bottle_cap_vision.practical_demo import load_learned_config
from bottle_cap_vision.sources import VideoSource


DEFAULT_VIDEO = ROOT / "data" / "input" / "videos" / "bottle_cap_conveyor_v2.mp4"
DEFAULT_CONFIG = ROOT / "configs" / "config.learned-p9-full.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "diagnostics" / "video_count_v2"
EXPECTED_COLOR_COUNTS = {
    "Green": 2,
    "Orange": 1,
    "Purple": 2,
    "Red": 2,
    "White": 1,
}


def _json_write(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _image_write(path: Path, image: np.ndarray) -> None:
    """Write through a byte buffer because OpenCV paths are not Unicode-safe on Windows."""
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode diagnostic image: {path}")
    path.write_bytes(encoded.tobytes())


def _detection_record(detection: Detection) -> dict[str, object]:
    return {
        "bbox": list(detection.bbox),
        "centroid": list(detection.centroid),
        "class_id": detection.class_id,
        "color": detection.color,
        "confidence": detection.confidence,
    }


def _track_record(track: Track) -> dict[str, object]:
    return {
        "age": track.age,
        "bbox": list(track.bbox),
        "centroid": list(track.centroid),
        "confirmed": track.confirmed,
        "counted": track.counted,
        "missing": track.missing,
        "observed_color": track.observed_color,
        "stable_color": track.color,
        "track_id": track.track_id,
    }


def _bbox_overlap(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> tuple[float, float]:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a, area_b = aw * ah, bw * bh
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    containment = intersection / min(area_a, area_b) if min(area_a, area_b) else 0.0
    return iou, containment


def _associated_detection(track: Track, detections: Iterable[Detection]) -> Detection | None:
    if track.missing:
        return None
    matches = [
        detection
        for detection in detections
        if detection.bbox == track.bbox and detection.centroid == track.centroid
    ]
    return matches[0] if len(matches) == 1 else None


def _foreground_components(frame: np.ndarray, background: np.ndarray) -> list[dict[str, object]]:
    """Extract the synthetic moving caps independently of learned inference."""
    difference = cv2.absdiff(frame, background)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray > 18, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    components: list[dict[str, object]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area < 300:
            continue
        components.append(
            {
                "area": area,
                "bbox": [x, y, width, height],
                "centroid": [float(centroids[index][0]), float(centroids[index][1])],
            }
        )
    return components


def _decode_fixture(video_path: Path, line_position: float) -> tuple[dict[str, object], dict[int, dict[int, list[float]]], dict[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_frames = 0
    background: np.ndarray | None = None
    samples: dict[int, np.ndarray] = {}
    foreground_tracks: dict[int, dict[str, object]] = {}
    next_track_id = 1

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index = decoded_frames
        decoded_frames += 1
        if background is None:
            background = frame.copy()
        if frame_index in {0, 15, 30, 45, 60, 75, 78, 90, 105, 119}:
            samples[frame_index] = frame.copy()

        components = _foreground_components(frame, background)
        active_ids = [
            track_id
            for track_id, state in foreground_tracks.items()
            if frame_index - int(state["last_frame"]) <= 2
        ]
        candidates: list[tuple[float, int, int]] = []
        for track_id in active_ids:
            previous = foreground_tracks[track_id]["points"][-1]["centroid"]
            for component_index, component in enumerate(components):
                candidates.append(
                    (
                        math.dist(previous, component["centroid"]),
                        track_id,
                        component_index,
                    )
                )
        unmatched_tracks = set(active_ids)
        unmatched_components = set(range(len(components)))
        for distance, track_id, component_index in sorted(candidates):
            if (
                distance > 40.0
                or track_id not in unmatched_tracks
                or component_index not in unmatched_components
            ):
                continue
            component = components[component_index]
            foreground_tracks[track_id]["points"].append(
                {"frame_index": frame_index, **component}
            )
            foreground_tracks[track_id]["last_frame"] = frame_index
            unmatched_tracks.remove(track_id)
            unmatched_components.remove(component_index)
        for component_index in sorted(unmatched_components):
            foreground_tracks[next_track_id] = {
                "last_frame": frame_index,
                "points": [{"frame_index": frame_index, **components[component_index]}],
            }
            next_track_id += 1
    capture.release()
    if background is None:
        raise RuntimeError(f"Video decoded no frames: {video_path}")

    line_y = height * line_position
    objects: list[dict[str, object]] = []
    foreground_by_frame: dict[int, dict[int, list[float]]] = defaultdict(dict)
    for state in foreground_tracks.values():
        points = state["points"]
        if len(points) < 5:
            continue
        transitions = []
        previous_y = float(points[0]["centroid"][1])
        for point in points[1:]:
            current_y = float(point["centroid"][1])
            if previous_y <= line_y < current_y:
                transitions.append(int(point["frame_index"]))
            previous_y = current_y
        object_id = len(objects) + 1
        for point in points:
            foreground_by_frame[int(point["frame_index"])][object_id] = point["centroid"]
        objects.append(
            {
                "object_id": object_id,
                "first_frame": int(points[0]["frame_index"]),
                "last_frame": int(points[-1]["frame_index"]),
                "observations": len(points),
                "mean_x": sum(float(point["centroid"][0]) for point in points) / len(points),
                "positive_crossing_frames": transitions,
            }
        )

    y = int(round(line_y))
    row_delta = np.abs(
        background[y].astype(np.int16)
        - (
            background[y - 3].astype(np.int16)
            + background[y + 3].astype(np.int16)
        )
        / 2.0
    )
    high_contrast_fraction = float((row_delta.max(axis=1) > 30).mean())
    no_embedded_line = high_contrast_fraction < 0.10
    fixture_valid = (
        reported_frames == decoded_frames == 120
        and width == 1280
        and height == 720
        and abs(fps - 15.0) < 1e-6
        and len(objects) == 8
        and all(len(item["positive_crossing_frames"]) == 1 for item in objects)
        and no_embedded_line
    )
    fixture = {
        "decoded_frames": decoded_frames,
        "expected_color_counts": EXPECTED_COLOR_COUNTS,
        "expected_crossings": 8,
        "expected_objects": 8,
        "fixture_valid": fixture_valid,
        "foreground_method": {
            "description": "absdiff from empty frame 0, gray threshold 18, 5x5 open, component area >= 300",
            "purpose": "independent physical-region count; not detector evaluation",
        },
        "fps": fps,
        "height": height,
        "no_counting_line_embedded": no_embedded_line,
        "physical_objects": objects,
        "reported_frames": reported_frames,
        "sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
        "video": str(video_path),
        "width": width,
        "line_pixel_check": {
            "high_contrast_column_fraction": high_contrast_fraction,
            "row": y,
            "threshold": 0.10,
        },
    }
    return fixture, foreground_by_frame, samples


def _nearest_fixture_object(
    frame_index: int,
    centroid: Iterable[float],
    foreground_by_frame: dict[int, dict[int, list[float]]],
) -> tuple[int | None, float | None]:
    candidates = foreground_by_frame.get(frame_index, {})
    if not candidates:
        return None, None
    distance, object_id = min(
        (math.dist(centroid, object_centroid), object_id)
        for object_id, object_centroid in candidates.items()
    )
    return object_id, distance


def _render_contact_sheet(samples: dict[int, np.ndarray], output_path: Path) -> None:
    indices = [0, 15, 30, 45, 60, 75, 78, 90, 105, 119]
    tile_width, tile_height = 512, 288
    canvas = np.zeros((tile_height * 2, tile_width * 5, 3), dtype=np.uint8)
    for offset, frame_index in enumerate(indices):
        frame = cv2.resize(samples[frame_index], (tile_width, tile_height))
        cv2.putText(
            frame,
            f"frame {frame_index}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        row, column = divmod(offset, 5)
        canvas[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = frame
    _image_write(output_path, canvas)


def diagnose(video_path: Path, config_path: Path, output_dir: Path) -> dict[str, object]:
    config, resolved_config, checkpoint, metadata = load_learned_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture, foreground_by_frame, samples = _decode_fixture(
        video_path, config.counting.line_position
    )
    _json_write(output_dir / "fixture_validation.json", fixture)
    _render_contact_sheet(samples, output_dir / "fixture_contact_sheet.jpg")

    detector = create_detector(config)
    pipeline = BottleCapPipeline(config, detector=detector)
    source = VideoSource(video_path)
    frame_records: list[dict[str, object]] = []
    count_events: list[dict[str, object]] = []
    track_votes: dict[int, Counter[int]] = defaultdict(Counter)
    event_frames: dict[int, np.ndarray] = {}
    source.open()
    try:
        for packet in source:
            result = pipeline.process_frame(packet.frame, packet.frame_index)
            detections = [_detection_record(item) for item in result.detections]
            tracks = []
            for track in result.tracks:
                associated = _associated_detection(track, result.detections)
                record = _track_record(track)
                record["associated_detection"] = (
                    _detection_record(associated) if associated is not None else None
                )
                record["stabilizer_history"] = list(
                    pipeline.color_stabilizer.history_for(track.track_id)
                )
                fixture_object, distance = _nearest_fixture_object(
                    packet.frame_index, track.centroid, foreground_by_frame
                )
                record["nearest_fixture_object"] = fixture_object
                record["fixture_centroid_distance"] = distance
                if track.missing == 0 and fixture_object is not None and distance is not None and distance <= 60:
                    track_votes[track.track_id][fixture_object] += 1
                tracks.append(record)
            for event in result.events:
                track = next(item for item in tracks if item["track_id"] == event.track_id)
                count_events.append(
                    {
                        "event_index": len(count_events) + 1,
                        **asdict(event),
                        "centroid": track["centroid"],
                        "bbox": track["bbox"],
                        "observed_color": track["observed_color"],
                        "stable_color": track["stable_color"],
                        "nearest_fixture_object": track["nearest_fixture_object"],
                        "fixture_centroid_distance": track["fixture_centroid_distance"],
                    }
                )
                event_frames[event.track_id] = packet.frame.copy()
            frame_records.append(
                {
                    "frame_index": packet.frame_index,
                    "detections": detections,
                    "tracks": tracks,
                }
            )
    finally:
        source.close()

    _json_write(output_dir / "count_events.json", count_events)
    blue_events = [item for item in count_events if item["color"] == "Blue"]
    if len(blue_events) != 1:
        raise RuntimeError(f"Expected exactly one Blue event; got {len(blue_events)}")
    blue_event = blue_events[0]
    blue_track_id = int(blue_event["track_id"])
    crossing_frame = int(blue_event["frame_index"])
    blue_history = [
        {"frame_index": frame["frame_index"], **track}
        for frame in frame_records
        for track in frame["tracks"]
        if track["track_id"] == blue_track_id
    ]
    associated_predictions = [
        item["associated_detection"]
        for item in blue_history
        if item["associated_detection"] is not None
    ]
    blue_payload = {
        "track_id": blue_track_id,
        "crossing_frame": crossing_frame,
        "lifetime": {
            "first_frame": blue_history[0]["frame_index"],
            "last_frame": blue_history[-1]["frame_index"],
            "present_frames": len(blue_history),
        },
        "observed_color_counts": dict(Counter(item["observed_color"] for item in blue_history)),
        "stable_color_counts": dict(Counter(item["stable_color"] for item in blue_history)),
        "detector_classes": sorted({item["color"] for item in associated_predictions}),
        "confidence_range": [
            min(item["confidence"] for item in associated_predictions),
            max(item["confidence"] for item in associated_predictions),
        ],
        "focus_window": [crossing_frame - 10, crossing_frame + 10],
        "history": blue_history,
    }
    _json_write(output_dir / "blue_track_history.json", blue_payload)

    relevant_frames = [
        {
            "frame_index": frame["frame_index"],
            "detections": frame["detections"],
            "in_focus_window": crossing_frame - 10 <= frame["frame_index"] <= crossing_frame + 10,
        }
        for frame in frame_records
        if any(track["track_id"] == blue_track_id for track in frame["tracks"])
    ]
    _json_write(output_dir / "relevant_frame_detections.json", relevant_frames)

    crossing_record = frame_records[crossing_frame]
    overlapping_pairs = []
    all_duplicate_pairs = []
    for frame in frame_records:
        detections = frame["detections"]
        for first_index, first in enumerate(detections):
            for second in detections[first_index + 1 :]:
                iou, containment = _bbox_overlap(tuple(first["bbox"]), tuple(second["bbox"]))
                if first["color"] == second["color"] or (iou < 0.5 and containment < 0.8):
                    continue
                pair = {
                    "frame_index": frame["frame_index"],
                    "first": first,
                    "second": second,
                    "iou": iou,
                    "intersection_over_smaller": containment,
                    "centroid_distance": math.dist(first["centroid"], second["centroid"]),
                }
                all_duplicate_pairs.append(pair)
                if frame["frame_index"] == crossing_frame:
                    overlapping_pairs.append(pair)
    overlap = {
        "duplicate_detection_present": bool(overlapping_pairs),
        "blue_crossing_frame": crossing_frame,
        "crossing_frame_detections": crossing_record["detections"],
        "crossing_frame_cross_class_overlaps": overlapping_pairs,
        "all_cross_class_duplicate_pairs": all_duplicate_pairs,
    }
    _json_write(output_dir / "detection_overlap.json", overlap)

    mapping = []
    for fixture_object in fixture["physical_objects"]:
        object_id = int(fixture_object["object_id"])
        track_ids = sorted(
            track_id
            for track_id, votes in track_votes.items()
            if votes and votes.most_common(1)[0][0] == object_id
        )
        mapped_events = [
            event for event in count_events if event["nearest_fixture_object"] == object_id
        ]
        mapping.append(
            {
                **fixture_object,
                "track_ids": track_ids,
                "counted_track_ids": [event["track_id"] for event in mapped_events],
                "counted_colors": [event["color"] for event in mapped_events],
                "count_event_frames": [event["frame_index"] for event in mapped_events],
            }
        )
    _json_write(output_dir / "physical_object_mapping.json", mapping)

    track_event_frequency = Counter(event["track_id"] for event in count_events)
    duplicate_same_track = any(count > 1 for count in track_event_frequency.values())
    blue_object = int(blue_event["nearest_fixture_object"])
    blue_object_mapping = next(item for item in mapping if item["object_id"] == blue_object)
    detector_blue_at_crossing = any(
        detection["color"] == "Blue" for detection in crossing_record["detections"]
    )
    if not fixture["fixture_valid"]:
        root_cause = "FIXTURE_DEFECT"
    elif duplicate_same_track:
        root_cause = "COUNTER_DEFECT"
    elif (
        detector_blue_at_crossing
        and overlap["duplicate_detection_present"]
        and len(blue_object_mapping["counted_track_ids"]) > 1
    ):
        root_cause = "DUPLICATE_DETECTION"
    elif detector_blue_at_crossing:
        root_cause = "DETECTOR_FALSE_POSITIVE"
    else:
        root_cause = "INSUFFICIENT_EVIDENCE"

    summary = {
        "config": str(resolved_config),
        "checkpoint": str(checkpoint),
        "metadata": str(metadata),
        "effective_config": {
            "confidence_threshold": config.detector.confidence_threshold,
            "tracking": asdict(config.tracking),
            "color_stabilization": asdict(config.color_stabilization),
            "counting": asdict(config.counting),
        },
        "fixture_valid": fixture["fixture_valid"],
        "frames_processed": len(frame_records),
        "total_count": len(count_events),
        "count_by_color": dict(Counter(event["color"] for event in count_events)),
        "blue_event": blue_event,
        "blue_physical_object": blue_object,
        "blue_object_counted_track_ids": blue_object_mapping["counted_track_ids"],
        "counter_duplicate_same_track_event": duplicate_same_track,
        "count_event_track_frequencies": dict(track_event_frequency),
        "duplicate_detection_at_blue_crossing": overlap["duplicate_detection_present"],
        "root_cause": root_cause,
        "production_code_changed": False,
    }
    _json_write(output_dir / "root_cause_summary.json", summary)

    crossing_image = event_frames[blue_track_id]
    line_y = int(round(crossing_image.shape[0] * config.counting.line_position))
    cv2.line(crossing_image, (0, line_y), (crossing_image.shape[1] - 1, line_y), (0, 255, 255), 2)
    for detection in crossing_record["detections"]:
        x, y, width, height = detection["bbox"]
        color = (255, 0, 0) if detection["color"] == "Blue" else (255, 255, 255)
        cv2.rectangle(crossing_image, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            crossing_image,
            f"{detection['color']} {detection['confidence']:.3f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    _image_write(output_dir / "blue_crossing_frame_78.jpg", crossing_image)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the extra Blue count in the canonical conveyor-v2 video"
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = diagnose(args.video.resolve(), args.config.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
