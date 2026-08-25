"""Calibration module for detector confidence thresholds.

Runs a held-out negative set through each detector, sweeps confidence
thresholds, and selects the lowest threshold where false positives drop
to zero (or below a configurable maximum).

Outputs a JSON config: {model_name: chosen_threshold}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEAPON_WEIGHTS = WEIGHTS_DIR / "weapon_detect_v1_best.pt"
KNIFE_WEIGHTS = WEIGHTS_DIR / "knife_detect_v2_best.pt"

WEAPON_KEEP = {"gun", "heavy-weapon"}
KNIFE_KEEP = {"Knife"}

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parent / "calibrated_thresholds.json"
THRESHOLDS = [round(0.1 + i * 0.05, 2) for i in range(17)]  # 0.10, 0.15, ..., 0.90


def _load_model(weights_path: Path):
    return YOLO(str(weights_path))


def _count_detections_at_threshold(model, video_path: Path, keep_classes: set[str],
                                    conf: float, skip: int = 1) -> int:
    """Count detections of kept classes at a given confidence threshold."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0

    total = 0
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip == 0:
                results = model(frame, conf=conf, verbose=False)
                for r in results:
                    names = r.names
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = names.get(cls_id, f"cls_{cls_id}")
                        if cls_name in keep_classes:
                            total += 1
            frame_idx += 1
    finally:
        cap.release()
    return total


def calibrate_model(
    model_name: str,
    weights_path: Path,
    keep_classes: set[str],
    negative_clips: list[Path],
    thresholds: list[float] = THRESHOLDS,
    max_fp: int = 0,
    skip: int = 1,
) -> tuple[float, dict[float, int]]:
    """Calibrate a single model.

    Returns:
        chosen_threshold: lowest threshold where total FPs <= max_fp
        fp_counts: dict mapping threshold -> total FP count across all negative clips
    """
    model = _load_model(weights_path)

    fp_counts: dict[float, int] = {}
    for conf in thresholds:
        total_fp = 0
        for clip in negative_clips:
            total_fp += _count_detections_at_threshold(model, clip, keep_classes, conf, skip)
        fp_counts[conf] = total_fp

    # Find lowest threshold where FPs <= max_fp
    chosen = None
    for conf in thresholds:
        if fp_counts[conf] <= max_fp:
            chosen = conf
            break

    # If no threshold meets the criterion, fall back to highest threshold
    if chosen is None:
        chosen = thresholds[-1]

    return chosen, fp_counts


def calibrate_all(
    negative_clips: list[Path],
    max_fp: int = 0,
    skip: int = 1,
    output_path: Optional[Path] = None,
) -> dict[str, float]:
    """Calibrate both weapon and knife models.

    Args:
        negative_clips: list of video paths with NO weapons/knives
        max_fp: maximum allowed false positives (default 0)
        skip: process every Nth frame (default 1 for thorough calibration)
        output_path: where to save JSON config (default: calibrated_thresholds.json)

    Returns:
        dict with chosen thresholds for each model
    """
    if output_path is None:
        output_path = DEFAULT_CALIBRATION_PATH

    print(f"Calibrating on {len(negative_clips)} negative clip(s)...")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Max allowed FPs: {max_fp}")
    print(f"Frame skip: {skip}")
    print("-" * 60)

    # Calibrate weapon model
    print(f"\n[{WEAPON_WEIGHTS.name}] keep={WEAPON_KEEP}")
    weapon_thresh, weapon_fps = calibrate_model(
        "weapon_v1", WEAPON_WEIGHTS, WEAPON_KEEP, negative_clips,
        max_fp=max_fp, skip=skip
    )
    for conf, count in weapon_fps.items():
        marker = "  <-- CHOSEN" if conf == weapon_thresh else ""
        print(f"  conf={conf:.2f}: {count} FP{marker}")

    # Calibrate knife model
    print(f"\n[{KNIFE_WEIGHTS.name}] keep={KNIFE_KEEP}")
    knife_thresh, knife_fps = calibrate_model(
        "knife_v2", KNIFE_WEIGHTS, KNIFE_KEEP, negative_clips,
        max_fp=max_fp, skip=skip
    )
    for conf, count in knife_fps.items():
        marker = "  <-- CHOSEN" if conf == knife_thresh else ""
        print(f"  conf={conf:.2f}: {count} FP{marker}")

    result = {
        "weapon_v1": round(weapon_thresh, 2),
        "knife_v2": round(knife_thresh, 2),
    }

    output_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved calibration to: {output_path}")
    print(f"Result: {result}")

    return result


def load_calibrated_thresholds(config_path: Optional[Path] = None) -> dict[str, float]:
    """Load calibrated thresholds from JSON, with fallback defaults."""
    if config_path is None:
        config_path = DEFAULT_CALIBRATION_PATH

    defaults = {
        "weapon_v1": 0.5,
        "knife_v2": 0.65,
    }

    if not config_path.exists():
        return defaults

    try:
        data = json.loads(config_path.read_text())
        # Merge with defaults in case keys are missing
        result = {k: data.get(k, v) for k, v in defaults.items()}
        return result
    except (json.JSONDecodeError, KeyError):
        return defaults


def _check_high_conf_detections(model, video_path: Path, keep_classes: set[str],
                                 thresholds: list[float], skip: int) -> list[tuple[float, float]]:
    """Check for suspiciously high-confidence detections at the lowest threshold.

    Returns list of (threshold, max_conf) where max_conf >= 0.9
    """
    # Only check at lowest threshold (most detections)
    conf = thresholds[0]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []

    frame_idx = 0
    max_conf_at_thresh = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % skip == 0:
                results = model(frame, conf=conf, verbose=False)
                for r in results:
                    names = r.names
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = names.get(cls_id, f"cls_{cls_id}")
                        if cls_name in keep_classes:
                            c = float(box.conf[0])
                            if c > max_conf_at_thresh:
                                max_conf_at_thresh = c
            frame_idx += 1
    finally:
        cap.release()

    if max_conf_at_thresh >= 0.9:
        return [(conf, max_conf_at_thresh)]
    return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate detector confidence thresholds on true negative clips."
    )
    parser.add_argument(
        "--negative-clip", type=Path, required=True,
        help="Path to a true negative video clip (no weapons/knives present)"
    )
    parser.add_argument(
        "--max-fp", type=int, default=0,
        help="Maximum allowed false positives (default: 0)"
    )
    parser.add_argument(
        "--skip", type=int, default=10,
        help="Process every Nth frame for speed (default: 10)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_CALIBRATION_PATH,
        help=f"Output JSON path (default: {DEFAULT_CALIBRATION_PATH})"
    )
    parser.add_argument(
        "--compare-manual", action="store_true",
        help="Print comparison with manual thresholds (weapon=0.5, knife=0.65)"
    )

    args = parser.parse_args()

    if not args.negative_clip.exists():
        print(f"ERROR: clip not found: {args.negative_clip}")
        sys.exit(1)

    # Load models once for high-conf check
    weapon_model = _load_model(WEAPON_WEIGHTS)
    knife_model = _load_model(KNIFE_WEIGHTS)

    # Sanity check: warn if high-conf detections found on "negative" clip
    print("Running high-confidence sanity check on negative clip...")
    weapon_suspicious = _check_high_conf_detections(
        weapon_model, args.negative_clip, WEAPON_KEEP, THRESHOLDS, args.skip
    )
    knife_suspicious = _check_high_conf_detections(
        knife_model, args.negative_clip, KNIFE_KEEP, THRESHOLDS, args.skip
    )

    all_suspicious = {}
    if weapon_suspicious:
        all_suspicious["weapon_v1"] = weapon_suspicious
    if knife_suspicious:
        all_suspicious["knife_v2"] = knife_suspicious

    if all_suspicious:
        print("\n" + "!" * 60)
        print("WARNING: HIGH-CONFIDENCE DETECTIONS FOUND ON NEGATIVE CLIP")
        print("This suggests the clip may NOT be a true negative.")
        print("!" * 60)
        for model, entries in all_suspicious.items():
            for conf, max_conf in entries:
                print(f"  {model}: at threshold {conf:.2f}, max confidence = {max_conf:.3f}")
        print("!" * 60 + "\n")
    else:
        print("  No high-confidence detections (>= 0.9) found. Clip appears clean.\n")

    # Run calibration
    result = calibrate_all([args.negative_clip], max_fp=args.max_fp, skip=args.skip, output_path=args.output)

    if args.compare_manual:
        print("\n" + "=" * 60)
        print("COMPARISON WITH MANUAL THRESHOLDS")
        print("=" * 60)
        manual = {"weapon_v1": 0.5, "knife_v2": 0.65}
        for model in ["weapon_v1", "knife_v2"]:
            auto = result[model]
            man = manual[model]
            diff = auto - man
            status = "MATCH" if diff == 0 else ("HIGHER" if diff > 0 else "LOWER")
            print(f"  {model}: auto={auto:.2f}  manual={man:.2f}  diff={diff:+.2f}  ({status})")