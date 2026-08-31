"""Motion-weighted temporal aggregation for VideoMAE action recognition.

Replaces uniform averaging of per-window predictions with motion-energy weighting.
Evaluates on UCF-Crime labeled test set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cv_pipeline.action import classify_clip, _load_model, LABELS

MODEL_NAME = "OPear/videomae-large-finetuned-UCF-Crime"
_NUM_FRAMES = 16

# UCF-Crime class directories (matching LABELS order)
UCF_CLASSES = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting",
    "Normal Videos", "Road Accidents", "Robbery", "Shooting", "Shoplifting",
    "Stealing", "Vandalism",
]

DIRECTORY_NAME_MAP = {
    "Normal_Videos_for_Event_Recognition": "Normal Videos",
}


def _resolve_class_name(dir_name: str) -> str | None:
    if dir_name in UCF_CLASSES:
        return dir_name
    return DIRECTORY_NAME_MAP.get(dir_name)


def compute_motion_energy(video_path: str | Path,
                          start_frame: int,
                          num_frames: int = _NUM_FRAMES,
                          skip: int = 1) -> float:
    """Compute mean frame-difference magnitude for a window."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_gray = None
    total_diff = 0.0
    diff_count = 0
    frames_read = 0
    idx = start_frame

    while frames_read < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (224, 224))
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                total_diff += np.mean(diff)
                diff_count += 1
            prev_gray = gray
            frames_read += 1
        idx += 1

    cap.release()
    return total_diff / diff_count if diff_count > 0 else 0.0


def aggregate_uniform(predictions: list[dict]) -> dict:
    """Uniform average of per-window probability distributions."""
    if not predictions:
        return {"label": "Unknown", "confidence": 0.0, "probabilities": {}}

    n = len(predictions)
    prob_sum = {label: 0.0 for label in LABELS}

    for pred in predictions:
        for label, prob in pred["probabilities"].items():
            prob_sum[label] += prob

    prob_avg = {label: prob_sum[label] / n for label in LABELS}
    top_label = max(prob_avg, key=prob_avg.get)

    return {
        "label": top_label,
        "confidence": prob_avg[top_label],
        "probabilities": prob_avg,
    }


def aggregate_motion_weighted(predictions: list[dict],
                              motion_energies: list[float]) -> dict:
    """Weighted average by normalized motion energy."""
    if not predictions:
        return {"label": "Unknown", "confidence": 0.0, "probabilities": {}}

    energies = np.array(motion_energies, dtype=np.float32)
    if energies.sum() == 0:
        return aggregate_uniform(predictions)

    weights = energies / energies.sum()

    prob_weighted = {label: 0.0 for label in LABELS}
    for pred, w in zip(predictions, weights):
        for label, prob in pred["probabilities"].items():
            prob_weighted[label] += prob * w

    top_label = max(prob_weighted, key=prob_weighted.get)

    return {
        "label": top_label,
        "confidence": prob_weighted[top_label],
        "probabilities": prob_weighted,
    }


def analyze_clip(
    video_path: str | Path,
    model,
    device,
    window_frames: int = _NUM_FRAMES,
    stride: int = 30,
    motion_skip: int = 1,
    use_motion_weighting: bool = True,
) -> dict:
    """Run sliding window analysis and aggregate predictions."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    predictions = []
    motion_energies = []
    timestamps = []

    for start in range(0, total - window_frames + 1, stride):
        t = start / fps
        timestamps.append(t)

        pred = classify_clip(
            video_path,
            start_frame=start,
            num_frames=window_frames,
            model=model,
            device=device,
        )
        predictions.append(pred)

        energy = compute_motion_energy(video_path, start, window_frames, motion_skip)
        motion_energies.append(energy)

    if use_motion_weighting:
        agg = aggregate_motion_weighted(predictions, motion_energies)
    else:
        agg = aggregate_uniform(predictions)

    return {
        "clip": Path(video_path).name,
        "windows": len(predictions),
        "duration": total / fps,
        "aggregation": agg,
        "per_window": [
            {"time": t, "label": p["label"], "conf": p["confidence"], "motion": m}
            for t, p, m in zip(timestamps, predictions, motion_energies)
        ],
    }


def discover_test_clips(test_root: Path) -> list[tuple[Path, str]]:
    """Discover all labeled clips in UCF-Crime directory structure.

    Handles flat, nested (double-dir), and irregular naming via DIRECTORY_NAME_MAP.

    Returns list of (clip_path, ground_truth_label).
    """
    clips: list[tuple[Path, str]] = []
    seen_classes: set[str] = set()

    def _scan(depth: int, path: Path):
        if depth > 3:
            return
        for child in path.iterdir():
            if not child.is_dir():
                continue
            canonical = _resolve_class_name(child.name)
            if canonical and canonical not in seen_classes:
                seen_classes.add(canonical)
                for vid in child.glob("*.mp4"):
                    clips.append((vid, canonical))
            else:
                _scan(depth + 1, child)

    _scan(0, test_root)
    clips.sort(key=lambda x: (x[1], x[0].name))
    return clips


def evaluate_test_set(
    test_root: Path,
    model,
    device,
    window_frames: int = _NUM_FRAMES,
    stride: int = 30,
    motion_skip: int = 1,
    max_clips_per_class: Optional[int] = None,
    class_filter: Optional[list[str]] = None,
) -> dict:
    """Evaluate both methods on the full labeled test set."""
    clips = discover_test_clips(test_root)
    if not clips:
        raise ValueError(f"No clips found in {test_root}")

    if class_filter:
        clips = [(c, l) for c, l in clips if l in class_filter]
        if not clips:
            raise ValueError(f"No clips found for classes: {class_filter}")

    # Optionally limit clips per class
    if max_clips_per_class:
        from collections import defaultdict
        by_class = defaultdict(list)
        for clip, label in clips:
            by_class[label].append(clip)
        clips = []
        for label, vids in by_class.items():
            clips.extend((v, label) for v in vids[:max_clips_per_class])

    print(f"Found {len(clips)} clips across {len(set(l for _, l in clips))} classes")
    print(f"Evaluating with stride={stride}, window={window_frames}...")

    results = []
    uniform_correct = 0
    weighted_correct = 0
    changed = 0
    changed_toward = 0
    changed_away = 0
    # Normal footage tracking
    normal_clips_total = 0
    uniform_normal_fp = 0
    weighted_normal_fp = 0

    for i, (clip_path, gt_label) in enumerate(clips, 1):
        is_normal = (gt_label == "Normal Videos")
        if is_normal:
            normal_clips_total += 1

        print(f"\n[{i}/{len(clips)}] {clip_path.name}  (GT: {gt_label})")

        uniform = analyze_clip(clip_path, model, device,
                              window_frames, stride, motion_skip, False)
        weighted = analyze_clip(clip_path, model, device,
                               window_frames, stride, motion_skip, True)

        u_label = uniform["aggregation"]["label"]
        w_label = weighted["aggregation"]["label"]

        if is_normal:
            u_correct = (u_label == "Normal Videos")
            w_correct = (w_label == "Normal Videos")
            u_fp = not u_correct
            w_fp = not w_correct
            if u_fp:
                uniform_normal_fp += 1
            if w_fp:
                weighted_normal_fp += 1
        else:
            u_correct = (u_label == gt_label)
            w_correct = (w_label == gt_label)

        if u_correct:
            uniform_correct += 1
        if w_correct:
            weighted_correct += 1

        pred_changed = (u_label != w_label)
        if pred_changed:
            changed += 1
            if w_correct and not u_correct:
                changed_toward += 1
            elif u_correct and not w_correct:
                changed_away += 1

        results.append({
            "clip": clip_path.name,
            "ground_truth": gt_label,
            "uniform": {"label": u_label, "confidence": uniform["aggregation"]["confidence"], "correct": u_correct},
            "motion_weighted": {"label": w_label, "confidence": weighted["aggregation"]["confidence"], "correct": w_correct},
            "changed": pred_changed,
            "change_direction": "toward" if (w_correct and not u_correct) else ("away" if (u_correct and not w_correct) else "neutral"),
        })

        tag = " [NORMAL]" if is_normal else ""
        print(f"  Uniform:     {u_label:<18} {uniform['aggregation']['confidence']:.3f}  {'OK' if u_correct else ('FP' if is_normal else 'NO')}{tag}")
        print(f"  Motion-wt:   {w_label:<18} {weighted['aggregation']['confidence']:.3f}  {'OK' if w_correct else ('FP' if is_normal else 'NO')}{tag}")
        if pred_changed:
            direction = "-> CORRECT" if w_correct else ("-> WRONG" if u_correct else "-> neutral")
            print(f"  CHANGED: {u_label} -> {w_label}  ({direction})")

    n = len(clips)
    normal_fp_rate_u = uniform_normal_fp / normal_clips_total if normal_clips_total else None
    normal_fp_rate_w = weighted_normal_fp / normal_clips_total if normal_clips_total else None
    return {
        "total_clips": n,
        "uniform_accuracy": uniform_correct / n,
        "motion_weighted_accuracy": weighted_correct / n,
        "uniform_correct": uniform_correct,
        "motion_weighted_correct": weighted_correct,
        "predictions_changed": changed,
        "changed_toward_correct": changed_toward,
        "changed_away_from_correct": changed_away,
        "normal_clips_total": normal_clips_total,
        "uniform_normal_false_positives": uniform_normal_fp,
        "weighted_normal_false_positives": weighted_normal_fp,
        "uniform_normal_false_positive_rate": normal_fp_rate_u,
        "weighted_normal_false_positive_rate": normal_fp_rate_w,
        "per_clip": results,
    }


def print_summary_table(eval_results: dict):
    """Print evaluation summary."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total clips evaluated:     {eval_results['total_clips']}")
    print(f"Uniform accuracy:          {eval_results['uniform_accuracy']:.2%}  ({eval_results['uniform_correct']}/{eval_results['total_clips']})")
    print(f"Motion-weighted accuracy:  {eval_results['motion_weighted_accuracy']:.2%}  ({eval_results['motion_weighted_correct']}/{eval_results['total_clips']})")
    print(f"Predictions changed:       {eval_results['predictions_changed']}")
    print(f"  Changed toward correct:  {eval_results['changed_toward_correct']}")
    print(f"  Changed away from correct: {eval_results['changed_away_from_correct']}")

    if eval_results.get("normal_clips_total", 0) > 0:
        print("-" * 70)
        print("NORMAL FOOTAGE FALSE POSITIVE RATE")
        print(f"  Total normal clips:        {eval_results['normal_clips_total']}")
        print(f"  Uniform FP rate:           {eval_results['uniform_normal_false_positive_rate']:.2%}  "
              f"({eval_results['uniform_normal_false_positives']}/{eval_results['normal_clips_total']})")
        print(f"  Motion-weighted FP rate:   {eval_results['weighted_normal_false_positive_rate']:.2%}  "
              f"({eval_results['weighted_normal_false_positives']}/{eval_results['normal_clips_total']})")

    print("=" * 70)

    # Per-class breakdown
    print("\nPer-class accuracy:")
    print(f"{'Class':<20} {'GT':>4} {'Uniform':>8} {'Motion-wt':>10} {'Delta'}")
    print("-" * 55)
    from collections import defaultdict
    class_stats = defaultdict(lambda: {"total": 0, "u_correct": 0, "w_correct": 0})
    for r in eval_results["per_clip"]:
        gt = r["ground_truth"]
        class_stats[gt]["total"] += 1
        if r["uniform"]["correct"]:
            class_stats[gt]["u_correct"] += 1
        if r["motion_weighted"]["correct"]:
            class_stats[gt]["w_correct"] += 1

    for cls in UCF_CLASSES:
        if cls in class_stats:
            s = class_stats[cls]
            u_acc = s["u_correct"] / s["total"] if s["total"] else 0
            w_acc = s["w_correct"] / s["total"] if s["total"] else 0
            delta = w_acc - u_acc
            print(f"{cls:<20} {s['total']:>4}  {u_acc:>7.1%}   {w_acc:>8.1%}   {delta:+.1%}")


def visualize_clip(
    video_path: str | Path,
    model,
    device,
    window_frames: int = _NUM_FRAMES,
    stride: int = 30,
    motion_skip: int = 1,
    output_path: Optional[Path] = None,
) -> Path:
    """Create a composite visualization of window frames with annotations."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Run analysis with motion weighting to get per-window data
    result = analyze_clip(
        video_path, model, device,
        window_frames=window_frames,
        stride=stride,
        motion_skip=motion_skip,
        use_motion_weighting=True,
    )

    per_window = result["per_window"]
    if not per_window:
        raise ValueError("No windows found in clip")

    # Compute normalized weights (motion energies)
    energies = [w["motion"] for w in per_window]
    total_energy = sum(energies)
    weights = [e / total_energy if total_energy > 0 else 1.0 / len(energies) for e in energies]

    # Extract middle frame of each window
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for i, w in enumerate(per_window):
        start_frame = int(w["time"] * fps)
        middle_frame = start_frame + window_frames // 2
        middle_frame = min(middle_frame, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        if ret:
            frames.append((frame, w, weights[i], i))
    cap.release()

    if not frames:
        raise ValueError("Could not extract frames")

    # Build composite image
    frame_h, frame_w = 240, 320  # resized frame size
    pad = 10
    info_h = 120  # height for annotations below each frame
    n = len(frames)
    composite_w = n * (frame_w + pad) + pad
    composite_h = frame_h + info_h + 80  # 80px for header
    composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
    composite[:] = (30, 30, 35)  # dark background

    # Header
    header = f"{Path(video_path).name}  |  {n} windows  |  stride={stride}  |  window={window_frames} frames"
    cv2.putText(composite, header, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(composite, f"Model: {MODEL_NAME.split('/')[-1]}  |  Motion-weighted aggregation", (15, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Place frames and annotations
    for i, (frame, w, weight, idx) in enumerate(frames):
        x = pad + i * (frame_w + pad)
        y = 70

        # Resize and place frame
        small = cv2.resize(frame, (frame_w, frame_h))
        composite[y:y+frame_h, x:x+frame_w] = small

        # Window border color based on weight (green=high, red=low)
        border_color = (0, int(255 * weight), int(255 * (1 - weight)))
        cv2.rectangle(composite, (x, y), (x + frame_w, y + frame_h), border_color, 2)

        # Annotations below frame
        ay = y + frame_h + 5
        t_start = w["time"]
        t_end = t_start + window_frames / (cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else fps)

        lines = [
            f"Window {idx+1}/{n}",
            f"Time: {t_start:.1f}s - {t_end:.1f}s",
            f"Motion: {w['motion']:.3f}  |  Weight: {weight:.3f}",
            f"Pred: {w['label']} ({w['conf']:.2f})",
        ]

        for j, line in enumerate(lines):
            color = (255, 255, 255) if j < 3 else (0, 255, 150)
            cv2.putText(composite, line, (x + 5, ay + j * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Footer with aggregate prediction
    agg = result["aggregation"]
    footer_y = composite_h - 30
    cv2.putText(composite, f"AGGREGATE: {agg['label']}  ({agg['confidence']:.3f})  |  Top-3: "
                f"{', '.join(f'{l[:10]} {p:.2f}' for l,p in sorted(agg['probabilities'].items(), key=lambda x: -x[1])[:3])}",
                (15, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Save
    if output_path is None:
        output_path = Path(video_path).with_stem(f"{Path(video_path).stem}_window_breakdown").with_suffix(".jpg")
    cv2.imwrite(str(output_path), composite)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare uniform vs motion-weighted action aggregation on UCF-Crime test set."
    )
    parser.add_argument(
        "test_root", nargs="?", default="cv_pipeline/test_clips/Anomaly-Videos-Part-1",
        help="Root directory of UCF-Crime test set (class subdirs with .mp4 files)"
    )
    parser.add_argument(
        "--stride", type=int, default=60,
        help="Window stride in frames (default: 60)"
    )
    parser.add_argument(
        "--window", type=int, default=16,
        help="Window size in frames (default: 16)"
    )
    parser.add_argument(
        "--motion-skip", type=int, default=1,
        help="Frame skip for motion energy computation (default: 1)"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Limit clips per class for quick testing (default: all)"
    )
    parser.add_argument(
        "--classes", nargs="+",
        help="Only evaluate these UCF-Crime classes, e.g. --classes Burglary Fighting Vandalism"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("cv_pipeline/mwca_evaluation_results.json"),
        help="Output JSON path for results"
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Run ONLY uniform baseline on specified clips (original mode)"
    )
    parser.add_argument(
        "--clips", nargs="+",
        help="Specific clip paths (original single-clip mode)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Create window breakdown visualization for --clips"
    )

    args = parser.parse_args()

    # Visualization mode
    if args.visualize and args.clips:
        clips = [Path(c) for c in args.clips]
        for clip in clips:
            if not clip.exists():
                print(f"ERROR: clip not found: {clip}")
                sys.exit(1)

        print(f"Loading model: {MODEL_NAME} ...")
        model, device = _load_model()
        print(f"Device: {device}\n")

        for clip in clips:
            print(f"Visualizing: {clip.name}")
            out = visualize_clip(clip, model, device,
                                window_frames=args.window,
                                stride=args.stride,
                                motion_skip=args.motion_skip)
            print(f"Saved: {out}")
        sys.exit(0)

    # Original single-clip mode
    if args.clips:
        clips = [Path(c) for c in args.clips]
        for clip in clips:
            if not clip.exists():
                print(f"ERROR: clip not found: {clip}")
                sys.exit(1)

        print(f"Loading model: {MODEL_NAME} ...")
        model, device = _load_model()
        print(f"Device: {device}\n")

        for clip in clips:
            uniform_result = analyze_clip(
                clip, model, device,
                window_frames=args.window,
                stride=args.stride,
                use_motion_weighting=False,
            )

            if args.baseline:
                agg = uniform_result["aggregation"]
                top3 = sorted(agg["probabilities"].items(), key=lambda x: -x[1])[:3]
                print(f"=== BASELINE (uniform) for {clip.name} ===")
                print(f"Top-1: {agg['label']} ({agg['confidence']:.3f})")
                print(f"Top-3: {', '.join(f'{l} {p:.2f}' for l,p in top3)}")
                continue

            weighted_result = analyze_clip(
                clip, model, device,
                window_frames=args.window,
                stride=args.stride,
                use_motion_weighting=True,
            )

            u = uniform_result["aggregation"]
            w = weighted_result["aggregation"]
            print(f"\n{'='*80}")
            print(f"CLIP: {clip.name}  |  Windows: {uniform_result['windows']}  |  Duration: {uniform_result['duration']:.1f}s")
            print(f"{'='*80}")
            print(f"{'Method':<18} {'Top-1':<18} {'Confidence':>10}  {'Top-3'}")
            print(f"{'-'*80}")
            for method, agg in [("Uniform", u), ("Motion-weighted", w)]:
                top3 = sorted(agg["probabilities"].items(), key=lambda x: -x[1])[:3]
                print(f"{method:<18} {agg['label']:<18} {agg['confidence']:>10.3f}  {', '.join(f'{l[:12]} {p:.2f}' for l,p in top3)}")
        sys.exit(0)

    # Full test set evaluation mode
    test_root = Path(args.test_root)
    if not test_root.exists():
        print(f"ERROR: test root not found: {test_root}")
        sys.exit(1)

    print(f"Loading model: {MODEL_NAME} ...")
    model, device = _load_model()
    print(f"Device: {device}\n")

    eval_results = evaluate_test_set(
        test_root, model, device,
        window_frames=args.window,
        stride=args.stride,
        motion_skip=args.motion_skip,
        max_clips_per_class=args.max_per_class,
        class_filter=args.classes,
    )

    print_summary_table(eval_results)

    # Save results (convert numpy types)
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    args.output.write_text(json.dumps(convert(eval_results), indent=2))
    print(f"\nResults saved to: {args.output}")