"""ByteTrack object tracking on top of the triple-model detection pipeline.

Runs detect_video() to get per-frame raw detections, converts them into
supervision.Detections objects, and feeds them through a single shared
ByteTrack instance for cross-frame track_id continuity.

Key design notes:
  - Tracking treats all detected classes as generic objects by position and
    motion — it does NOT use source_model or class_name as a tracking
    signal, only spatial/motion continuity.
  - Track continuity may be less reliable with skip > 1, since object
    motion between skipped frames could exceed ByteTrack's matching
    threshold — a known compute-cost vs tracking-accuracy trade-off.
"""

import sys
from pathlib import Path

import numpy as np
import supervision as sv

from cv_pipeline.detect import detect_video

CLASS_MAP = {
    "gun": 0,
    "heavy-weapon": 1,
    "Knife": 2,
    "knife": 3,
    "person": 4,
    "bicycle": 5,
    "car": 6,
    "motorcycle": 7,
    "airplane": 8,
    "bus": 9,
    "train": 10,
    "truck": 11,
    "boat": 12,
    "traffic light": 13,
    "fire hydrant": 14,
    "stop sign": 15,
    "parking meter": 16,
    "bench": 17,
    "bird": 18,
    "cat": 19,
    "dog": 20,
    "horse": 21,
    "sheep": 22,
    "cow": 23,
    "elephant": 24,
    "bear": 25,
    "zebra": 26,
    "giraffe": 27,
    "backpack": 28,
    "umbrella": 29,
    "handbag": 30,
    "tie": 31,
    "suitcase": 32,
    "frisbee": 33,
    "skis": 34,
    "snowboard": 35,
    "sports ball": 36,
    "kite": 37,
    "baseball bat": 38,
    "baseball glove": 39,
    "skateboard": 40,
    "surfboard": 41,
    "tennis racket": 42,
    "bottle": 43,
    "wine glass": 44,
    "cup": 45,
    "fork": 46,
    "spoon": 47,
    "bowl": 48,
    "banana": 49,
    "apple": 50,
    "sandwich": 51,
    "orange": 52,
    "broccoli": 53,
    "carrot": 54,
    "hot dog": 55,
    "pizza": 56,
    "donut": 57,
    "cake": 58,
    "chair": 59,
    "couch": 60,
    "potted plant": 61,
    "bed": 62,
    "dining table": 63,
    "toilet": 64,
    "tv": 65,
    "laptop": 66,
    "mouse": 67,
    "remote": 68,
    "keyboard": 69,
    "cell phone": 70,
    "microwave": 71,
    "oven": 72,
    "toaster": 73,
    "sink": 74,
    "refrigerator": 75,
    "book": 76,
    "clock": 77,
    "vase": 78,
    "scissors": 79,
    "teddy bear": 80,
    "hair drier": 81,
    "toothbrush": 82,
}


def track_video(video_path, conf_weapon=0.5, conf_knife=0.7, conf_coco=0.5, skip=5):
    """Run detection + ByteTrack tracking on a video.

    Yields (frame_idx, tracked_detections) where each entry is a list of
    dicts with keys: bbox, class_name, confidence, source_model, track_id,
    class_changed.

    Tracking uses a single shared ByteTrack instance across the whole video
    for track_id continuity. All classes are tracked simultaneously by
    spatial/motion continuity only.

    A class-consistency guard detects when ByteTrack reuses a track_id for a
    different class_name than the one previously associated with it. When
    this happens the logical track_id is reassigned with a suffix (e.g.
    "2_b") to prevent silent conflation of two different real-world objects,
    and class_changed=True is set on that detection.
    """
    tracker = sv.ByteTrack()
    tid_class_map: dict[int, str] = {}
    tid_suffix_counter: dict[int, int] = {}
    tid_logical_map: dict[int, str] = {}

    for frame_idx, dets in detect_video(video_path, skip=skip,
                                        conf_weapon=conf_weapon,
                                        conf_knife=conf_knife,
                                        conf_coco=conf_coco):
        if not dets:
            yield frame_idx, []
            continue

        xyxy = np.array([d["bbox"] for d in dets], dtype=np.float32)
        confidence = np.array([d["confidence"] for d in dets], dtype=np.float32)
        class_id = np.array([CLASS_MAP.get(d["class_name"], -1) for d in dets],
                            dtype=np.int32)

        sv_dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = tracker.update_with_detections(sv_dets)

        results = []
        for i in range(len(tracked)):
            raw_tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            cls = dets[i]["class_name"]
            class_changed = False

            if raw_tid != -1:
                prev_cls = tid_class_map.get(raw_tid)
                if prev_cls is not None and prev_cls != cls:
                    class_changed = True
                    count = tid_suffix_counter.get(raw_tid, 0) + 1
                    tid_suffix_counter[raw_tid] = count
                    logical = f"{raw_tid}_{chr(96 + count)}"
                    tid_logical_map[(raw_tid, cls)] = logical
                else:
                    tid_class_map[raw_tid] = cls

                logical = tid_logical_map.get((raw_tid, cls), str(raw_tid))
            else:
                logical = "-1"

            results.append({
                "bbox": dets[i]["bbox"],
                "class_name": cls,
                "confidence": dets[i]["confidence"],
                "source_model": dets[i]["source_model"],
                "track_id": logical,
                "class_changed": class_changed,
            })
        yield frame_idx, results


if __name__ == "__main__":
    from collections import defaultdict

    clips = [
        ("positive_sample.mp4", "GUN CLIP"),
        ("positive_sample5.mp4", "STREET SCENE"),
    ]
    base = Path(__file__).resolve().parent / "test_clips"

    for filename, label in clips:
        video = base / filename
        if not video.exists():
            print(f"ERROR: video not found: {video}")
            continue

        track_log = defaultdict(list)

        print(f"\n{'='*60}")
        print(f"  {label}: {filename}")
        print(f"{'='*60}")
        print(f"{'frame_idx':<10} {'class_name':<15} {'track_id':<10} {'conf':<7} {'changed'}")
        print("-" * 52)

        for frame_idx, tracked_dets in track_video(str(video)):
            for d in tracked_dets:
                flag = "  ***" if d["class_changed"] else ""
                print(f"{frame_idx:<10} {d['class_name']:<15} {d['track_id']:<10} "
                      f"{d['confidence']:<7.3f} {flag}")
                track_log[d["track_id"]].append((frame_idx, d["class_name"]))

        print(f"\n  TRACK SUMMARY (track_ids appearing > 5 times)")
        print(f"  {'-'*55}")
        for tid in sorted(track_log):
            entries = track_log[tid]
            if len(entries) <= 5:
                continue
            frames = [e[0] for e in entries]
            classes = set(e[1] for e in entries)
            multi_class = "  ** CLASS CHANGE **" if len(classes) > 1 else ""
            print(f"\n  track_id={tid}  class={entries[0][1]}  count={len(frames)}{multi_class}")
            print(f"    frames: {frames}")
