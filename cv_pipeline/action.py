"""
Action / violence recognition using a pretrained VideoMAE model.

Model: OPear/videomae-large-finetuned-UCF-Crime
        (fine-tuned from MCG-NJU/videomae-large)

Classifies a short video clip (16 sampled frames) into one of 14 UCF-Crime
categories:
    Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting,
    Normal Videos, Road Accidents, Robbery, Shooting, Shoplifting,
    Stealing, Vandalism

-----
DOMAIN SHIFT WARNING (from the model card):

    "This model is not suitable for scenarios where input data deviates
    significantly from the types of videos in the UCF Crime dataset."

UCF-Crime consists of specific surveillance camera clips with particular
resolutions, angles, and scene types.  Our own CCTV test footage almost
certainly differs in all of these respects — the same kind of domain shift
that degraded our knife detector's precision on real-world clips.  Treat
every prediction below as an honest experiment, not an assumed success.
If the model seems confident, that is encouraging but not proof that it
generalises to our footage.  If it seems confused, that is informative
and expected.
-----
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import cv2
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification

# ---------------------------------------------------------------------------
# Class labels (index → name)
# ---------------------------------------------------------------------------
LABELS: list[str] = [
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "Normal Videos",
    "Road Accidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]

MODEL_NAME = "OPear/videomae-large-finetuned-UCF-Crime"
KINETICS_MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
_NUM_FRAMES = 16
_SIZE = (224, 224)

# Kinetics-400 labels loaded lazily from model config
_kinetics_labels: dict[int, str] | None = None


def _load_model(device: str | torch.device | None = None):
    """Lazy-load the model (downloaded once, cached by HuggingFace)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.environ.get("HF_TOKEN")
    model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME, token=token)
    model.to(device).eval()
    return model, device


def _extract_frames(
    video_path: str | Path,
    start_frame: int = 0,
    num_frames: int = _NUM_FRAMES,
) -> torch.Tensor:
    """Return a tensor of shape [1, 3, num_frames, 224, 224].

    Samples *num_frames* evenly-spaced frames starting at *start_frame*.
    If the clip is shorter than requested the last available frame is
    duplicated to pad.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    upper = max(start_frame, total - 1)
    target_indices = set(np.linspace(start_frame, upper, num_frames, dtype=int))

    frames: list[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in target_indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(rgb, _SIZE))
        idx += 1
    cap.release()

    # Pad with last frame if clip was too short
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((*_SIZE, 3), dtype=np.uint8))

    arr = np.stack(frames, dtype=np.float32)       # [N, 224, 224, 3]
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2) / 255.0  # [N, 3, 224, 224]
    return tensor.unsqueeze(0)                      # [1, 3, N, 224, 224]


def classify_clip(
    video_path: str | Path,
    start_frame: int = 0,
    num_frames: int = _NUM_FRAMES,
    *,
    model: Optional[VideoMAEForVideoClassification] = None,
    device: Optional[str | torch.device] = None,
) -> dict:
    """Classify a video clip using the pretrained VideoMAE UCF-Crime model.

    Parameters
    ----------
    video_path : path to any video file OpenCV can read.
    start_frame : first frame index to sample from (default 0).
    num_frames : how many frames to sample (default 16, the model's native
                 input size).
    model, device : optional pre-loaded model / device to avoid re-loading
                    on repeated calls.

    Returns
    -------
    dict with keys:
        label          – name of the top-1 predicted class
        confidence     – probability of the top-1 class (0–1)
        probabilities  – dict mapping every class name to its probability
    """
    if model is None:
        model, device = _load_model(device)

    video_tensor = _extract_frames(video_path, start_frame, num_frames).to(
        device
    )

    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_np = probs.cpu().numpy().flatten()

    top_idx = int(np.argmax(probs_np))
    prob_dict = {LABELS[i]: float(probs_np[i]) for i in range(len(LABELS))}

    return {
        "label": LABELS[top_idx],
        "confidence": float(probs_np[top_idx]),
        "probabilities": prob_dict,
    }


# ---------------------------------------------------------------------------
# Kinetics-400 general action model
# ---------------------------------------------------------------------------
def _load_kinetics_model(device: str | torch.device | None = None):
    """Lazy-load the Kinetics-400 VideoMAE model."""
    global _kinetics_labels
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    token = os.environ.get("HF_TOKEN")
    model = VideoMAEForVideoClassification.from_pretrained(
        KINETICS_MODEL_NAME, token=token
    )
    if _kinetics_labels is None:
        _kinetics_labels = model.config.id2label
    model.to(device).eval()
    return model, device


def classify_clip_general(
    video_path: str | Path,
    start_frame: int = 0,
    num_frames: int = _NUM_FRAMES,
    *,
    model: Optional[VideoMAEForVideoClassification] = None,
    device: Optional[str | torch.device] = None,
) -> dict:
    """Classify a video clip using the Kinetics-400 VideoMAE model.

    Same interface as classify_clip() but uses the general-purpose
    Kinetics-400 model (400 everyday action classes) instead of
    UCF-Crime's 14 anomaly categories.

    Returns
    -------
    dict with keys:
        label          - name of the top-1 predicted class
        confidence     - probability of the top-1 class (0-1)
        probabilities  - dict mapping every class name to its probability
    """
    if model is None:
        model, device = _load_kinetics_model(device)

    video_tensor = _extract_frames(video_path, start_frame, num_frames).to(
        device
    )

    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        probs_np = probs.cpu().numpy().flatten()

    top_idx = int(np.argmax(probs_np))
    prob_dict = {_kinetics_labels[i]: float(probs_np[i]) for i in range(len(probs_np))}

    return {
        "label": _kinetics_labels[top_idx],
        "confidence": float(probs_np[top_idx]),
        "probabilities": prob_dict,
    }


# ---------------------------------------------------------------------------
# CLI: run on a single clip and print the full distribution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_clip = Path(__file__).resolve().parent / "test_clips" / "positive_sample.mp4"
    if len(sys.argv) > 1:
        test_clip = Path(sys.argv[1])

    if not test_clip.exists():
        print(f"ERROR: clip not found: {test_clip}")
        sys.exit(1)

    print(f"Loading model: {MODEL_NAME} ...")
    mdl, dev = _load_model()
    print(f"Device: {dev}")
    print(f"Classifying: {test_clip.name}\n")

    result = classify_clip(test_clip, model=mdl, device=dev)

    # Sort by probability descending for readability
    ranked = sorted(result["probabilities"].items(), key=lambda x: -x[1])
    print(f"{'Class':<20s}  {'Prob':>7s}")
    print("-" * 30)
    for name, p in ranked:
        bar = "#" * int(p * 40)
        marker = " <-- top-1" if name == result["label"] else ""
        print(f"{name:<20s}  {p:7.4f}  {bar}{marker}")

    print(f"\nTop-1: {result['label']} ({result['confidence']:.4f})")
