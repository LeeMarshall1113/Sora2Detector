#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional
import os
import sys
import cv2
import numpy as np
import warnings

# Suppress OpenCV, Torch, and EasyOCR logs/warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_WARNINGS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
sys.stderr = open(os.devnull, "w")  # silence stderr globally (OCR/tensorflow/torch logs)

try:
    import easyocr
except Exception as e:
    sys.stderr = sys.__stderr__
    raise SystemExit("easyocr is required. Install with: pip install easyocr") from e


@dataclass
class StepHit:
    step_name: str
    frame: int
    time_s: float
    text: str
    conf: float
    box: Tuple[int, int, int, int]


def get_rois(frame_wh: Tuple[int, int]) -> dict:
    W, H = frame_wh
    return {
        "TL": (0, 0, int(0.48 * W), int(0.38 * H)),
        "MR": (int(0.52 * W), int(0.30 * H), int(0.46 * W), int(0.40 * H)),
        "BL": (0, int(0.62 * H), int(0.38 * W)),
    }


def text_matches_sora(reader, frame_bgr, roi_xywh, min_conf=0.5) -> Tuple[bool, str, float]:
    x, y, w, h = roi_xywh
    roi = frame_bgr[y:y + h, x:x + w]
    if roi.size == 0:
        return False, "", 0.0

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    results = reader.readtext(
        roi_rgb,
        detail=1,
        paragraph=False,
        text_threshold=0.4,
        low_text=0.3,
        link_threshold=0.4
    )

    for (_, text, conf) in results:
        if "".join(ch for ch in text.lower() if ch.isalnum()) == "sora" and conf >= min_conf:
            return True, text, float(conf)
    return False, "", 0.0


def required_steps_for_duration(seconds: float) -> int:
    if seconds < 2.0:
        return 1
    if seconds < 4.0:
        return 2
    if seconds >= 5.0:
        return 3
    return 2


def detect_sequence(video_path: str, need_consec=2, stride=3, min_conf=0.55, gpu=False, min_steps=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_s = frames / fps if frames > 0 else 0.0
    required_steps = min_steps if min_steps is not None else required_steps_for_duration(duration_s)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rois = get_rois((W, H))

    reader = easyocr.Reader(['en'], gpu=gpu, verbose=False)
    ordered_steps = ["TL", "MR", "BL"]
    step_idx, consec = 0, 0
    hits: List[StepHit] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        roi = rois[ordered_steps[step_idx]]
        found, _, conf = text_matches_sora(reader, frame, roi, min_conf)
        consec = consec + 1 if found else 0

        if consec >= need_consec:
            hits.append(StepHit(ordered_steps[step_idx], frame_idx, frame_idx / fps, "Sora", conf, roi))
            consec = 0
            step_idx = min(step_idx + 1, len(ordered_steps) - 1)
            if step_idx == len(ordered_steps) - 1 and len(hits) >= required_steps:
                break
        frame_idx += 1

    cap.release()
    return len(hits) >= required_steps


def main():
    ap = argparse.ArgumentParser(description="Silent detector — prints only AI or Not AI.")
    ap.add_argument("video", help="Path to the video file")
    ap.add_argument("--need-consec", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--min-steps", type=int, default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        sys.stderr = sys.__stderr__
        print(f"Error: {args.video} not found.", file=sys.stderr)
        sys.exit(2)

    try:
        is_ai = detect_sequence(
            args.video,
            need_consec=args.need_consec,
            stride=args.stride,
            min_conf=args.min_conf,
            gpu=args.gpu,
            min_steps=args.min_steps
        )
    except Exception:
        sys.stderr = sys.__stderr__
        sys.exit(2)

    sys.stderr = sys.__stderr__
    print("AI" if is_ai else "Not AI")
    sys.exit(1 if is_ai else 0)


if __name__ == "__main__":
    main()
