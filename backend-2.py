import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional
import cv2
import numpy as np

# OCR: EasyOCR is simple to install (no Tesseract path hassles)
try:
    import easyocr
except Exception as e:
    raise SystemExit("easyocr is required. Install with: pip install easyocr") from e


@dataclass
class StepHit:
    step_name: str
    frame: int
    time_s: float
    text: str
    conf: float
    box: Tuple[int, int, int, int]  # x, y, w, h (OCR ROI bounds for reference)


def get_rois(frame_wh: Tuple[int, int]) -> dict:
    """Define TL / MR / BL ROIs as fractions of the frame."""
    W, H = frame_wh
    return {
        "TL": (0, 0, int(0.48 * W), int(0.38 * H)),                              # top-left
        "MR": (int(0.52 * W), int(0.30 * H), int(0.46 * W), int(0.40 * H)),     # middle-right
        "BL": (0, int(0.62 * H), int(0.48 * W), int(0.38 * H)),                 # bottom-left
    }


def text_matches_sora(reader, frame_bgr, roi_xywh, min_conf=0.5) -> Tuple[bool, str, float]:
    """
    Run OCR in the ROI and check for the literal word 'sora' (case-insensitive).
    Returns (hit, matched_text, confidence).
    """
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

    best_text, best_conf = "", 0.0
    for (bbox, text, conf) in results:
        norm = "".join(ch for ch in text.lower() if ch.isalnum())
        if norm == "sora" and conf >= min_conf:
            return True, text, float(conf)
        if conf > best_conf:
            best_text, best_conf = text, float(conf)

    return False, best_text, best_conf


def required_steps_for_duration(seconds: float) -> int:
    """
    Map duration to min steps:
      < 2s -> 1
      < 4s -> 2
      >= 5s -> 3
      [4,5) -> 2 (by your spec)
    """
    if seconds < 2.0:
        return 1
    if seconds < 4.0:
        return 2
    if seconds >= 5.0:
        return 3
    # 4.0 <= seconds < 5.0
    return 2


def detect_sequence(
    video_path: str,
    need_consec: int = 2,
    stride: int = 3,
    min_conf: float = 0.55,
    gpu: bool = False,
    min_steps: Optional[int] = None  # if None, compute from duration
) -> Tuple[bool, List[StepHit], float, int]:
    """
    Detect ordered appearance of the literal word 'Sora' in:
    TL -> MR -> BL. If min_steps is None, it is computed from video duration.
    Returns (accepted, hits, duration_seconds, required_steps_used).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration_s = frames / fps if frames > 0 else 0.0

    # Auto-determine required steps if not provided
    required_steps = min_steps if min_steps is not None else required_steps_for_duration(duration_s)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rois = get_rois((W, H))

    reader = easyocr.Reader(['en'], gpu=gpu)

    ordered_steps = ["TL", "MR", "BL"]
    step_idx = 0
    consec = 0
    hits: List[StepHit] = []

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        current_step = ordered_steps[step_idx]
        roi = rois[current_step]

        found, txt, conf = text_matches_sora(reader, frame, roi, min_conf=min_conf)

        if found:
            consec += 1
        else:
            consec = 0

        if consec >= need_consec:
            hits.append(
                StepHit(
                    step_name=current_step,
                    frame=frame_idx,
                    time_s=frame_idx / fps,
                    text="Sora",
                    conf=conf,
                    box=roi,
                )
            )
            consec = 0
            if step_idx < len(ordered_steps) - 1:
                step_idx += 1
            else:
                # full sequence done
                break

        frame_idx += 1

    cap.release()

    accept = (len(hits) >= required_steps)
    return accept, hits, duration_s, required_steps


def main():
    ap = argparse.ArgumentParser(
        description="Detect literal word 'Sora' in sequence: Top-Left → Middle-Right → Bottom-Left with auto step requirement by video length."
    )
    ap.add_argument("video", help="Path to .mp4/.mov (H.264 is fine)")
    ap.add_argument("--need-consec", type=int, default=2,
                    help="Consecutive sampled frames required to confirm each step (default 2)")
    ap.add_argument("--stride", type=int, default=3,
                    help="Process every Nth frame for speed (default 3)")
    ap.add_argument("--min-conf", type=float, default=0.55,
                    help="Minimum OCR confidence for accepting 'Sora' (default 0.55)")
    ap.add_argument("--gpu", action="store_true",
                    help="Use GPU for EasyOCR if available")
    ap.add_argument("--min-steps", type=int, default=None,
                    help="Override required steps (1..3). If omitted, auto-set from duration rules.")
    args = ap.parse_args()

    accepted, hits, dur_s, req_steps = detect_sequence(
        args.video,
        need_consec=args.need_consec,
        stride=args.stride,
        min_conf=args.min_conf,
        gpu=args.gpu,
        min_steps=args.min_steps  # None => auto
    )

    print(f"\nVideo length: {dur_s:.2f}s → required steps: {req_steps}")
    if accepted:
        if len(hits) == 3:
            print("✅ FOUND: Completed full sequence TL → MR → BL.")
        else:
            print(f"🟡 LIKELY: Completed {len(hits)}/3 steps in order "
                  f"(required ≥{req_steps}).")
        for h in hits:
            print(f"  {h.step_name}: t={h.time_s:.2f}s frame={h.frame} conf={h.conf:.2f} roi={h.box}")
    else:
        print(f"❌ NOT CONFIRMED: Sequence incomplete (got {len(hits)}/3, needed ≥{req_steps}).")
        if hits:
            for h in hits:
                print(f"  Partial: {h.step_name} at t={h.time_s:.2f}s (conf={h.conf:.2f})")


if __name__ == "__main__":
    main()
