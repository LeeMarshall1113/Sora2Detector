import cv2
import numpy as np
import os

def detect_sora_watermark(video_path, watermark_path="Sora2Watermark.png",
                          threshold=0.9, frame_stride=10, debug=False):
    """
    Detects if the Sora watermark appears in an H.264 video.

    Args:
        video_path (str): path to the .mp4 video.
        watermark_path (str): path to the watermark PNG (RGBA supported).
        threshold (float): correlation threshold (0–1).
        frame_stride (int): analyze every Nth frame for speed.
        debug (bool): print match scores.

    Returns:
        dict: {found: bool, matches: [frame info]}
    """
    # Load watermark (RGBA if present)
    templ = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    if templ is None:
        raise FileNotFoundError(f"Watermark not found at {watermark_path}")

    if templ.shape[2] == 4:
        bgr = templ[:, :, :3]
        alpha = templ[:, :, 3]
        mask = (alpha > 10).astype(np.uint8) * 255
    else:
        bgr = templ
        mask = np.ones(bgr.shape[:2], np.uint8) * 255

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    found_hits = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        res = cv2.matchTemplate(frame, bgr, cv2.TM_CCORR_NORMED, mask=mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if debug:
            print(f"Frame {frame_idx}: match={max_val:.4f}")

        if max_val >= threshold:
            time_s = frame_idx / fps
            found_hits.append({
                "frame": frame_idx,
                "time_seconds": round(time_s, 2),
                "confidence": round(float(max_val), 4),
                "position": max_loc
            })

        frame_idx += 1

    cap.release()

    result = {"found": len(found_hits) > 0, "matches": found_hits}
    return result


if __name__ == "__main__":
    video_file = r"C:\Users\Lee\Downloads\Reg-No-Watermark.mp4"  # <-- change this
    result = detect_sora_watermark(video_file, threshold=0.9, frame_stride=5, debug=True)

    if result["found"]:
        print(f"\n✅ Sora watermark FOUND in {len(result['matches'])} frame(s).")
        for m in result["matches"][:10]:
            print(f"  Frame {m['frame']} | t={m['time_seconds']}s | conf={m['confidence']} | pos={m['position']}")
    else:
        print("\n❌ No Sora watermark detected.")
