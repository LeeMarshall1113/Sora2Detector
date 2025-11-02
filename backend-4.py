#!/usr/bin/env python3
"""
backend-4.py — visual AI/counterfactual detector (banded decision)
------------------------------------------------------------------
Decision rule (default):
  prob <= 0.45 -> Likely Real
  prob >= 0.55 -> AI / Counterfactual
  otherwise    -> Unknown / Uncertain

Outputs (quiet mode): "Chance of being AI: XX%"
Exit codes:
  1 = AI (>= upper)
  0 = Not AI / Unknown (<= upper)
  2 = Error
"""

import os, json, argparse, tempfile, subprocess, sys, traceback
from pathlib import Path

import cv2
import numpy as np
from joblib import load
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops

# ====== Debug toggle ======
DEBUG = True  # set False for production-clean output

# ---------------- Paths ----------------
MODEL_PATH = Path("models/visual_rf.joblib")
META_PATH  = Path("models/visual_rf.meta.json")
if not MODEL_PATH.exists():
    raise FileNotFoundError("Missing models/visual_rf.joblib. Train it first.")
model = load(MODEL_PATH)
meta  = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}

# ---------------- Config pulled from meta ----------------
SCALES       = meta.get("scales", [320, 640])
SECONDS_BASE = meta.get("seconds_grid", [0.0, 0.8, 1.6, 2.4, 3.2])
FEATURE_KEYS = meta.get("features", [])

# for diagnostics only (does not affect verdict now)
CONSISTENCY_STD_THRESHOLD = 0.15

# ---------------- Normalization ----------------
def normalize_video(input_path: str, ffmpeg_exe: str | None = None) -> str:
    inp = Path(input_path)
    tmp_dir = Path(tempfile.gettempdir())
    out = tmp_dir / f"norm_{inp.stem}.mp4"

    exe = ffmpeg_exe or os.environ.get("FFMPEG_EXE") or "ffmpeg"
    cmd = [
        exe, "-y", "-i", str(inp),
        "-vf", "scale=640:-2,fps=24",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        str(out)
    ]
    # If debugging, allow stderr to surface; otherwise keep quiet
    try:
        if DEBUG:
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        raise RuntimeError(f"Normalization failed (is ffmpeg available?): {e}")

    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError("Normalization failed: output not created.")
    return str(out)

# ---------------- Features (match training) ----------------
def to_rgb_resized(bgr, width):
    h,w = bgr.shape[:2]
    s = width/float(w)
    out = cv2.resize(bgr,(width,int(h*s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def feat_freq(img):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    f=np.fft.fftshift(np.fft.fft2(g)); mag=np.abs(f)
    h1,h2=int(mag.shape[0]*0.25),int(mag.shape[0]*0.75)
    w1,w2=int(mag.shape[1]*0.25),int(mag.shape[1]*0.75)
    high=mag[h1:h2,w1:w2]
    return float(np.mean(high)/(np.mean(mag)+1e-8))

def feat_noise(img):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    n=g-cv2.GaussianBlur(g,(5,5),1)
    return float(np.var(n))

def feat_color_entropy(img):
    ents=[]
    for c in cv2.split(img):
        hist=cv2.calcHist([c],[0],None,[256],[0,256]).ravel()
        p=hist/(np.sum(hist)+1e-8)
        ents.append(-np.sum(p*np.log2(p+1e-8)))
    return float(np.mean(ents))

def feat_edges(img):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(gx*gx+gy*gy)
    return float(np.mean(mag))

def feat_laplacian_var(img):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g,cv2.CV_64F).var())

def feat_texture_contrast(img):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    g8=(g/8).astype(np.uint8)
    m=graycomatrix(g8,[1],[0],256,symmetric=True,normed=True)
    return float(graycoprops(m,'contrast')[0,0])

def feat_temporal(prev,curr):
    h=min(prev.shape[0],curr.shape[0]); w=min(prev.shape[1],curr.shape[1])
    A=cv2.resize(prev,(w,h)); B=cv2.resize(curr,(w,h))
    s_val=ssim(cv2.cvtColor(A,cv2.COLOR_RGB2GRAY),
               cv2.cvtColor(B,cv2.COLOR_RGB2GRAY),data_range=255)
    return float(1.0-s_val)

def sample_frames_timebased(path, seconds):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): return []
    fps=cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames=[]
    for t in seconds:
        idx=int(round(t*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok,f=cap.read()
        if ok:
            f=cv2.resize(f,(640,int(f.shape[0]*640/f.shape[1])),interpolation=cv2.INTER_AREA)
            frames.append(f)
    cap.release()
    return frames

def per_video_features_multiscale(frames_bgr, scales=SCALES):
    feats={}
    for s in scales:
        rgb=[to_rgb_resized(f,s) for f in frames_bgr]
        F={"freq":[], "noise":[], "color":[], "edge":[], "lapvar":[], "temporal":[0.0], "texture":[]}
        for i,fr in enumerate(rgb):
            F["freq"].append(feat_freq(fr))
            F["noise"].append(feat_noise(fr))
            F["color"].append(feat_color_entropy(fr))
            F["edge"].append(feat_edges(fr))
            F["lapvar"].append(feat_laplacian_var(fr))
            F["texture"].append(feat_texture_contrast(fr))
            if i>0:
                F["temporal"].append(feat_temporal(rgb[i-1], fr))
        for k,arr in F.items():
            arr = arr or [0.0]
            feats[f"{k}_mean_{s}"]=float(np.mean(arr))
            feats[f"{k}_std_{s}"] =float(np.std(arr))
    return feats

# ---------------- Core analysis on a normalized file ----------------
def _analyze_normalized(norm_path: str, lower_band: float, upper_band: float):
    p = Path(norm_path)

    # duration → define segments
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = total_frames / fps if fps else 0
    cap.release()

    classes = getattr(model, "classes_", np.array([0,1]))
    ai_col = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else 1
    feat_cols = FEATURE_KEYS

    def prob_ai_for_seconds(seconds):
        frames = sample_frames_timebased(p, seconds)
        if not frames:
            return None, 0.0
        feats = per_video_features_multiscale(frames)
        cols = feat_cols or list(feats.keys())
        vec  = [feats.get(c,0.0) for c in cols]
        coverage = (sum(1 for v in vec if v!=0.0) / max(1,len(vec)))
        probs = model.predict_proba(np.array([vec], dtype=np.float32))[0]
        return float(probs[ai_col]), float(coverage)

    segs = []
    segs.append(SECONDS_BASE)  # early
    mid_off  = min(max(2.0, dur*0.50), max(0.0, dur-3.0))
    late_off = min(max(2.0, dur*0.75), max(0.0, dur-3.0))
    segs.append([t+mid_off  for t in SECONDS_BASE if t+mid_off  < dur])
    segs.append([t+late_off for t in SECONDS_BASE if t+late_off < dur])

    segment_probs, coverages = [], []
    for seconds in segs:
        if not seconds: continue
        pa, cov = prob_ai_for_seconds(seconds)
        if pa is not None:
            segment_probs.append(pa)
            coverages.append(cov)

    if not segment_probs:
        return {"error": "no_frames"}

    prob_med = float(np.median(segment_probs))

    # simple per-frame consistency (diagnostic)
    frames_main = sample_frames_timebased(p, SECONDS_BASE)
    frame_probs = []
    if frames_main:
        cols = feat_cols
        for f in frames_main[:min(len(frames_main), 5)]:
            small_feats = per_video_features_multiscale([f])
            cols_eff = cols or list(small_feats.keys())
            small_x = np.array([[small_feats.get(c,0.0) for c in cols_eff]], dtype=np.float32)
            probs_f = model.predict_proba(small_x)[0]
            frame_probs.append(float(probs_f[ai_col]))
    consistency_std = float(np.std(frame_probs)) if frame_probs else 0.0

    # --------- Banded decision rule ---------
    if prob_med >= upper_band:
        verdict = "AI / Counterfactual"
    elif prob_med <= lower_band:
        verdict = "Likely Real"
    else:
        verdict = "Unknown / Uncertain"
    # ---------------------------------------

    out = {
        "file": str(Path(norm_path)),
        "probability": float(prob_med),
        "lower_band": float(lower_band),
        "upper_band": float(upper_band),
        "counterfactual_percent": round(float(prob_med)*100, 2),
        "consistency_std": round(consistency_std, 4),
        "segment_probs": [round(x,4) for x in segment_probs],
        "verdict": verdict
    }

    if coverages:
        cov_med = float(np.median(coverages))
        if cov_med < 0.20:
            out["warn_feature_coverage"] = round(cov_med, 3)

    return out

# ---------------- Public entry: normalize → analyze → cleanup ----------------
def analyze_video(video_path: str, lower_band: float, upper_band: float, ffmpeg_exe: str | None = None):
    normalized = normalize_video(video_path, ffmpeg_exe=ffmpeg_exe)
    try:
        result = _analyze_normalized(normalized, lower_band, upper_band)
    finally:
        try:
            Path(normalized).unlink(missing_ok=True)
        except Exception:
            pass
    # Report the original file path, not the temp
    if isinstance(result, dict) and "file" in result:
        result["file"] = str(Path(video_path))
    return result

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to .mp4")
    ap.add_argument("--lower", type=float, default=0.45, help="Real if prob <= lower (default 0.45)")
    ap.add_argument("--upper", type=float, default=0.55, help="AI if prob >= upper (default 0.55)")
    ap.add_argument("--ffmpeg-exe", help="Path to ffmpeg.exe (overrides PATH / FFMPEG_EXE)")
    ap.add_argument("--debug-json", action="store_true", help="When DEBUG, print full JSON diagnostics")
    args = ap.parse_args()

    # safety: ensure lower < upper
    lower = min(args.lower, args.upper - 1e-6)
    upper = max(args.upper, args.lower + 1e-6)

    try:
        res = analyze_video(args.video, lower, upper, ffmpeg_exe=args.ffmpeg_exe)
        if not isinstance(res, dict):
            raise RuntimeError("Unexpected result type from analyze_video")
    except Exception as e:
        if DEBUG:
            print("❌ Exception occurred:", e)
            traceback.print_exc()
        else:
            print("Chance of being AI: N/A%")
        sys.exit(2)

    # percent & exit code
    percent = None
    if "counterfactual_percent" in res:
        percent = res["counterfactual_percent"]
    elif "probability" in res:
        percent = round(float(res["probability"]) * 100, 2)

    # Quiet print
    if isinstance(percent, (int, float)):
        print(f"Chance of being AI: {int(percent)}%")
    else:
        print("Chance of being AI: N/A%")

    # Optional diagnostics
    if DEBUG and args.debug_json:
        try:
            print(json.dumps(res, indent=2))
        except Exception:
            pass

    # Exit codes: align with backend-3 semantics (1=AI, 0=otherwise)
    prob = float(res.get("probability", -1.0)) if isinstance(res, dict) else -1.0
    if prob >= upper:
        sys.exit(1)  # AI
    else:
        sys.exit(0)  # Not AI / Unknown

if __name__ == "__main__":
    main()
