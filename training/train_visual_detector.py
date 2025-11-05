"""
train_visual_rf.py
-------------------------------------------------------
Train the visual AI-video detector to match backend-4.py.

Inputs:
  --real DIR   : folder of normalized REAL clips (SDR, 640-wide, 24fps)
  --ai   DIR   : folder of normalized AI clips   (same profile)
  (Optionally multiple --real and --ai can be passed)

What it does:
  • Extracts features identical to backend-4.py (320/640, early/mid/late segments)
  • Robustly fuses segment scores (median)
  • Trains RandomForest (class_weight balanced)
  • Optional probability calibration (isotonic) on validation set
  • Computes metrics on held-out test set
  • Picks threshold via Youden J on validation set
  • Saves model and meta usable by backend-4.py

Outputs:
  models/visual_rf.joblib
  models/visual_rf.meta.json
"""

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from joblib import dump
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.feature import graycomatrix, graycoprops

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, f1_score, accuracy_score)
from sklearn.calibration import CalibratedClassifierCV

# ---------------- CONFIG (MUST MATCH INFERENCE) ----------------
SCALES = [320, 640]
SECONDS_BASE = [0.0, 0.8, 1.6, 2.4, 3.2]  # per segment
N_SEGMENTS = 3   # early/middle/late
MAX_FRAMES_PER_SEG = 5
RANDOM_STATE = 1337

# RandomForest defaults (tuned for stability)
RF_PARAMS = dict(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# Where to save artifacts
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "visual_rf.joblib"
META_PATH  = MODEL_DIR / "visual_rf.meta.json"


# ---------------- FEATURE PIPELINE (MATCHES backend-4.py) ----------------
def to_rgb_resized(bgr, width):
    h, w = bgr.shape[:2]
    s = width / float(w)
    out = cv2.resize(bgr, (width, int(h*s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def feat_freq(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    h1, h2 = int(mag.shape[0]*0.25), int(mag.shape[0]*0.75)
    w1, w2 = int(mag.shape[1]*0.25), int(mag.shape[1]*0.75)
    high = mag[h1:h2, w1:w2]
    return float(np.mean(high) / (np.mean(mag)+1e-8))

def feat_noise(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    noise = g - cv2.GaussianBlur(g,(5,5),1)
    return float(np.var(noise))

def feat_color_entropy(img):
    ents=[]
    for c in cv2.split(img):
        hist = cv2.calcHist([c],[0],None,[256],[0,256]).ravel()
        p = hist/(np.sum(hist)+1e-8)
        ents.append(-np.sum(p*np.log2(p+1e-8)))
    return float(np.mean(ents))

def feat_edges(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F,1,0,ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(gx*gx+gy*gy)
    return float(np.mean(mag))

def feat_laplacian_var(img):
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g,cv2.CV_64F).var())

def feat_texture_contrast(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g8 = (g / 8).astype(np.uint8)  # reduce bins → stabler GLCM
    m = graycomatrix(g8, [1], [0], 256, symmetric=True, normed=True)
    return float(graycoprops(m, 'contrast')[0,0])

def feat_temporal(prev,curr):
    h=min(prev.shape[0],curr.shape[0]); w=min(prev.shape[1],curr.shape[1])
    A=cv2.resize(prev,(w,h)); B=cv2.resize(curr,(w,h))
    s_val=ssim(cv2.cvtColor(A,cv2.COLOR_RGB2GRAY),
               cv2.cvtColor(B,cv2.COLOR_RGB2GRAY),data_range=255)
    return float(1.0-s_val)

def sample_frames_timebased(path: Path, seconds: List[float]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames=[]
    for t in seconds:
        idx=int(round(t*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            # enforce 640p width (you normalized to 640 anyway; this keeps exact match)
            f = cv2.resize(f, (640, int(f.shape[0]*640/f.shape[1])), interpolation=cv2.INTER_AREA)
            frames.append(f)
    cap.release()
    return frames

def per_video_features_multiscale(frames_bgr: List[np.ndarray], scales: List[int]) -> Dict[str,float]:
    feats={}
    for s in scales:
        rgb_frames=[to_rgb_resized(f,s) for f in frames_bgr]
        F={"freq":[], "noise":[], "color":[], "edge":[], "lapvar":[], "temporal":[0.0], "texture":[]}
        for i,fr in enumerate(rgb_frames):
            F["freq"].append(feat_freq(fr))
            F["noise"].append(feat_noise(fr))
            F["color"].append(feat_color_entropy(fr))
            F["edge"].append(feat_edges(fr))
            F["lapvar"].append(feat_laplacian_var(fr))
            F["texture"].append(feat_texture_contrast(fr))
            if i>0:
                F["temporal"].append(feat_temporal(rgb_frames[i-1], fr))
        for k,arr in F.items():
            arr = arr or [0.0]
            feats[f"{k}_mean_{s}"] = float(np.mean(arr))
            feats[f"{k}_std_{s}"]  = float(np.std(arr))
    return feats

def segments_for_duration(duration: float) -> List[List[float]]:
    """Compute early/middle/late second grids within video duration."""
    segs = []
    # early
    segs.append(SECONDS_BASE)
    # middle
    mid_off  = min(max(2.0, duration*0.50), max(0.0, duration-3.0))
    mid_grid = [t+mid_off for t in SECONDS_BASE if t+mid_off < duration]
    segs.append(mid_grid)
    # late
    late_off = min(max(2.0, duration*0.75), max(0.0, duration-3.0))
    late_grid= [t+late_off for t in SECONDS_BASE if t+late_off < duration]
    segs.append(late_grid)
    return [g for g in segs if len(g)>0]

def extract_feature_vector(video_path: Path) -> Dict[str,float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = total_frames / fps if fps else 0
    cap.release()

    seg_grids = segments_for_duration(dur)
    # per-segment feature dicts
    seg_feat_list = []
    for grid in seg_grids[:N_SEGMENTS]:
        frames = sample_frames_timebased(video_path, grid)
        if not frames:
            continue
        feats = per_video_features_multiscale(frames, SCALES)
        seg_feat_list.append(feats)

    if not seg_feat_list:
        return {}

    # Robust fusion across segments: take median per feature key
    all_keys = set().union(*[d.keys() for d in seg_feat_list])
    fused = {}
    for k in all_keys:
        vals = [d.get(k, 0.0) for d in seg_feat_list]
        fused[k] = float(np.median(vals))
    return fused


# ---------------- DATA DISCOVERY ----------------
def list_videos(dirs: List[str]) -> List[Path]:
    exts = {".mp4",".mkv",".mov",".m4v",".webm"}
    vids=[]
    for d in dirs:
        p=Path(d)
        if p.is_file() and p.suffix.lower() in exts:
            vids.append(p)
        elif p.is_dir():
            vids += [x for x in p.rglob("*") if x.suffix.lower() in exts]
    return sorted(vids)


# ---------------- TRAINING ----------------
def youden_threshold(y_true, y_prob) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i])

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train visual RandomForest for AI-video detection.")
    ap.add_argument("--real", nargs="+", required=True, help="One or more folders/files of REAL normalized SDR videos")
    ap.add_argument("--ai",   nargs="+", required=True, help="One or more folders/files of AI normalized SDR videos")
    ap.add_argument("--val-size", type=float, default=0.2, help="Validation size fraction (default 0.2)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test size fraction (default 0.2)")
    ap.add_argument("--calibrate", action="store_true", help="Apply isotonic calibration on validation set")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count()//2), help="Parallel feature workers")
    args = ap.parse_args()

    real_vids = list_videos(args.real)
    ai_vids   = list_videos(args.ai)

    print(f"[i] Found REAL: {len(real_vids)}  AI: {len(ai_vids)}")
    if len(real_vids)==0 or len(ai_vids)==0:
        raise SystemExit("Need both REAL and AI videos.")

    X = []
    y = []
    paths = []

    all_samples = [(p,0) for p in real_vids] + [(p,1) for p in ai_vids]

    # --- feature extraction in parallel ---
    print("[i] Extracting features ...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_feature_vector, p): (p,lab) for p,lab in all_samples}
        for f in tqdm(as_completed(futs), total=len(futs)):
            p,lab = futs[f]
            try:
                feats = f.result()
            except Exception:
                feats = {}
            if not feats:
                continue
            X.append(feats)
            y.append(lab)
            paths.append(str(p))

    if not X:
        raise SystemExit("No features extracted. Check input videos / codecs.")

    # align feature columns
    feat_keys = sorted(set().union(*[set(d.keys()) for d in X]))
    X_mat = np.array([[d.get(k,0.0) for k in feat_keys] for d in X], dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[i] Final feature dim: {X_mat.shape[1]}  Samples: {X_mat.shape[0]}")

    # train/val/test split (stratified)
    X_train, X_tmp, y_train, y_tmp, p_train, p_tmp = train_test_split(
        X_mat, y, paths, test_size=(args.val_size+args.test_size), random_state=RANDOM_STATE, stratify=y)

    rel_val = args.val_size / (args.val_size + args.test_size)
    X_val, X_test, y_val, y_test, p_val, p_test = train_test_split(
        X_tmp, y_tmp, p_tmp, test_size=(1.0 - rel_val), random_state=RANDOM_STATE, stratify=y_tmp)

    print(f"[i] Split → train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")

    # model
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)

    # raw val metrics
    val_prob = rf.predict_proba(X_val)[:,1]
    val_auc  = roc_auc_score(y_val, val_prob)
    thr_youden = youden_threshold(y_val, val_prob)

    print(f"\n=== VAL (uncalibrated) ===")
    print(f"ROC-AUC: {val_auc:.4f}")
    print(f"Recommended threshold (Youden J): {thr_youden:.3f}")

    # Optional calibration on validation set
    model_to_save = rf
    if args.calibrate:
        print("[i] Calibrating probabilities (isotonic) on validation set...")
        cal = CalibratedClassifierCV(rf, cv="prefit", method="isotonic")
        cal.fit(X_val, y_val)
        model_to_save = cal
        # re-evaluate threshold on calibrated probs
        val_prob_cal = model_to_save.predict_proba(X_val)[:,1]
        thr_youden = youden_threshold(y_val, val_prob_cal)
        val_auc_cal = roc_auc_score(y_val, val_prob_cal)
        print(f"ROC-AUC (calibrated): {val_auc_cal:.4f}")
        print(f"Recommended threshold (Youden J, calibrated): {thr_youden:.3f}")

    # test metrics
    test_prob = model_to_save.predict_proba(X_test)[:,1]
    test_auc  = roc_auc_score(y_test, test_prob)
    youden_thr = thr_youden

    y_pred = (test_prob >= youden_thr).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print("\n=== TEST ===")
    print(f"Accuracy: {acc:.4f}   F1: {f1:.4f}   ROC-AUC: {test_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # save model + meta for backend-4.py
    dump(model_to_save, MODEL_PATH)
    meta = {
        "features": feat_keys,
        "scales": SCALES,
        "seconds_grid": SECONDS_BASE,
        "threshold_recommendation": float(youden_thr),
        "rf_params": RF_PARAMS,
        "calibrated": bool(args.calibrate),
        "stats": {
            "val_auc": float(val_auc),
            "test_auc": float(test_auc),
            "test_accuracy": float(acc),
            "test_f1": float(f1)
        }
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"\n[✓] Saved:\n  {MODEL_PATH}\n  {META_PATH}")


if __name__ == "__main__":
    main()
