# trainer.py
import os, json, argparse, joblib, warnings
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
import soundfile as sf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(y, sr):
    """
    Returns a 1D feature vector for one audio array.
    Uses MFCCs + spectral stats + waveform stats.
    """
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # ensure at least 1s
    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)), mode="constant")

    # consistent STFT params
    n_fft = 1024
    hop_length = 256
    win_length = 512

    # |STFT|
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    S_power = (S ** 2).astype(np.float32)

    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(S=S_power, sr=sr, n_mels=64, fmin=20, fmax=min(8000, sr // 2))
    mel_db = librosa.power_to_db(mel + 1e-10)

    # MFCCs (+ deltas)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    dmfcc = librosa.feature.delta(mfcc)
    ddmfcc = librosa.feature.delta(mfcc, order=2)

    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(S=S_power, sr=sr)
    spec_bw   = librosa.feature.spectral_bandwidth(S=S_power, sr=sr)
    spec_con  = librosa.feature.spectral_contrast(S=S_power, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(S=S_power, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=hop_length)
    # RMS with frame_length matching n_fft to avoid warnings
    rms       = librosa.feature.rms(S=S_power, frame_length=n_fft)

    def stats(mat):
        """aggregate [mean, std, median, min, max, p25, p75] per row"""
        arr = np.asarray(mat)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        feats = []
        for row in arr:
            m = np.nanmean(row); s = np.nanstd(row)
            feats += [
                m, s, np.nanmedian(row), np.nanmin(row), np.nanmax(row),
                np.nanpercentile(row, 25), np.nanpercentile(row, 75)
            ]
        return np.array(feats, dtype=np.float32)

    feat_vec = np.concatenate([
        stats(mfcc), stats(dmfcc), stats(ddmfcc),
        stats(spec_cent), stats(spec_bw), stats(spec_con),
        stats(rolloff), stats(zcr), stats(rms),
        stats(mel_db),
    ], axis=0)

    # waveform stats
    from scipy.stats import skew, kurtosis
    y_f32 = np.asarray(y, dtype=np.float32)
    wav_stats = np.array([
        float(np.mean(y_f32)), float(np.std(y_f32)), float(np.median(y_f32)),
        float(np.min(y_f32)), float(np.max(y_f32)),
        float(skew(y_f32)), float(kurtosis(y_f32)),
        float(np.percentile(y_f32, 1)), float(np.percentile(y_f32, 99))
    ], dtype=np.float32)

    out = np.concatenate([feat_vec, wav_stats], axis=0)
    # sanitize
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out

def load_audio_any(path, target_sr=16000):
    """Loads audio via librosa; mono float32 at target_sr."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr

# -----------------------------
# Dataset scanning
# -----------------------------
def scan_dataset(ai_dir, real_dir, sr=16000, win_sec=6.0, stride_sec=6.0):
    """
    Slice each file into windows and extract features per-window.
    """
    X, y, files = [], [], []
    win = int(sr * win_sec)
    hop = int(sr * stride_sec)

    def iter_paths(d):
        p = Path(d)
        exts = [".wav", ".flac", ".ogg", ".mp3", ".m4a"]
        paths = []
        for e in exts:
            paths += list(p.glob(f"*{e}"))
        return sorted(paths)

    def process_dir(d, label):
        for p in tqdm(iter_paths(d), desc=f"Features from {d}"):
            try:
                sig, _ = load_audio_any(str(p), target_sr=sr)
                if len(sig) < win:
                    seg = np.pad(sig, (0, win - len(sig)))
                    feats = extract_features(seg, sr)
                    X.append(feats); y.append(label); files.append(str(p))
                else:
                    for start in range(0, len(sig) - win + 1, hop):
                        seg = sig[start:start + win]
                        feats = extract_features(seg, sr)
                        X.append(feats); y.append(label); files.append(str(p))
            except Exception as e:
                print(f"[WARN] {p}: {e}")

    process_dir(real_dir, 0)
    process_dir(ai_dir,   1)

    X = np.stack(X, axis=0)
    # ensure numeric, finite, contiguous, writeable
    X = np.asarray(X, dtype=np.float32, order="C")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.ascontiguousarray(X).copy()

    y = np.array(y, dtype=np.int32)
    files = np.array(files)
    return X, y, files

# -----------------------------
# Training
# -----------------------------
def train_model(X, y, random_state=42):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # enforce contiguous, writeable
    Xs = np.ascontiguousarray(Xs, dtype=np.float32).copy()
    y  = np.ascontiguousarray(y,  dtype=np.int32).copy()

    classes = np.array([0, 1], dtype=np.int32)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, class_weights)}

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, class_weight=cw, random_state=random_state
    )
    gb = GradientBoostingClassifier(random_state=random_state)
    lr = LogisticRegression(max_iter=2000, class_weight=cw, solver="lbfgs")

    # single-process on Windows avoids loky read-only view edge-case
    clf = StackingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        final_estimator=lr,
        passthrough=True,
        n_jobs=1
    )
    clf.fit(Xs, y)
    return scaler, clf

# -----------------------------
# Quick eval (window-level holdout)
# -----------------------------
def quick_eval(X, y):
    rng = np.random.default_rng(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    cut = int(0.8 * len(idx))
    tr, te = idx[:cut], idx[cut:]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[tr])
    Xte = scaler.transform(X[te])

    # contiguous
    Xtr = np.ascontiguousarray(Xtr, dtype=np.float32).copy()
    Xte = np.ascontiguousarray(Xte, dtype=np.float32).copy()
    ytr = np.ascontiguousarray(y[tr], dtype=np.int32).copy()
    yte = np.ascontiguousarray(y[te], dtype=np.int32).copy()

    classes = np.array([0, 1], dtype=np.int32)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=ytr)
    cw = {int(c): float(w) for c, w in zip(classes, class_weights)}

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        n_jobs=-1, class_weight=cw, random_state=42
    )
    gb = GradientBoostingClassifier(random_state=42)
    lr = LogisticRegression(max_iter=2000, class_weight=cw, solver="lbfgs")
    clf = StackingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        final_estimator=lr,
        passthrough=True,
        n_jobs=1
    )
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    auc = roc_auc_score(yte, prob)
    f1  = f1_score(yte, pred)
    print(f"[Quick Eval] AUC={auc:.4f}  F1={f1:.4f}")
    print(classification_report(yte, pred, digits=3))

@dataclass
class BundleMeta:
    sr: int = 16000
    win_sec: float = 6.0
    stride_sec: float = 3.0
    feature_version: str = "v1_basic_mfcc_spec"
    label_map: dict = None

def save_bundle(out_dir, scaler, clf, meta: BundleMeta):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out / "scaler.joblib")
    joblib.dump(clf,    out / "model.joblib")
    with open(out / "meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2)
    print(f"[OK] Saved bundle to: {out.resolve()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-dir", required=True, help="Folder with AI (synthetic) WAVs")
    ap.add_argument("--real-dir", required=True, help="Folder with real/human WAVs")
    ap.add_argument("--out", default="aiaudio_bundle", help="Output directory for model bundle")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--win-sec", type=float, default=6.0)
    ap.add_argument("--stride-sec", type=float, default=6.0, help="training window stride")
    args = ap.parse_args()

    X, y, files = scan_dataset(args.ai_dir, args.real_dir, sr=args.sr,
                               win_sec=args.win_sec, stride_sec=args.stride_sec)
    print(f"Windows: {len(y)} | Pos(AI)={(y==1).sum()} Neg(Real)={(y==0).sum()} | Feats={X.shape[1]}")

    quick_eval(X, y)  # evaluate on a window-level holdout set

    scaler, clf = train_model(X, y)
    meta = BundleMeta(sr=args.sr, win_sec=args.win_sec, stride_sec=3.0,
                      label_map={"0": "real", "1": "ai"})
    save_bundle(args.out, scaler, clf, meta)

if __name__ == "__main__":
    main()
