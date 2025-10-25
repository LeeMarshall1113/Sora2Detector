# detect_audio.py
import argparse, json, io, subprocess
from pathlib import Path

import numpy as np
import joblib
import soundfile as sf
import librosa

# reuse the exact same feature function to stay consistent
from trainer import extract_features

def read_any(media_path, target_sr=16000):
    """
    Read audio from common audio files directly, and from video (e.g., MP4/MOV/MKV)
    by decoding the audio track to WAV bytes via ffmpeg -> stdout (no temp file).
    """
    p = Path(media_path)
    ext = p.suffix.lower()

    if ext in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
        y, sr = librosa.load(str(p), sr=target_sr, mono=True)
        return y.astype(np.float32), sr

    # assume video container -> pipe to WAV
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(p),
        "-f", "wav", "-acodec", "pcm_s16le",
        "-ar", str(target_sr), "-ac", "1",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wav_bytes, err = proc.communicate()
    if proc.returncode != 0 or len(wav_bytes) == 0:
        raise RuntimeError(f"ffmpeg failed to decode audio: {err.decode('utf-8', errors='ignore')}")
    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr

def window_indices(n, win, hop):
    if n < win:
        yield 0, n
    else:
        for s in range(0, n - win + 1, hop):
            yield s, s + win

def predict_clip(bundle_dir, media_path, agg="mean", threshold=0.5):
    bundle = Path(bundle_dir)
    scaler = joblib.load(bundle / "scaler.joblib")
    clf    = joblib.load(bundle / "model.joblib")
    meta   = json.load(open(bundle / "meta.json", "r"))

    sr       = int(meta.get("sr", 16000))
    win_sec  = float(meta.get("win_sec", 6.0))
    hop_sec  = float(meta.get("stride_sec", 3.0))
    labelmap = meta.get("label_map", {"0": "real", "1": "ai"})

    y, _ = read_any(media_path, target_sr=sr)
    win = int(sr * win_sec)
    hop = int(sr * hop_sec)

    feats = []
    for s, e in window_indices(len(y), win, hop):
        seg = y[s:e]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        f = extract_features(seg, sr)
        feats.append(f)

    X = np.stack(feats, axis=0)
    X = np.asarray(X, dtype=np.float32, order="C")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xs = scaler.transform(X)
    probs = clf.predict_proba(Xs)[:, 1]  # probability of AI (class 1)

    if agg == "mean":
        clip_prob = float(np.mean(probs))
    elif agg == "max":
        clip_prob = float(np.max(probs))
    else:
        votes = (probs >= threshold).astype(int)
        clip_prob = float(np.mean(votes))

    is_ai = clip_prob >= threshold
    return {
        "file": str(media_path),
        "prob_ai": clip_prob,
        "label": labelmap["1"] if is_ai else labelmap["0"],
        "segment_probs": probs.tolist(),
        "win_sec": win_sec,
        "hop_sec": hop_sec
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media", help="Path to MP4/MP3/WAV/etc.")
    ap.add_argument("--bundle", default="aiaudio_bundle", help="Folder with model.joblib/scaler.joblib/meta.json")
    ap.add_argument("--agg", choices=["mean", "max", "vote"], default="mean")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    res = predict_clip(args.bundle, args.media, agg=args.agg, threshold=args.threshold)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
