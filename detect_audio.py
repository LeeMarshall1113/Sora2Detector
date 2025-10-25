# detect_audio.py
import argparse, tempfile, os, sys
from pathlib import Path
import ffmpeg  # pip install ffmpeg-python

# Your current sets
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv"}

def extract_audio_to_wav(src_path: Path, target_sr: int = 16000) -> Path:
    tmp_wav = Path(tempfile.mkstemp(suffix=".wav")[1])
    (
        ffmpeg
        .input(str(src_path))
        .output(str(tmp_wav), format='wav', acodec='pcm_s16le', ac=1, ar=str(target_sr), loglevel="error")
        .overwrite_output()
        .run()
    )
    return tmp_wav

# ============================
# ADD START: backend-3 adapter
# ============================
def _bundle_has_trainer_format(bundle_dir: Path) -> bool:
    """Check for trainer-style bundle: scaler.joblib, model.joblib, meta.json."""
    return all((bundle_dir / name).exists() for name in ("scaler.joblib", "model.joblib", "meta.json"))

def _bundle_has_aiaudio_detector(bundle_dir: Path) -> bool:
    """
    Check for ai_audio_detector format: models/ai_audio_detector.joblib
    We allow either `bundle_dir/models/...` or just `bundle_dir/ai_audio_detector.joblib`.
    """
    return (bundle_dir / "models" / "ai_audio_detector.joblib").exists() or \
           (bundle_dir / "ai_audio_detector.joblib").exists()

def predict_clip(bundle_dir, media_path, agg="mean", threshold=0.5):
    """
    Unified callable for backend-3:
      - If `bundle_dir` looks like your trainer bundle (scaler/model/meta), use that.
      - Else, try ai_audio_detector (AIAudioDetector) in bundle_dir.
    Returns dict with keys at least: prob_ai (0..1), label ("ai"/"real").
    """
    bundle = Path(bundle_dir)
    media_path = Path(media_path)

    # Case 1: Trainer bundle flow (uses trainer.extract_features)
    if _bundle_has_trainer_format(bundle):
        import json, joblib, numpy as np, soundfile as sf, librosa
        from trainer import extract_features

        # Read audio (use librosa directly for audio, or extract from video first)
        if media_path.suffix.lower() in AUDIO_EXTS:
            # librosa loads straight from file
            y, sr = librosa.load(str(media_path), sr=None, mono=True)
        else:
            # extract wav and load via soundfile to avoid double decode
            tmp = extract_audio_to_wav(media_path)
            try:
                data, sr = sf.read(str(tmp), dtype="float32")
                if getattr(data, "ndim", 1) > 1:
                    data = data.mean(axis=1)
                y = data
            finally:
                try: os.remove(tmp)
                except OSError: pass

        meta = json.load(open(bundle / "meta.json", "r"))
        sr_tgt   = int(meta.get("sr", 16000))
        win_sec  = float(meta.get("win_sec", 6.0))
        hop_sec  = float(meta.get("stride_sec", 3.0))
        labelmap = meta.get("label_map", {"0": "real", "1": "ai"})

        # resample if needed
        if sr != sr_tgt:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_tgt)
            sr = sr_tgt

        win = int(sr * win_sec)
        hop = int(sr * hop_sec)

        def window_indices(n, win, hop):
            if n < win:
                yield 0, n
            else:
                for s in range(0, n - win + 1, hop):
                    yield s, s + win

        feats = []
        for s, e in window_indices(len(y), win, hop):
            seg = y[s:e]
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)))
            feats.append(extract_features(seg, sr))

        X = np.stack(feats, axis=0)
        X = np.asarray(X, dtype=np.float32, order="C")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = joblib.load(bundle / "scaler.joblib")
        clf    = joblib.load(bundle / "model.joblib")
        Xs = scaler.transform(X)
        probs = clf.predict_proba(Xs)[:, 1]

        if agg == "mean":
            clip_prob = float(np.mean(probs))
        elif agg == "max":
            clip_prob = float(np.max(probs))
        else:  # "vote"
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

    # Case 2: ai_audio_detector flow (AIAudioDetector)
    elif _bundle_has_aiaudio_detector(bundle):
        # Try to import the package and run it
        from ai_audio_detector import AIAudioDetector
        # Determine base_dir so detector finds ./models/ai_audio_detector.joblib
        base_dir = bundle if (bundle / "models").exists() else bundle.parent
        detector = AIAudioDetector(base_dir=Path(base_dir))

        # Ensure we pass a WAV file path (use extract if video)
        if media_path.suffix.lower() in AUDIO_EXTS:
            audio_path = media_path
            cleanup = None
        else:
            audio_path = extract_audio_to_wav(media_path)
            cleanup = audio_path
        try:
            res = detector.predict_file(str(audio_path))
        finally:
            if cleanup and cleanup.exists():
                try: os.remove(cleanup)
                except OSError: pass

        # Map result to unified schema
        is_ai = bool(res.get("is_ai", False))
        conf  = float(res.get("confidence", 0.0))
        return {
            "file": str(media_path),
            "prob_ai": conf,               # treat model's confidence as prob_ai
            "label": "ai" if is_ai else "real",
            "segment_probs": [conf],       # no segments; single value
            "win_sec": None,
            "hop_sec": None,
        }

    # Neither bundle type found
    else:
        raise RuntimeError(
            f"Bundle not recognized at {bundle}.\n"
            "Expected either trainer format (scaler.joblib, model.joblib, meta.json)\n"
            "or ai_audio_detector format (models/ai_audio_detector.joblib)."
        )
# ==========================
# ADD END: backend-3 adapter
# ==========================

def main():
    p = argparse.ArgumentParser(description="Detect likelihood that audio is AI-generated.")
    p.add_argument("path", help="Path to .mp3 audio or .mp4 video")
    p.add_argument("--bundle", default="./ai_audiodetector",
                   help="Bundle directory (trainer format) or base dir containing models/ for AIAudioDetector")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)

    try:
        result = predict_clip(bundle_dir=args.bundle, media_path=str(path), agg="mean", threshold=args.threshold)
    except Exception as e:
        print(f"Detector failed: {e}", file=sys.stderr)
        sys.exit(4)

    # Human-friendly CLI output (unchanged spirit)
    label = "AI" if result.get("label") == "ai" else "Human"
    conf = float(result.get("prob_ai", 0.0))
    print(f"Prediction: {label}")
    print(f"Confidence: {conf:.3f}")  # 0–1 scale

if __name__ == "__main__":
    main()
