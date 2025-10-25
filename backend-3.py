#!/usr/bin/env python3
# backend-3.py — clean output version for production use
# Prints exactly: "Chance of being AI: XX%"
# Exit codes: 1 (AI >= threshold), 0 (Not AI), 2 (error)

import argparse, os, sys, importlib, importlib.util, json, traceback
from pathlib import Path

def load_module_from_path(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    sys.modules[mod_name] = mod
    return mod

def main():
    ap = argparse.ArgumentParser(description="Detect likelihood of AI-generated audio.")
    ap.add_argument("media", help="Path to video/audio file")
    ap.add_argument("--bundle", default=r".\ai_audiodetector", help="Bundle folder with model.joblib, scaler.joblib, meta.json")
    ap.add_argument("--threshold", type=float, default=0.55, help="AI threshold (default 0.55)")
    ap.add_argument("--detect-audio", help="Path to detect_audio.py (if not importable)")
    ap.add_argument("--trainer", help="Path to trainer.py (if not importable)")
    args = ap.parse_args()

    try:
        # Load trainer if given
        if args.trainer:
            trainer_path = Path(args.trainer)
            load_module_from_path("trainer", str(trainer_path))

        # Load detect_audio
        if args.detect_audio:
            da_path = Path(args.detect_audio)
            detect_audio = load_module_from_path("detect_audio", str(da_path))
        else:
            import detect_audio

        # Run detection
        result = detect_audio.predict_clip(
            bundle_dir=args.bundle,
            media_path=args.media,
            agg="mean",
            threshold=args.threshold,
        )

        prob_ai = float(result.get("prob_ai", 0.0))
        percent = int(round(prob_ai * 100))
        print(f"Chance of being AI: {percent}%")
        sys.exit(1 if prob_ai >= args.threshold else 0)

    except Exception:
        # Quiet single-line error
        print("ERROR")
        sys.exit(2)

if __name__ == "__main__":
    main()
