from __future__ import annotations
import subprocess, sys
from pathlib import Path
from typing import Dict, List

BACKENDS: List[tuple[str, str]] = [
    ("Metadata", "backend.py"),     # sora hex/embed scan
    ("Watermark", "backend-2.py"),  # watermark visual OCR detector
    ("Audio", "backend-3.py"),      # audio classifier
    ("Video", "backend-4.py"),      # motion/video AI detector
]

def run_backend(script: str, video_path: str) -> str:
    """Run a backend .py file with the given video, return its stdout (stripped)."""
    try:
        result = subprocess.run(
            [sys.executable, script, video_path],
            capture_output=True, text=True, timeout=120
        )
        output = (result.stdout or "").strip() or (result.stderr or "").strip()
        return output if output else "No output"
    except subprocess.TimeoutExpired:
        return f"{Path(script).name}: TIMEOUT"
    except Exception as e:
        return f"{Path(script).name}: ERROR ({e})"

def analyze_video_return(video_path: str) -> Dict[str, dict]:
    """Run all backend detectors and return a structured dict."""
    results: Dict[str, dict] = {}

    for name, script in BACKENDS:
        if not Path(script).exists():
            results[name] = {
                "short": f"{script} not found",
                "full": f"[{name}] {script} not found."
            }
            continue
        full = run_backend(script, video_path)
        lines = [l for l in full.splitlines() if l.strip()]
        short = lines[-1] if lines else full
        results[name] = {"short": short, "full": full}

    return {"file": str(Path(video_path).resolve()), "results": results}

def analyze_video_cli(video_path: str) -> None:
    print("=== Sora2Detector Combined Analysis ===")
    print(f"File: {video_path}\n")
    data = analyze_video_return(video_path)
    for name, payload in data["results"].items():
        print(f"{name}: {payload['short']}")
    print("\n=== End of Analysis ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python controller.py <path_to_video>")
        sys.exit(1)
    analyze_video_cli(sys.argv[1])
