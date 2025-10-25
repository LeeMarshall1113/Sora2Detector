import subprocess, sys
from pathlib import Path

def run_backend(script: str, video_path: str) -> str:
    """Run a backend .py file with the given video, return its stdout (stripped)."""
    try:
        result = subprocess.run(
            [sys.executable, script, video_path],
            capture_output=True, text=True, timeout=120
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output if output else "No output"
    except subprocess.TimeoutExpired:
        return f"{Path(script).name}: TIMEOUT"
    except Exception as e:
        return f"{Path(script).name}: ERROR ({e})"


def analyze_video(video_path: str):
    """
    Run all backend detectors on one file and print combined results.
    """
    backends = [
        ("Metadata", "backend.py"),     # sora hex/embed scan
        ("Watermark", "backend-2.py"),  # watermark visual OCR detector
        ("Audio", "backend-3.py"),      # audio classifier
        ("Video", "backend-4.py"),      # motion/video AI detector
    ]

    print(f"=== Sora2Detector Combined Analysis ===")
    print(f"File: {video_path}\n")

    for name, script in backends:
        if not Path(script).exists():
            print(f"[{name}] {script} not found.")
            continue
        output = run_backend(script, video_path)
        # Take only the last non-empty line for clean summary if you prefer:
        lines = [l for l in output.splitlines() if l.strip()]
        short = lines[-1] if lines else output
        print(f"{name}: {short}")

    print("\n=== End of Analysis ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python controller.py <path_to_video>")
        sys.exit(1)
    analyze_video(sys.argv[1])
