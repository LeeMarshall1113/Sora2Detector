# controller.py
import subprocess, sys, json, re
from pathlib import Path

BACKENDS = [
    ("Metadata", "backend.py"),
    ("Watermark", "backend-2.py"),
    ("Audio", "backend-3.py"),
    ("Video", "backend-4.py"),
]

ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def _extract_last_json_blob(s: str) -> dict | None:
    end = s.rfind("}")
    if end == -1:
        return None
    depth = 0
    for i in range(end, -1, -1):
        ch = s[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                blob = s[i:end+1]
                try:
                    obj = json.loads(blob)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None

def choose_short(s: str) -> str:
    """Return a meaningful summary line, or 'AI' if nothing is detected."""
    if not s:
        return "AI"

    s = ANSI.sub("", s.replace("\r", "\n"))

    # 1) Try to parse JSON
    obj = _extract_last_json_blob(s)
    if obj is not None:
        for k in ("summary", "short", "result", "message"):
            if obj.get(k):
                return str(obj[k])[:500]
        try:
            return json.dumps(obj, ensure_ascii=False)[:500]
        except Exception:
            pass

    # 2) Keyword-based scan
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if lines:
        keywords = ("watermark", "found", "confidence", "detected", "result")
        for kw in keywords:
            cand = [ln for ln in lines if kw in ln.lower()]
            if cand:
                return cand[-1][:500]
        return lines[0][:500]

    # 3) No usable text -> AI
    return "AI"


def run_backend(script: str, video_path: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, script, video_path],
            capture_output=True, text=True, timeout=120
        )
        output = (result.stdout or "").strip() or (result.stderr or "").strip()
        return output if output else "AI"
    except subprocess.TimeoutExpired:
        return f"{Path(script).name}: TIMEOUT"
    except Exception as e:
        return f"{Path(script).name}: ERROR ({e})"


def analyze_video_return(video_path: str) -> dict:
    results: dict[str, dict] = {}

    for name, script in BACKENDS:
        if not Path(script).exists():
            results[name] = {"short": f"{script} not found", "full": f"[{name}] {script} not found."}
            continue
        full = run_backend(script, video_path)
        short = choose_short(full)
        results[name] = {"short": short, "full": full}

    return {"file": str(Path(video_path).resolve()), "results": results}


def analyze_video(video_path: str):
    print(f"=== Sora2Detector Combined Analysis ===")
    print(f"File: {video_path}\n")

    data = analyze_video_return(video_path)
    for name, payload in data["results"].items():
        print(f"{name}: {payload['short']}")

    print("\n=== End of Analysis ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python controller.py <path_to_video>")
        sys.exit(1)
    analyze_video(sys.argv[1])
