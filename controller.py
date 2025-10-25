# controller.py
import subprocess, sys, json, re
from pathlib import Path

# === Detector Configuration ===
BACKENDS = [
    ("Metadata", "backend.py"),     # 1
    ("Watermark", "backend-2.py"),  # 2
    ("Audio", "backend-3.py"),      # 3
    ("Video", "backend-4.py"),      # 4
]

# === Rule Settings ===
DET1 = "Metadata"
DET2 = "Watermark"
DET3 = "Audio"
DET4 = "Video"

AUDIO_WEIGHT = 0.6   # bias toward detector 3
VIDEO_WEIGHT = 0.4
THRESHOLD = 0.50     # >50% → AI

ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# =====================================================
# --- Output Summarization ---
# =====================================================

def _extract_last_json_blob(s: str) -> dict | None:
    """Extract last {...} JSON block in a string."""
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
    """Pick a meaningful summary line or return 'AI' if none."""
    if not s:
        return "AI"
    s = ANSI.sub("", s.replace("\r", "\n"))
    obj = _extract_last_json_blob(s)
    if obj is not None:
        for k in ("summary", "short", "result", "message"):
            if obj.get(k):
                return str(obj[k])[:500]
        try:
            return json.dumps(obj, ensure_ascii=False)[:500]
        except Exception:
            pass
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if lines:
        keywords = ("watermark", "found", "confidence", "detected", "result")
        for kw in keywords:
            cand = [ln for ln in lines if kw in ln.lower()]
            if cand:
                return cand[-1][:500]
        return lines[0][:500]
    return "AI"

# =====================================================
# --- Run Backends ---
# =====================================================

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

# =====================================================
# --- Probability Parsing & Rule Logic ---
# =====================================================

RE_LABELED = re.compile(
    r"(confidence|prob(?:ability)?|score|ai[_\s-]*prob)\s*[:=]\s*(\d{1,3}\s*%|0?\.\d+|1\.0+)",
    re.I,
)
RE_PERCENT = re.compile(r"(\d{1,3})\s*%")
RE_PROB01  = re.compile(r"\b(0?\.\d+|1\.0+)\b")
ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def _clean(text: str) -> str:
    if not text:
        return ""
    t = ANSI.sub("", text.replace("\r", "\n"))
    return re.sub(r"[ \t]+", " ", t)

def _normalize_val(tok: str) -> float | None:
    tok = tok.strip()
    if tok.endswith("%"):
        try:
            v = float(tok[:-1].strip())
            return v/100.0 if 0 <= v <= 100 else None
        except:
            return None
    try:
        v = float(tok)
        return v if 0.0 <= v <= 1.0 else None
    except:
        return None

def _parse_prob(text: str) -> float | None:
    """Prefer labeled confidence; else use max numeric percentage/0..1."""
    t = _clean(text)
    if not t:
        return None

    # 1) labeled is strongest
    labeled = []
    for m in RE_LABELED.finditer(t):
        val = _normalize_val(m.group(2))
        if val is not None:
            labeled.append(val)
    if labeled:
        return max(labeled)

    # 2) all candidates
    candidates = []
    for m in RE_PERCENT.finditer(t):
        span = m.span()
        ctx = t[max(0, span[0]-6):min(len(t), span[1]+6)].lower()
        if any(s in ctx for s in ["]", "[", "progress", "load", "0/"]):
            continue
        val = _normalize_val(m.group(1) + "%")
        if val is not None:
            candidates.append(val)
    for m in RE_PROB01.finditer(t):
        span = m.span()
        ctx = t[max(0, span[0]-3):min(len(t), span[1]+3)].lower()
        if any(s in ctx for s in ["ms", "s ", "px", "kb", "mb"]):
            continue
        val = _normalize_val(m.group(1))
        if val is not None:
            candidates.append(val)

    return max(candidates) if candidates else None

def _heuristic_lights_up(text: str) -> bool:
    if not text:
        return False
    t = _clean(text).lower()
    pos = any(k in t for k in [
        "ai-generated", "synthetic", "detected", "watermark", "model fingerprint",
        "deepfake", "diffusion", "sora"
    ])
    neg = any(k in t for k in [
        "no watermark", "authentic", "camera make", "exif intact"
    ])
    return pos and not neg

def _get_prob_or_flag(payload: dict) -> tuple[float | None, bool]:
    short = (payload.get("short") or "").strip()
    full = (payload.get("full") or "").strip()
    p = _parse_prob(short) or _parse_prob(full)
    lights = (p is not None and p >= 0.5) or _heuristic_lights_up(short) or _heuristic_lights_up(full)
    return p, lights

def _overall_rule(results: dict[str, dict]) -> dict:
    """If 1 or 2 lights up → AI; else weighted avg(3,4). Also expose VIDEO pct for UI mirroring."""
    d1 = results.get(DET1, {"short": "", "full": ""})
    d2 = results.get(DET2, {"short": "", "full": ""})
    d3 = results.get(DET3, {"short": "", "full": ""})
    d4 = results.get(DET4, {"short": "", "full": ""})

    p1, light1 = _get_prob_or_flag(d1)
    p2, light2 = _get_prob_or_flag(d2)
    p3, _ = _get_prob_or_flag(d3)
    p4, _ = _get_prob_or_flag(d4)

    # Compute the integer percentage for Video so the UI can mirror it
    video_pct = int(round((p4 * 100))) if isinstance(p4, (int, float)) else None

    if light1 or light2:
        return {
            "decision": "AI",
            "method": "rule_v1",
            "details": {
                "reason": f"{DET1 if light1 else DET2} lit up",
                "p": {DET1: p1, DET2: p2, DET3: p3, DET4: p4},
                "video_pct": video_pct,           # ⬅️ expose for UI
                "weights": {DET3: AUDIO_WEIGHT, DET4: VIDEO_WEIGHT},
                "threshold": THRESHOLD
            }
        }

    a = p3 if p3 is not None else 0.5
    v = p4 if p4 is not None else 0.5
    weighted = (AUDIO_WEIGHT * a + VIDEO_WEIGHT * v) / (AUDIO_WEIGHT + VIDEO_WEIGHT)
    decision = "AI" if weighted > THRESHOLD else "Human"

    return {
        "decision": decision,
        "method": "rule_v1",
        "details": {
            "reason": f"weighted_avg({DET3},{DET4}) = {weighted:.3f} {'>' if weighted>THRESHOLD else '<='} {THRESHOLD}",
            "weighted_avg": round(weighted, 4),
            "p": {DET1: p1, DET2: p2, DET3: p3, DET4: p4},
            "video_pct": video_pct,               # ⬅️ expose for UI
            "weights": {DET3: AUDIO_WEIGHT, DET4: VIDEO_WEIGHT},
            "threshold": THRESHOLD
        }
    }

# =====================================================
# --- Main API + CLI Interface ---
# =====================================================

def analyze_video_return(video_path: str) -> dict:
    results: dict[str, dict] = {}
    for name, script in BACKENDS:
        if not Path(script).exists():
            results[name] = {"short": f"{script} not found", "full": f"[{name}] {script} not found."}
            continue
        full = run_backend(script, video_path)
        short = choose_short(full)
        results[name] = {"short": short, "full": full}

    overall = _overall_rule(results)
    return {
        "file": str(Path(video_path).resolve()),
        "results": results,
        "overall_ai": overall
    }

def analyze_video(video_path: str):
    print(f"=== Sora2Detector Combined Analysis ===")
    print(f"File: {video_path}\n")
    data = analyze_video_return(video_path)
    for name, payload in data["results"].items():
        print(f"{name}: {payload['short']}")
    overall = data["overall_ai"]
    print(f"\n[Overall Decision] → {overall['decision']} ({overall['details']['reason']})")
    if overall["details"].get("video_pct") is not None:
        print(f"[Video Estimate] → {overall['details']['video_pct']}%")
    print("\n=== End of Analysis ===")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python controller.py <path_to_video>")
        sys.exit(1)
    analyze_video(sys.argv[1])
