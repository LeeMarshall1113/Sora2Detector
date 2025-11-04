#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sora2 batch tester (single-file report) with frontend-parity decision logic.
- Stage 1: If Metadata or Watermark "lights up" -> AI (ai_source = that detector)
- Stage 2: Else weighted(Audio, Video) vs threshold -> AI / Not AI (ai_source = "Weighted")
- 3-class mapping: Stage1 AI -> sora-watermark; Stage2 AI -> sora-no-watermark; Not AI -> real
- Writes ONE Markdown file with per-file rows + backend coverage + metrics
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ===========================
# Config
# ===========================
DEFAULT_SCRIPT_DIR = Path(__file__).parent.resolve()  # used if --backend-dir not provided
BACKENDS = [
    ("Metadata",  "backend.py"),     # 1
    ("Watermark", "backend-2.py"),   # 2
    ("Audio",     "backend-3.py"),   # 3
    ("Video",     "backend-4.py"),   # 4
]
DET1 = "Metadata"
DET2 = "Watermark"
DET3 = "Audio"
DET4 = "Video"

AUDIO_WEIGHT = 0.6
VIDEO_WEIGHT = 0.4
THRESHOLD = 0.50      # >50% => AI

# Canonical 3-class labels
CANON = ["real", "sora-watermark", "sora-no-watermark"]

# Infer GT from filename
RX_REAL = re.compile(r"(^|[^a-z])real([^a-z]|$)", re.I)
RX_SWM  = re.compile(r"sora[-_ ]?watermark", re.I)
RX_SNW  = re.compile(r"sora[-_ ]?no[-_ ]?watermark", re.I)

def infer_gt(stem: str):
    if RX_SNW.search(stem): return "sora-no-watermark"
    if RX_SWM.search(stem): return "sora-watermark"
    if RX_REAL.search(stem): return "real"
    return None

# ===========================
# Controller logic inlined
# ===========================
ANSI = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def _extract_last_json_blob(s: str):
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

def choose_short(s: str):
    if not s: return "AI"
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

def run_backend(script_dir: Path, script_name: str, video_path: str, timeout: float, verbose: bool=False):
    """Return (text, status, ms). status in {OK, NOT_FOUND, TIMEOUT, ERROR, EMPTY_OUTPUT}."""
    script_path = (script_dir / script_name).resolve()
    start = time.perf_counter()
    status = "OK"
    text = ""
    try:
        if not script_path.exists():
            status = "NOT_FOUND"
            text = f"{script_path.name}: NOT_FOUND"
        else:
            proc = subprocess.run(
                [sys.executable, str(script_path), video_path],
                capture_output=True, text=True, timeout=timeout
            )
            text = (proc.stdout or "").strip() or (proc.stderr or "").strip()
            if text == "":
                status = "EMPTY_OUTPUT"
                text = "EMPTY_OUTPUT"
        ms = int((time.perf_counter() - start) * 1000)
        if verbose:
            print(f"[{script_path.name:<12}] {status:>11} {ms:>5} ms  {video_path}")
        return text, status, ms
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
        text = f"{script_path.name}: TIMEOUT"
        ms = int((time.perf_counter() - start) * 1000)
        if verbose:
            print(f"[{script_path.name:<12}] {status:>11} {ms:>5} ms  {video_path}")
        return text, status, ms
    except Exception as e:
        status = "ERROR"
        text = f"{script_path.name}: ERROR ({e})"
        ms = int((time.perf_counter() - start) * 1000)
        if verbose:
            print(f"[{script_path.name:<12}] {status:>11} {ms:>5} ms  {video_path}")
        return text, status, ms

RE_LABELED = re.compile(
    r"(confidence|prob(?:ability)?|score|ai[_\s-]*prob)\s*[:=]\s*(\d{1,3}\s*%|0?\.\d+|1\.0+)",
    re.I,
)
RE_PERCENT = re.compile(r"(\d{1,3})\s*%")
RE_PROB01  = re.compile(r"\b(0?\.\d+|1\.0+)\b")

def _clean(text: str):
    if not text: return ""
    t = ANSI.sub("", text.replace("\r", "\n"))
    return re.sub(r"[ \t]+", " ", t)

def _normalize_val(tok: str):
    tok = tok.strip()
    if tok.endswith("%"):
        try:
            v = float(tok[:-1].strip())
            return v/100.0 if 0 <= v <= 100 else None
        except Exception:
            return None
    try:
        v = float(tok)
        return v if 0.0 <= v <= 1.0 else None
    except Exception:
        return None

def _parse_prob(text: str):
    t = _clean(text)
    if not t: return None
    labeled = []
    for m in RE_LABELED.finditer(t):
        val = _normalize_val(m.group(2))
        if val is not None:
            labeled.append(val)
    if labeled:
        return max(labeled)
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
    if not text: return False
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
    full  = (payload.get("full") or "").strip()
    p = _parse_prob(short) or _parse_prob(full)
    lights = (p is not None and p >= 0.5) or _heuristic_lights_up(short) or _heuristic_lights_up(full)
    return p, lights

# === Frontend-parity overall rule ===
def _overall_rule(results: dict[str, dict]) -> dict:
    d1 = results.get(DET1, {"short": "", "full": ""})
    d2 = results.get(DET2, {"short": "", "full": ""})
    d3 = results.get(DET3, {"short": "", "full": ""})
    d4 = results.get(DET4, {"short": "", "full": ""})

    p1, light1 = _get_prob_or_flag(d1)
    p2, light2 = _get_prob_or_flag(d2)
    p3, _ = _get_prob_or_flag(d3)
    p4, _ = _get_prob_or_flag(d4)

    audio_pct = int(round(p3 * 100)) if isinstance(p3, (int, float)) else None
    video_pct = int(round(p4 * 100)) if isinstance(p4, (int, float)) else None

    # Stage 1: short-circuit like the website
    if light1 or light2:
        ai_src = DET1 if light1 else DET2
        return {
            "ai_binary": "AI",
            "ai_source": ai_src,            # which detector tripped it
            "decision": "AI",
            "method": "rule_v1",
            "details": {
                "reason": f"{ai_src} lit up",
                "p": {DET1: p1, DET2: p2, DET3: p3, DET4: p4},
                "audio_pct": audio_pct,
                "video_pct": video_pct,
                "weights": {DET3: AUDIO_WEIGHT, DET4: VIDEO_WEIGHT},
                "threshold": THRESHOLD
            }
        }

    # Stage 2: weighted Audio/Video
    a = p3 if p3 is not None else 0.5
    v = p4 if p4 is not None else 0.5
    weighted = (AUDIO_WEIGHT * a + VIDEO_WEIGHT * v) / (AUDIO_WEIGHT + VIDEO_WEIGHT)
    is_ai = weighted > THRESHOLD
    return {
        "ai_binary": "AI" if is_ai else "Not AI",
        "ai_source": "Weighted",          # parity with frontend
        "decision": "AI" if is_ai else "Human",
        "method": "rule_v1",
        "details": {
            "reason": f"weighted_avg({DET3},{DET4}) = {weighted:.3f} {'>' if is_ai else '<='} {THRESHOLD}",
            "weighted_avg": round(weighted, 4),
            "p": {DET1: p1, DET2: p2, DET3: p3, DET4: p4},
            "audio_pct": audio_pct,
            "video_pct": video_pct,
            "weights": {DET3: AUDIO_WEIGHT, DET4: VIDEO_WEIGHT},
            "threshold": THRESHOLD
        }
    }

def _map_to_3class(overall: dict) -> str:
    ai_bin = overall.get("ai_binary")         # "AI" or "Not AI"
    ai_src = (overall.get("ai_source") or "").lower()
    if ai_bin != "AI":
        return "real"
    if ai_src in ("metadata", "watermark"):
        return "sora-watermark"
    return "sora-no-watermark"

# ===========================
# Batch plumbing
# ===========================
def analyze_one_file(path: Path, backend_timeout: float, backend_dir: Path, verbose=False):
    t0 = time.perf_counter()
    results = {}
    diag = {}  # name -> {status, ms}
    for name, script in BACKENDS:
        full, status, ms = run_backend(backend_dir, script, str(path), backend_timeout, verbose=verbose)
        short = choose_short(full)
        results[name] = {"short": short, "full": full}
        diag[name] = {"status": status, "ms": ms}

    overall = _overall_rule(results)
    label3 = _map_to_3class(overall)
    details = overall.get("details", {}) or {}
    runtime_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "file": str(path),
        "label3": label3,
        "overall": overall,
        "reason": details.get("reason"),
        "audio_pct": details.get("audio_pct"),
        "video_pct": details.get("video_pct"),
        "runtime_ms": runtime_ms,
        "diag": diag  # per-backend diagnostics
    }

def confusion(y_true, y_pred, labels):
    m = {t:{p:0 for p in labels} for t in labels}
    for t,p in zip(y_true,y_pred):
        if t in labels and p in labels:
            m[t][p]+=1
    return m

def prf1(cm, labels):
    out={}
    for c in labels:
        tp = cm[c][c]
        fp = sum(cm[t][c] for t in labels if t!=c)
        fn = sum(cm[c][p] for p in labels if p!=c)
        prec = tp/(tp+fp) if tp+fp else 0.0
        rec  = tp/(tp+fn) if tp+fn else 0.0
        f1   = (2*prec*rec)/(prec+rec) if prec+rec else 0.0
        out[c] = dict(precision=prec, recall=rec, f1=f1, support=tp+fn)
        for k in ("precision","recall","f1"):
            out[c][k] = float(out[c][k])
    return out

def write_single_markdown(out_path: Path, merged_rows, cm, metrics, acc, correct, total, args, src_display):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # derive backend coverage + any errors
    backends = [name for name,_ in BACKENDS]
    coverage = {b: {"OK":0,"NOT_FOUND":0,"TIMEOUT":0,"ERROR":0,"EMPTY_OUTPUT":0,"UNKNOWN":0} for b in backends}
    any_problem = False
    for m in merged_rows:
        diag = m.get("diag") or {}
        for b in backends:
            s = (diag.get(b) or {}).get("status", "UNKNOWN")
            if s not in coverage[b]:
                coverage[b][s] = 0
            coverage[b][s] += 1
            if s in ("NOT_FOUND","TIMEOUT","ERROR"):
                any_problem = True

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Sora2 Batch Report\n\n")
        if any_problem:
            f.write("> **⚠️ Diagnostics:** One or more backends reported `NOT_FOUND`, `TIMEOUT`, or `ERROR`. See Backend Coverage and per-file diagnostics below.\n\n")

        f.write(f"- Generated: **{ts}**\n")
        f.write(f"- Source: `{src_display}`\n")
        f.write(f"- Threshold: **{THRESHOLD:.2f}**, Weights: Audio **{AUDIO_WEIGHT:.2f}**, Video **{VIDEO_WEIGHT:.2f}**\n")
        f.write(f"- Backend timeout: **{args.timeout}s**, Workers: **{args.workers}**\n")
        f.write(f"- Backend dir: `{args.backend_dir if args.backend_dir else DEFAULT_SCRIPT_DIR}`\n\n")

        # Backend coverage summary
        f.write("## Backend Coverage\n\n")
        f.write("| Backend | OK | NOT_FOUND | TIMEOUT | ERROR | EMPTY_OUTPUT | UNKNOWN |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for b in backends:
            row = coverage[b]
            f.write(f"| {b} | {row.get('OK',0)} | {row.get('NOT_FOUND',0)} | {row.get('TIMEOUT',0)} | {row.get('ERROR',0)} | {row.get('EMPTY_OUTPUT',0)} | {row.get('UNKNOWN',0)} |\n")

        # Per-file results (now includes AI? and Source)
        f.write("\n## Per-file Results\n\n")
        f.write("| # | File | GT | Pred | AI? | Source | Audio% | Video% | Reason | Runtime (ms) | Correct |\n")
        f.write("|---:|---|---|---|:--:|---|---:|---:|---|---:|:---:|\n")
        for i, m in enumerate(merged_rows, 1):
            reason = (m.get('reason') or '').replace('|','/')
            f.write(
                f"| {i} | `{m['file']}` | {m['gt']} | {m['pred'] or ''} | "
                f"{m.get('ai_binary','')} | {m.get('ai_source','')} | "
                f"{'' if m['audio_pct'] is None else m['audio_pct']} | "
                f"{'' if m['video_pct'] is None else m['video_pct']} | "
                f"{reason} | "
                f"{m.get('runtime_ms') or ''} | "
                f"{'✅' if m['correct'] else '❌'} |\n"
            )

        # Per-file backend diagnostics (collapsible)
        f.write("\n<details>\n<summary><strong>Per-file Backend Diagnostics</strong></summary>\n\n")
        f.write("| # | File | " + " | ".join([f"{b} status" for b in backends]) + " | " + " | ".join([f"{b} ms" for b in backends]) + " |\n")
        f.write("|---:|---|" + "|".join(["---:"]*len(backends)) + "|" + "|".join(["---:"]*len(backends)) + "|\n")
        for i, m in enumerate(merged_rows, 1):
            diag = m.get("diag") or {}
            statuses = [ (diag.get(b) or {}).get("status","") for b in backends ]
            times    = [ str((diag.get(b) or {}).get("ms","")) for b in backends ]
            f.write(f"| {i} | `{m['file']}` | " + " | ".join(statuses) + " | " + " | ".join(times) + " |\n")
        f.write("\n</details>\n")

        # Summary (confusion + metrics)
        f.write("\n---\n\n## Summary\n\n")
        f.write(f"- **Files evaluated:** {total}\n")
        f.write(f"- **Accuracy:** **{acc*100:.2f}%** ({correct}/{total})\n\n")

        # Confusion matrix
        f.write("### Confusion Matrix\n\n")
        f.write("| True \\ Pred | " + " | ".join(CANON) + " |\n")
        f.write("|---" + "|---"*len(CANON) + "|\n")
        for t in CANON:
            row = " | ".join(str(cm[t][p]) for p in CANON)
            f.write(f"| {t} | {row} |\n")

        # Per-class metrics
        f.write("\n### Per-class Metrics\n\n")
        f.write("| Class | Precision | Recall | F1 | Support |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for c in CANON:
            m=metrics[c]
            f.write(f"| {c} | {m['precision']*100:.2f}% | {m['recall']*100:.2f}% | {m['f1']*100:.2f}% | {m['support']} |\n")

        f.write("\n*End of report.*\n")

def main():
    ap = argparse.ArgumentParser(
        description="Sora2 batch tester (single-file report, frontend-parity)."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", help="Path to a single file OR a folder to scan.")
    src.add_argument("--folder", help="Explicit folder to scan (alternative to --root).")
    ap.add_argument("--backend-dir", default=None,
                    help="Directory containing backend.py, backend-2.py, backend-3.py, backend-4.py. "
                         "Defaults to this script's folder.")
    ap.add_argument("--extensions", default=".mp4,.mov,.mkv,.avi,.webm",
                    help="Comma-separated extensions when scanning a folder.")
    ap.add_argument("--pattern", default=None,
                    help="Optional glob (e.g. **/*.mp4). When set, overrides --extensions.")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Concurrent files.")
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-backend timeout (sec).")
    ap.add_argument("--out", default="batch_report.md", help="Output path (file or directory).")
    ap.add_argument("--strict-labels", action="store_true", help="Drop files whose GT can’t be inferred.")
    ap.add_argument("--verbose", action="store_true", help="Print per-backend timings and status during run.")
    args = ap.parse_args()

    backend_dir = Path(args.backend_dir).resolve() if args.backend_dir else DEFAULT_SCRIPT_DIR

    # Resolve source path
    src_path = Path(args.folder if args.folder else args.root)

    # Collect files
    if src_path.is_file():
        files = [src_path]
        src_display = str(src_path)
    else:
        if not src_path.exists():
            print(f"Folder not found: {src_path}", file=sys.stderr); sys.exit(2)
        if args.pattern:
            files = [p for p in src_path.rglob(args.pattern) if p.is_file()]
        else:
            exts = {
                ("." + e.lower().strip()) if not e.strip().startswith(".") else e.lower().strip()
                for e in args.extensions.split(",") if e.strip()
            }
            files = [p for p in src_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
        files.sort()
        src_display = str(src_path)

    if not files:
        print("No files found.", file=sys.stderr); sys.exit(2)

    # Ground truth list
    rows=[]
    skipped=0
    for p in files:
        gt = infer_gt(p.stem)
        if gt is None and args.strict_labels:
            skipped += 1
            continue
        rows.append(dict(file=str(p), gt=gt or "__unknown__"))

    if not rows:
        print("No labelable files (strict mode).", file=sys.stderr); sys.exit(3)
    if skipped:
        print(f"Skipped {skipped} files (strict labels).")

    # Run concurrently over files
    results=[]
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(analyze_one_file, Path(r["file"]), args.timeout, backend_dir, args.verbose) for r in rows]
        for f in cf.as_completed(futs):
            results.append(f.result())
    by_file = {r["file"]: r for r in results}

    # Merge rows for table + metrics
    merged=[]
    for r in rows:
        rr = by_file.get(r["file"])
        if rr is None:
            merged.append({
                "file": r["file"], "gt": r["gt"], "pred": None,
                "audio_pct": None, "video_pct": None,
                "ai_binary": "", "ai_source": "",
                "reason": "", "correct": False, "runtime_ms": None,
                "diag": {}
            })
            continue
        pred = rr["label3"]
        merged.append({
            "file": r["file"],
            "gt": r["gt"],
            "pred": pred,
            "audio_pct": rr.get("audio_pct"),
            "video_pct": rr.get("video_pct"),
            "ai_binary": rr["overall"].get("ai_binary"),
            "ai_source": rr["overall"].get("ai_source"),
            "reason": rr.get("reason"),
            "correct": (pred is not None and r["gt"] in CANON and pred==r["gt"]),
            "runtime_ms": rr.get("runtime_ms"),
            "diag": rr.get("diag", {})
        })

    # Metrics
    eval_rows=[m for m in merged if m["gt"] in CANON and m["pred"] in CANON]
    y_true=[m["gt"] for m in eval_rows]
    y_pred=[m["pred"] for m in eval_rows]

    def _confusion(y_true, y_pred, labels):
        m = {t:{p:0 for p in labels} for t in labels}
        for t,p in zip(y_true,y_pred):
            if t in labels and p in labels:
                m[t][p]+=1
        return m

    def _prf1(cm, labels):
        out={}
        for c in labels:
            tp = cm[c][c]
            fp = sum(cm[t][c] for t in labels if t!=c)
            fn = sum(cm[c][p] for p in labels if p!=c)
            prec = tp/(tp+fp) if tp+fp else 0.0
            rec  = tp/(tp+fn) if tp+fn else 0.0
            f1   = (2*prec*rec)/(prec+rec) if prec+rec else 0.0
            out[c] = dict(precision=prec, recall=rec, f1=f1, support=tp+fn)
        return out

    cm = _confusion(y_true,y_pred,CANON)
    total = sum(sum(cm[t].values()) for t in CANON)
    correct = sum(cm[c][c] for c in CANON)
    acc = (correct/total) if total else 0.0
    metrics = _prf1(cm,CANON)

    # Normalize output path: allow file or directory
    out_path = Path(args.out)
    try:
        is_dir = out_path.exists() and out_path.is_dir()
    except Exception:
        is_dir = False
    if is_dir or str(out_path).endswith(("\\", "/")):
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / "batch_report.md"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".md")

    # Write single markdown file
    write_single_markdown(out_path, merged, cm, metrics, acc, correct, total, args, src_display)

    print(f"Report written to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
