from __future__ import annotations
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from controller import analyze_video_return

app = FastAPI(title="Sora2Detector API", version="1.0.0")

# CORS (loose for local dev; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def _call_analyzer(video_path: str, fast_mode: bool):
    """
    Call controller.analyze_video_return with fast_mode if supported.
    Fallback: use FAST_MODE env var for older controllers that don't accept fast_mode kwarg.
    """
    try:
        # Prefer explicit kwarg (new controller)
        return analyze_video_return(video_path, fast_mode=fast_mode)  # type: ignore[arg-type]
    except TypeError:
        # Older controller: toggle env var around the call
        prev = os.environ.get("FAST_MODE")
        try:
            if fast_mode:
                os.environ["FAST_MODE"] = "1"
            else:
                if "FAST_MODE" in os.environ:
                    del os.environ["FAST_MODE"]
            return analyze_video_return(video_path)
        finally:
            # restore env
            if prev is None:
                os.environ.pop("FAST_MODE", None)
            else:
                os.environ["FAST_MODE"] = prev or ""

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    fast: str | None = Form(default=None)  # <-- captures checkbox "fast" from the HTML form
):
    # Basic content-type check (best-effort)
    if file.content_type not in {"video/mp4", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Please upload an MP4 video.")

    # Stream upload to a temp .mp4 file (minimizes memory)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    # Parse fast flag from form
    fast_mode = str(fast).lower() in {"1", "true", "yes", "on"}

    try:
        data = _call_analyzer(str(tmp_path), fast_mode=fast_mode)

        # Include telemetry if present (helps verify Watermark was skipped)
        payload = {
            "file": data.get("file"),
            "results": data.get("results"),
            "overall_ai": data.get("overall_ai"),
        }
        if "telemetry" in data:
            payload["telemetry"] = data["telemetry"]

        return JSONResponse(content=payload)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# Mount static UI AFTER routes so /analyze isn't shadowed by StaticFiles
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

# Run (dev):
# uvicorn api:app --reload --port 8000
