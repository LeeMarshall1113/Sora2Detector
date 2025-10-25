from __future__ import annotations
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
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

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # Basic content-type check (best-effort)
    if file.content_type not in {"video/mp4", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Please upload an MP4 video.")

    # Stream upload to a temp .mp4 file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)

    try:
        data = analyze_video_return(str(tmp_path))
        # ⬇️ include overall_ai in response
        return JSONResponse(content={
            "file": data["file"],
            "results": data["results"],
            "overall_ai": data.get("overall_ai"),
        })
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

# Mount static UI AFTER routes so /analyze isn't shadowed by StaticFiles
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

# Run:
# uvicorn api:app --reload --port 8000
