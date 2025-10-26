# gate.py
import os, json, time, secrets, pathlib
from typing import Dict, List
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.requests import Request

DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
PENDING_FILE = DATA_DIR / "pending.json"
CODES_FILE   = DATA_DIR / "codes.json"
SESSION_TTL = int(os.getenv("SESSION_TTL_HOURS", "48")) * 3600

def _load(path: pathlib.Path, default):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _save(path: pathlib.Path, obj):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)

def list_pending() -> List[Dict]:
    return _load(PENDING_FILE, [])

def add_pending(email: str, note: str = ""):
    items = list_pending()
    items.append({"id": secrets.token_hex(8), "email": email, "note": note, "ts": int(time.time())})
    _save(PENDING_FILE, items)

def remove_pending(pid: str):
    items = [p for p in list_pending() if p["id"] != pid]
    _save(PENDING_FILE, items)

def list_codes() -> List[Dict]:
    return _load(CODES_FILE, [])

def add_code(label: str, uses: int = 1, ttl_hours: int = 48):
    codes = list_codes()
    code = secrets.token_urlsafe(16)
    exp = int(time.time()) + ttl_hours*3600 if ttl_hours > 0 else 0
    codes.append({"code": code, "label": label, "uses": uses, "exp": exp})
    _save(CODES_FILE, codes)
    return code

def consume_code(code: str) -> bool:
    now = int(time.time())
    changed = False
    codes = list_codes()
    for c in codes:
        if c["code"] == code:
            if c["exp"] and now > c["exp"]:
                return False
            if c["uses"] <= 0:
                return False
            c["uses"] -= 1
            changed = True
            _save(CODES_FILE, codes)
            return True
    if changed:
        _save(CODES_FILE, codes)
    return False

OPEN_PATHS = ("/healthz", "/access", "/access/", "/access/request", "/access/login",
              "/docs", "/openapi.json", "/access/request")

class ApprovalGate(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/static/") or path in OPEN_PATHS:
            return await call_next(request)

        # allow admin endpoints to be reached; those are separately basic-auth protected
        if path.startswith("/admin"):
            return await call_next(request)

        sess = request.session or {}
        ok = sess.get("approved") is True
        exp = sess.get("approved_exp", 0)
        now = int(time.time())
        if ok and (exp == 0 or now < exp):
            return await call_next(request)

        # API callers: return 401 JSON; browser UI: redirect to /access
        wants_json = "application/json" in (request.headers.get("accept") or "")
        if wants_json or path.startswith("/api"):
            return JSONResponse({"detail": "Approval required"}, status_code=401)
        return RedirectResponse("/access")
