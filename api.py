"""
api.py — Sora2Detector server (FastAPI + manual approval system)
"""

import os
import time
import secrets
from fastapi import FastAPI, Form, Depends, Request, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.sessions import SessionMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import List, Dict
import json
import pathlib

# ===============================================================
# Config
# ===============================================================

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "password")
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "48"))
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

PENDING_FILE = DATA_DIR / "pending.json"
CODES_FILE = DATA_DIR / "codes.json"


# ===============================================================
# Helpers for simple JSON-based storage
# ===============================================================

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
    exp = int(time.time()) + ttl_hours * 3600 if ttl_hours > 0 else 0
    codes.append({"code": code, "label": label, "uses": uses, "exp": exp})
    _save(CODES_FILE, codes)
    return code


def consume_code(code: str) -> bool:
    now = int(time.time())
    codes = list_codes()
    for c in codes:
        if c["code"] == code:
            if c["exp"] and now > c["exp"]:
                return False
            if c["uses"] <= 0:
                return False
            c["uses"] -= 1
            _save(CODES_FILE, codes)
            return True
    return False


# ===============================================================
# Middleware for approval checking
# ===============================================================

OPEN_PATHS = ("/healthz", "/access", "/access/", "/access/request",
              "/access/login", "/docs", "/openapi.json", "/admin")

class ApprovalGate(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path.startswith("/static/") or any(path.startswith(p) for p in OPEN_PATHS):
            return await call_next(request)

        sess = request.scope.get("session") or {}
        ok = sess.get("approved") is True
        exp = sess.get("approved_exp", 0)
        now = int(time.time())

        if ok and (exp == 0 or now < exp):
            return await call_next(request)

        wants_json = "application/json" in (request.headers.get("accept") or "")
        if wants_json or path.startswith("/api"):
            return JSONResponse({"detail": "Approval required"}, status_code=401)
        return RedirectResponse("/access")


# ===============================================================
# FastAPI app setup
# ===============================================================

app = FastAPI(title="Sora2Detector Access Controlled Server")

app.add_middleware(ProxyHeadersMiddleware)
app.add_middleware(ApprovalGate)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)


# Serve static UI if present
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# ===============================================================
# Health check
# ===============================================================

@app.get("/healthz")
def healthz():
    return {"ok": True}


# ===============================================================
# Access flow
# ===============================================================

ACCESS_HTML = """
<!doctype html><meta charset="utf-8">
<title>Request Access</title>
<h2>Request Access</h2>
<form method="post" action="/access/request">
  <label>Email</label><br><input name="email" type="email" required><br>
  <label>Reason (optional)</label><br><input name="note" type="text"><br><br>
  <button type="submit">Request</button>
</form>
<hr>
<h2>Got a code?</h2>
<form method="post" action="/access/login">
  <label>Access Code</label><br><input name="code" required><br><br>
  <button type="submit">Enter</button>
</form>
"""

@app.get("/access", response_class=HTMLResponse)
def access_page():
    return HTMLResponse(ACCESS_HTML)


@app.post("/access/request")
def access_request(email: str = Form(...), note: str = Form("")):
    add_pending(email=email.strip(), note=note.strip())
    return HTMLResponse("<p>Request submitted. An admin will review it.</p><a href='/access'>Back</a>")


@app.post("/access/login")
async def access_login(request: Request, code: str = Form(...)):
    code = code.strip()
    if consume_code(code):
        exp = int(time.time()) + SESSION_TTL_HOURS * 3600
        request.session["approved"] = True
        request.session["approved_exp"] = exp
        return RedirectResponse("/", status_code=303)
    return HTMLResponse("<p>Invalid or expired code.</p><a href='/access'>Back</a>", status_code=401)


# ===============================================================
# Admin (Basic Auth protected)
# ===============================================================

security = HTTPBasic()

def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    if not (secrets.compare_digest(credentials.username, ADMIN_USER) and
            secrets.compare_digest(credentials.password, ADMIN_PASS)):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


ADMIN_HTML_HEAD = "<!doctype html><meta charset='utf-8'><title>Admin</title>"

@app.get("/admin", response_class=HTMLResponse)
def admin_home(_: str = Depends(require_admin)):
    return HTMLResponse(ADMIN_HTML_HEAD + """
    <h2>Admin Dashboard</h2>
    <ul>
      <li><a href="/admin/pending">Pending Requests</a></li>
      <li><a href="/admin/codes">Access Codes</a></li>
    </ul>
    """)


@app.get("/admin/pending", response_class=HTMLResponse)
def admin_pending(_: str = Depends(require_admin)):
    items = list_pending()
    rows = "".join(
        f"<tr><td>{p['id']}</td><td>{p['email']}</td><td>{p.get('note','')}</td>"
        f"<td><form method='post' action='/admin/approve'><input type='hidden' name='id' value='{p['id']}'/>"
        f"<input name='label' placeholder='Label/note'/>"
        f"<input name='uses' type='number' value='1' min='1' style='width:4em'/>"
        f"<input name='ttl' type='number' value='48' min='0' style='width:4em'/>"
        f"<button>Approve → Create Code</button></form></td></tr>"
        for p in items
    )
    return HTMLResponse(ADMIN_HTML_HEAD + f"""
    <h2>Pending Requests</h2>
    <table border="1" cellpadding="6">
      <tr><th>ID</th><th>Email</th><th>Note</th><th>Action</th></tr>
      {rows or "<tr><td colspan=4>No pending</td></tr>"}
    </table>
    <p><a href="/admin">Back</a></p>
    """)


@app.post("/admin/approve", response_class=HTMLResponse)
async def admin_approve(id: str = Form(...), label: str = Form(""),
                        uses: int = Form(1), ttl: int = Form(48),
                        _: str = Depends(require_admin)):
    code = add_code(label or f"for:{id}", uses=uses, ttl_hours=ttl)
    remove_pending(id)
    return HTMLResponse(ADMIN_HTML_HEAD + f"""
    <h2>Approved</h2>
    <p>Share this access code with the user:</p>
    <pre style="font-size:1.2em">{code}</pre>
    <p><a href="/admin/pending">Back to pending</a></p>
    """)


@app.get("/admin/codes", response_class=HTMLResponse)
def admin_codes(_: str = Depends(require_admin)):
    codes = list_codes()
    rows = "".join(
        f"<tr><td>{c['label']}</td><td><code>{c['code']}</code></td>"
        f"<td>{c['uses']}</td><td>{'never' if c['exp']==0 else time.strftime('%Y-%m-%d %H:%M', time.localtime(c['exp']))}</td></tr>"
        for c in codes
    )
    return HTMLResponse(ADMIN_HTML_HEAD + f"""
    <h2>Active Codes</h2>
    <table border="1" cellpadding="6">
      <tr><th>Label</th><th>Code</th><th>Uses Left</th><th>Expires</th></tr>
      {rows or "<tr><td colspan=4>No codes yet</td></tr>"}
    </table>
    <p><a href="/admin">Back</a></p>
    """)


# ===============================================================
# Example main route (protected)
# ===============================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
    <h1>Welcome to Sora2Detector</h1>
    <p>Your access has been approved ✅</p>
    <p>You can now use the main app or API endpoints.</p>
    """)


# ===============================================================
# Run locally (Windows safe)
# ===============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
