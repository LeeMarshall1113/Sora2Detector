import os
import time
import pathlib
import json
from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

# ==========================================================
# Configuration
# ==========================================================
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "password")
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", 24))
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "./data")).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
PENDING_FILE = DATA_DIR / "pending.json"
CODES_FILE = DATA_DIR / "codes.json"

BASE_DIR = pathlib.Path(__file__).parent.resolve()

# ==========================================================
# Utility Functions
# ==========================================================
def read_json(path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_codes():
    return read_json(CODES_FILE)

def save_codes(codes):
    write_json(CODES_FILE, codes)

def load_pending():
    return read_json(PENDING_FILE)

def save_pending(reqs):
    write_json(PENDING_FILE, reqs)

def consume_code(code):
    codes = load_codes()
    now = int(time.time())
    info = codes.get(code)
    if not info:
        return False
    if info["expires"] < now:
        del codes[code]
        save_codes(codes)
        return False
    if info["uses"] <= 0:
        del codes[code]
        save_codes(codes)
        return False
    info["uses"] -= 1
    codes[code] = info
    save_codes(codes)
    return True

# ==========================================================
# FastAPI app setup
# ==========================================================
app = FastAPI(title="Sora2Detector Access System")

app.add_middleware(ProxyHeadersMiddleware)

# ==========================================================
# Access Control Middleware
# ==========================================================
@app.middleware("http")
async def approval_gate(request: Request, call_next):
    allowed_paths = {"/access", "/access/login", "/healthz", "/admin", "/admin/codes", "/admin/requests", "/static"}
    if any(request.url.path.startswith(p) for p in allowed_paths):
        return await call_next(request)

    sess = request.scope.get("session") or {}
    approved = sess.get("approved", False)
    exp = sess.get("approved_exp", 0)
    if approved and exp > time.time():
        return await call_next(request)
    return RedirectResponse("/access")

# ==========================================================
# Routes
# ==========================================================
@app.get("/healthz")
def healthz():
    return {"ok": True}

# -------- Access page --------
@app.get("/access", response_class=HTMLResponse)
def access_page():
    return """
    <h2>Enter Access Code</h2>
    <form method="post" action="/access/login">
        <input type="text" name="code" placeholder="Access code" required>
        <button type="submit">Submit</button>
    </form>
    <hr>
    <p>Don't have a code?</p>
    <a href="/access/request">
        <button type="button">Request Access</button>
    </a>
    """


@app.post("/access/login")
async def access_login(request: Request, code: str = Form(...)):
    code = code.strip()
    if consume_code(code):
        exp = int(time.time()) + SESSION_TTL_HOURS * 3600
        request.session["approved"] = True
        request.session["approved_exp"] = exp
        return RedirectResponse("/", status_code=303)
    return HTMLResponse("<p>Invalid or expired code.</p><a href='/access'>Back</a>", status_code=401)

# -------- Admin routes --------
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
security = HTTPBasic()

def require_admin(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, ADMIN_USER)
    correct_pass = secrets.compare_digest(credentials.password, ADMIN_PASS)
    if not (correct_user and correct_pass):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

@app.get("/admin", dependencies=[Depends(require_admin)], response_class=HTMLResponse)
def admin_panel():
    reqs = load_pending()
    html = "<h2>Pending Requests</h2>"
    if not reqs:
        html += "<p>No pending requests.</p>"
    for rid, info in reqs.items():
        html += f"""
        <div style='margin-bottom:1em;'>
            <b>{rid}</b>: {info['email']} - {info['note']}<br>
            <form method='post' action='/admin/approve'>
                <input type='hidden' name='rid' value='{rid}'>
                <input type='text' name='label' placeholder='Label (optional)'>
                <input type='number' name='uses' value='1' min='1'>
                <input type='number' name='ttl' value='48' min='1'> (hours)
                <button type='submit'>Approve</button>
            </form>
        </div>
        """
    return HTMLResponse(html)

@app.post("/admin/approve", dependencies=[Depends(require_admin)])
def approve_request(rid: str = Form(...), label: str = Form(""), uses: int = Form(1), ttl: int = Form(48)):
    reqs = load_pending()
    if rid not in reqs:
        return HTMLResponse("Request not found", status_code=404)
    code = os.urandom(8).hex()
    codes = load_codes()
    codes[code] = {
        "email": reqs[rid]["email"],
        "note": reqs[rid]["note"],
        "label": label,
        "uses": uses,
        "expires": int(time.time()) + ttl * 3600
    }
    save_codes(codes)
    del reqs[rid]
    save_pending(reqs)
    return HTMLResponse(f"<p>Code created: <b>{code}</b></p><a href='/admin'>Back</a>")

@app.get("/admin/codes", dependencies=[Depends(require_admin)], response_class=HTMLResponse)
def list_codes():
    codes = load_codes()
    now = int(time.time())
    html = "<h2>Active Codes</h2>"
    for code, info in codes.items():
        status = "✅ active" if info["expires"] > now else "❌ expired"
        html += f"<p><b>{code}</b> ({status}) - {info['email']} [{info['uses']} uses left]</p>"
    return HTMLResponse(html)

# -------- Request form (optional) --------
@app.get("/access/request", response_class=HTMLResponse)
def request_access_form():
    return """
    <h2>Request Access</h2>
    <form method="post" action="/access/request">
        <input type="email" name="email" placeholder="Your email" required><br>
        <textarea name="note" placeholder="Why do you need access?" required></textarea><br>
        <button type="submit">Submit</button>
    </form>
    """

@app.post("/access/request")
async def request_access(email: str = Form(...), note: str = Form(...)):
    reqs = load_pending()
    rid = os.urandom(4).hex()
    reqs[rid] = {"email": email, "note": note, "time": int(time.time())}
    save_pending(reqs)
    return HTMLResponse("<p>Request submitted. Await approval.</p><a href='/access'>Back</a>")

# -------- Main content page --------
@app.get("/")
def main_page():
    p = BASE_DIR / "static" / "index.html"
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("<p>static/index.html not found.</p>", status_code=500)

# -------- Serve static files (optional) --------
BASE_DIR = pathlib.Path(__file__).parent.resolve()
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# -------- Session Middleware (last added runs first) --------
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="sora_session",
    same_site="lax",
    https_only=True,
    max_age=SESSION_TTL_HOURS * 3600,
)
