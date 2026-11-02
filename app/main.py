"""Web app: restore photos with AI and make portrait cards; history per browser."""
import asyncio
import hashlib
import logging
import os
import re
import secrets
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response

from . import auth, card, comfy, gpu, jobs, store

STATIC = Path(__file__).resolve().parent / "static"
SAVE_DIR = Path(os.environ["SAVE_DIR"]) if os.environ.get("SAVE_DIR") else None
MAX_PHOTOS = int(os.environ.get("MAX_PHOTOS", "20"))   # portrait photos kept in memory
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "40"))
RESTORE_MAX_SIDE = 2048  # restored photos are at most this big on the long side
MAX_QUEUED = int(os.environ.get("MAX_QUEUED", "5"))  # unfinished restores one person may have
COOKIE = "portrait_owner"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app):
    tasks = [asyncio.create_task(jobs.worker()), asyncio.create_task(jobs.cleanup())]
    yield
    for t in tasks:  # the worker hands a paused GPU service back as it stops
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Portrait card maker", lifespan=lifespan)
_photos: "OrderedDict[str, dict]" = OrderedDict()  # portrait session id -> photo, face


PUBLIC = {"/login", "/api/login", "/healthz"}


def _user_owner(username):
    return "u" + hashlib.sha256(username.encode()).hexdigest()[:32]


@app.middleware("http")
async def owner_and_login(request: Request, call_next):
    """Who owns the history: the signed-in account, or else this browser (random id in a cookie)."""
    browser = request.cookies.get(COOKIE)
    if not browser or not browser.isalnum():
        browser = secrets.token_hex(16)
        request.state.new_browser = browser
    request.state.browser = browser
    request.state.owner = browser
    request.state.user = None
    if auth.setup_needed() and request.url.path != "/healthz":
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Sign-in is not set up on this server."}, 503)
        return FileResponse(STATIC / "setup.html", status_code=503)
    if auth.enabled():
        user = auth.session_user(request.cookies.get(auth.COOKIE))
        if user:
            request.state.user = user
            request.state.owner = _user_owner(user)
        elif request.url.path not in PUBLIC:
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": "Please sign in."}, 401)
            return RedirectResponse("login", 303)
    response = await call_next(request)
    if getattr(request.state, "new_browser", None):
        response.set_cookie(COOKIE, browser, max_age=10 * 365 * 86400, httponly=True, samesite="lax")
    return response


def _https(request):
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


def _address(request):
    return (request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "?"))


@app.get("/login")
def login_page():
    if not auth.enabled():
        return RedirectResponse("./", 303)
    return FileResponse(STATIC / "login.html")


@app.post("/api/login")
async def login(request: Request, username: str = Form(""), password: str = Form("")):
    if not auth.enabled():
        return {"ok": True}
    ok = await auth.check(_address(request), username, password)
    if ok is None:
        raise HTTPException(429, "Too many wrong tries. Wait 15 minutes and try again.")
    if not ok:
        raise HTTPException(401, "Wrong username or password.")
    owner = _user_owner(username)
    for item in store.all_items():  # bring this browser's earlier history into the account
        if item["owner"] == request.state.browser:
            item["owner"] = owner
            store.save(item)
    value, max_age = auth.make_session(username)
    response = JSONResponse({"ok": True})
    response.set_cookie(auth.COOKIE, value, max_age=max_age, httponly=True, samesite="lax", secure=_https(request))
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.COOKIE)
    return response


def _owner(request):
    return request.state.owner


def _item(request, item_id):
    try:
        return store.owned(item_id, _owner(request))
    except KeyError:
        raise HTTPException(404, "Not found. It may have expired.")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/config")
def config(request: Request):
    return {"can_save": SAVE_DIR is not None, "can_restore": comfy.enabled(), "can_pause": gpu.enabled(),
            "retention_days": store.RETENTION_DAYS, "auth": auth.enabled(), "user": request.state.user}


async def _read_upload(photo: UploadFile):
    data = await photo.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Photo is over {MAX_UPLOAD_MB} MB.")
    try:
        return await run_in_threadpool(card.load_photo, data)
    except Exception:
        raise HTTPException(400, "Could not read that file as a photo.")


def _queue_limit(request):
    waiting = [i for i in store.for_owner(_owner(request)) if i["state"] in ("queued", "pausing", "running")]
    if len(waiting) >= MAX_QUEUED:
        raise HTTPException(429, f"You already have {len(waiting)} restores waiting. "
                                 "Add more when some have finished.")


async def _pause_check(allow_pause):
    """None if a restore can go ahead, else a 409 reply asking to pause a service."""
    if allow_pause or jobs._lease or not comfy.enabled():
        return None
    if any(i.get("allow_pause") for i in store.all_items() if i["state"] in ("queued", "pausing", "running")):
        return None  # a restore already allowed to pause is ahead; this one runs in the same batch
    free = await comfy.vram_free()
    if free is None or free >= gpu.NEED_GB * 2**30:
        return None
    units = await gpu.active_units()
    if not units:
        return None  # nothing to pause; the queue reports if it really doesn't fit
    return JSONResponse({"needs_pause": True, "units": units, "free_gb": round(free / 2**30, 1),
                         "need_gb": gpu.NEED_GB}, 409)


def _thumb_of_source(item, im):
    t = im.convert("RGB")
    t.thumbnail((store.THUMB, store.THUMB))
    t.save(store.path(item, "source_thumb.jpg"), "JPEG", quality=80)


# ---- history ---------------------------------------------------------------

def _public(item):
    keep = ("id", "kind", "created", "state", "error", "name", "seconds", "progress", "eta_seconds",
            "target_size", "origin")
    return {k: item[k] for k in keep if k in item}


@app.get("/api/items")
def list_items(request: Request):
    items = jobs.with_eta(store.for_owner(_owner(request)))
    return {"items": [_public(i) for i in items], "retention_days": store.RETENTION_DAYS}


@app.get("/api/items/{item_id}/thumb")
def item_thumb(request: Request, item_id: str):
    item = _item(request, item_id)
    for name in ("thumb.jpg", "source_thumb.jpg"):
        if store.path(item, name).exists():
            return FileResponse(store.path(item, name), headers={"Cache-Control": "no-cache"})
    raise HTTPException(404, "No picture yet.")


def _download_name(item):
    stem = re.sub(r"[^\w-]+", "-", (item.get("name") or item["kind"]).rsplit(".", 1)[0]).strip("-") or item["kind"]
    return f"{stem}.png" if item["kind"] == "card" else f"{stem}-restored.png"


@app.get("/api/items/{item_id}/file")
def item_file(request: Request, item_id: str):
    item = _item(request, item_id)
    name = "result.png" if item["kind"] == "restore" else "card.png"
    if item["state"] != "done" or not store.path(item, name).exists():
        raise HTTPException(409, "Not ready yet.")
    return FileResponse(store.path(item, name), media_type="image/png", filename=_download_name(item))


@app.post("/api/items/{item_id}/cancel")
async def cancel_item(request: Request, item_id: str):
    _item(request, item_id)
    await jobs.cancel(item_id)
    return _public(store.load(item_id))


@app.delete("/api/items/{item_id}")
async def delete_item(request: Request, item_id: str):
    _item(request, item_id)
    await jobs.cancel(item_id)
    store.delete(item_id)
    return {"deleted": item_id}


# ---- restore tab -----------------------------------------------------------

def _restore_target(w, h):
    """Restored size: the photo's shape and its own size, capped at RESTORE_MAX_SIDE
    but never smaller than what the model produces."""
    mw, mh = comfy.model_size(w, h)
    long_side = max(max(mw, mh), min(max(w, h), RESTORE_MAX_SIDE))
    s = long_side / max(w, h)
    return [round(w * s), round(h * s)]


@app.post("/api/restores")
async def start_restore(request: Request, photo: UploadFile = File(...), allow_pause: bool = Form(False)):
    if not comfy.enabled():
        raise HTTPException(400, "AI restore is not set up on this server.")
    _queue_limit(request)
    im = await _read_upload(photo)
    ask = await _pause_check(allow_pause)
    if ask:
        return ask
    item = store.create(_owner(request), "restore", publish=False, state="queued", name=photo.filename or "photo",
                        target_size=_restore_target(*im.size), allow_pause=allow_pause, origin="upload")
    await run_in_threadpool(store.put_image, item, "source.png", im)
    await run_in_threadpool(_thumb_of_source, item, im)
    jobs.submit(item)
    return _public(item)


# ---- portrait tab ----------------------------------------------------------

def _get(photo_id):
    if photo_id not in _photos:
        raise HTTPException(404, "Photo expired, please upload it again.")
    _photos.move_to_end(photo_id)
    return _photos[photo_id]


def _framing(zoom, dx, dy):
    return [round(min(max(zoom, 0.5), 2.5), 3), dx, dy]


def _session(im, name):
    try:
        face = card.find_face(im)
    except card.NoFaceError as e:
        raise HTTPException(422, str(e))
    photo_id = uuid.uuid4().hex
    _photos[photo_id] = {"photo": im, "face": face, "name": name}
    while len(_photos) > MAX_PHOTOS:
        _photos.popitem(last=False)
    return {"id": photo_id, "name": name}


@app.post("/api/photo")
async def upload(photo: UploadFile = File(...)):
    im = await _read_upload(photo)
    return await run_in_threadpool(_session, im, photo.filename or "photo")


@app.post("/api/photo/from-item")
async def photo_from_item(request: Request, item: str = Form(...)):
    it = _item(request, item)
    if it["kind"] != "restore" or it["state"] != "done":
        raise HTTPException(409, "That restore isn't finished.")
    im = await run_in_threadpool(store.get_image, it, "result.png")
    return await run_in_threadpool(_session, im, it.get("name") or "photo")


def _render(photo_id, date, name, zoom, dx, dy):
    entry = _get(photo_id)
    return card.render(entry["photo"], entry["face"], date, name, *_framing(zoom, dx, dy))


@app.post("/api/card")
async def make_card(id: str = Form(...), date: str = Form(""), name: str = Form(""),
                    zoom: float = Form(1.0), dx: int = Form(0), dy: int = Form(0), preview: bool = Form(False)):
    """Card preview. X-Dx/X-Dy report the shift actually used, after stopping at the photo's edges."""
    im, dx, dy = await run_in_threadpool(_render, id, date, name, zoom, dx, dy)
    data, kind = (card.to_jpeg(im), "image/jpeg") if preview else (card.to_png(im), "image/png")
    return Response(data, media_type=kind, headers={"Cache-Control": "no-store", "X-Dx": str(dx), "X-Dy": str(dy)})


def _safe_stem(name):
    first = (name.split() or ["card"])[0]
    stem = re.sub(r"[^\w-]", "", first.lower())
    return stem or "card"


async def _keep_card(request, png, name):
    item = store.create(_owner(request), "card", publish=False, name=f"{_safe_stem(name)}.png")
    store.put_bytes(item, "card.png", png)
    await run_in_threadpool(store.put_thumb, item, card.Image.open(store.path(item, "card.png")))
    store.save(item)
    return item


@app.post("/api/cards")
async def keep_card(request: Request, id: str = Form(...), date: str = Form(""), name: str = Form(""),
                    zoom: float = Form(1.0), dx: int = Form(0), dy: int = Form(0)):
    """Render the card, add it to the history and return its download link."""
    im, _, _ = await run_in_threadpool(_render, id, date, name, zoom, dx, dy)
    item = await _keep_card(request, card.to_png(im), name)
    return {"item": item["id"], "download": f"api/items/{item['id']}/file"}


@app.post("/api/save")
async def save(request: Request, id: str = Form(...), date: str = Form(""), name: str = Form(""),
               zoom: float = Form(1.0), dx: int = Form(0), dy: int = Form(0)):
    if SAVE_DIR is None:
        raise HTTPException(400, "Saving to a folder is not set up on this server.")
    im, _, _ = await run_in_threadpool(_render, id, date, name, zoom, dx, dy)
    png = card.to_png(im)
    stem = _safe_stem(name)
    out, n = SAVE_DIR / f"{stem}.png", 2
    while out.exists():  # never overwrite
        out, n = SAVE_DIR / f"{stem}-{n}.png", n + 1
    try:
        out.write_bytes(png)
    except OSError:
        raise HTTPException(500, f"Cannot write to the save folder ({SAVE_DIR}). Check it exists and is writable by user 1000.")
    await _keep_card(request, png, name)
    return {"saved": out.name}
