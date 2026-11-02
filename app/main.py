"""Web app: upload a photo once, then re-render the card as the user tweaks it."""
import os
import re
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.concurrency import run_in_threadpool

from . import card

STATIC = Path(__file__).resolve().parent / "static"
SAVE_DIR = Path(os.environ["SAVE_DIR"]) if os.environ.get("SAVE_DIR") else None
MAX_PHOTOS = int(os.environ.get("MAX_PHOTOS", "20"))   # photos kept in memory
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "40"))

app = FastAPI(title="Portrait card maker")
_photos: "OrderedDict[str, tuple]" = OrderedDict()   # id -> (photo, face)


def _get(photo_id):
    if photo_id not in _photos:
        raise HTTPException(404, "Photo expired, please upload it again.")
    _photos.move_to_end(photo_id)
    return _photos[photo_id]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {"can_save": SAVE_DIR is not None}


@app.post("/api/photo")
async def upload(photo: UploadFile = File(...)):
    data = await photo.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Photo is over {MAX_UPLOAD_MB} MB.")

    def prepare():
        im = card.load_photo(data)
        return im, card.find_face(im)

    try:
        im, face = await run_in_threadpool(prepare)
    except card.NoFaceError as e:
        raise HTTPException(422, str(e))
    except Exception:
        raise HTTPException(400, "Could not read that file as a photo.")
    photo_id = uuid.uuid4().hex
    _photos[photo_id] = (im, face)
    while len(_photos) > MAX_PHOTOS:
        _photos.popitem(last=False)
    return {"id": photo_id}


def _render(photo_id, date, name, zoom, dx, dy):
    im, face = _get(photo_id)
    zoom = min(max(zoom, 0.5), 2.5)
    return card.to_png(card.render(im, face, date, name, zoom, dx, dy))


@app.post("/api/card")
async def make_card(id: str = Form(...), date: str = Form(""), name: str = Form(""),
                    zoom: float = Form(1.0), dx: int = Form(0), dy: int = Form(0)):
    png = await run_in_threadpool(_render, id, date, name, zoom, dx, dy)
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


def _safe_stem(name):
    first = (name.split() or ["card"])[0]
    stem = re.sub(r"[^\w-]", "", first.lower())
    return stem or "card"


@app.post("/api/save")
async def save(id: str = Form(...), date: str = Form(""), name: str = Form(""),
               zoom: float = Form(1.0), dx: int = Form(0), dy: int = Form(0)):
    if SAVE_DIR is None:
        raise HTTPException(400, "Saving to a folder is not set up on this server.")
    png = await run_in_threadpool(_render, id, date, name, zoom, dx, dy)
    stem = _safe_stem(name)
    out, n = SAVE_DIR / f"{stem}.png", 2
    while out.exists():  # never overwrite
        out, n = SAVE_DIR / f"{stem}-{n}.png", n + 1
    try:
        out.write_bytes(png)
    except OSError:
        raise HTTPException(500, f"Cannot write to the save folder ({SAVE_DIR}). Check it exists and is writable by user 1000.")
    return {"saved": out.name}
