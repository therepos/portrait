"""Web app: upload a photo once, then re-render the card as the user tweaks it."""
import asyncio
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response

from . import card, comfy, gpu

STATIC = Path(__file__).resolve().parent / "static"
SAVE_DIR = Path(os.environ["SAVE_DIR"]) if os.environ.get("SAVE_DIR") else None
MAX_PHOTOS = int(os.environ.get("MAX_PHOTOS", "20"))   # photos kept in memory
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "40"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app):
    yield
    # shutting down (e.g. an update): cancel restores so their GPU pause is handed back now
    tasks = [j["task"] for j in _jobs.values() if j.get("task") and not j["task"].done()]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Portrait card maker", lifespan=lifespan)
_photos: "OrderedDict[str, dict]" = OrderedDict()  # id -> photo, face, restored
_jobs: "OrderedDict[str, dict]" = OrderedDict()    # restore jobs


def _get(photo_id):
    if photo_id not in _photos:
        raise HTTPException(404, "Photo expired, please upload it again.")
    _photos.move_to_end(photo_id)
    return _photos[photo_id]


def _framing(zoom, dx, dy):
    return (round(min(max(zoom, 0.5), 2.5), 3), dx, dy)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/api/config")
def config():
    return {"can_save": SAVE_DIR is not None, "can_restore": comfy.enabled(), "can_pause": gpu.enabled()}


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
    _photos[photo_id] = {"photo": im, "face": face, "restored": None}
    while len(_photos) > MAX_PHOTOS:
        _photos.popitem(last=False)
    return {"id": photo_id}


def _render(photo_id, date, name, zoom, dx, dy, restored):
    entry = _get(photo_id)
    framing = _framing(zoom, dx, dy)
    if restored:
        r = entry["restored"]
        if not r or r[0] != framing:
            raise HTTPException(409, "The framing changed since the restore. Restore again.")
        return card.to_png(card.finish(r[1], date, name))
    return card.to_png(card.render(entry["photo"], entry["face"], date, name, *framing))


@app.post("/api/card")
async def make_card(id: str = Form(...), date: str = Form(""), name: str = Form(""),
                    zoom: float = Form(1.0), dx: int = Form(0), dy: int = Form(0),
                    restored: bool = Form(False)):
    png = await run_in_threadpool(_render, id, date, name, zoom, dx, dy, restored)
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


async def _run_restore(job, entry, framing, pause):
    t = card.TEMPLATE
    size = comfy.model_size(t.width, t.height)
    # the model's shape at card width; a few extra rows at the bottom get trimmed
    card_size = (t.width, round(t.width * size[1] / size[0]))
    box = card.frame(entry["photo"], entry["face"], *framing, extra_h=card_size[1] - t.height)
    # send the photo's own pixels; only shrink when there are more than needed
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    send = size if box_w >= size[0] else (round(box_w), round(box_h))
    picture = card.crop(entry["photo"], box, send)

    def status(s):
        job["state"] = s

    lease = None
    try:
        if pause:
            job["state"] = "pausing"
            pausing = asyncio.ensure_future(gpu.pause())
            try:
                lease = await asyncio.shield(pausing)
            except asyncio.CancelledError:  # cancelled mid-pause: still collect the lease to give it back
                try:
                    lease = await pausing
                except gpu.PauseError:
                    pass
                raise
            await gpu.wait_for_vram(comfy.vram_free)
        result = await comfy.restore(picture, size, status)
        entry["restored"] = (framing, result.resize(card_size, card.Image.LANCZOS))
        job["state"] = "done"
    except asyncio.CancelledError:
        job["state"] = "cancelled"
    except (comfy.ComfyError, gpu.PauseError) as e:
        job.update(state="error", error=str(e))
    except Exception as e:
        job.update(state="error", error=f"Restore failed ({e.__class__.__name__}).")
    finally:
        if lease:  # hand the GPU back whatever happened
            await comfy.free()
            await gpu.resume(lease)
    job["finished"] = time.monotonic()


@app.post("/api/restore")
async def start_restore(id: str = Form(...), zoom: float = Form(1.0), dx: int = Form(0), dy: int = Form(0),
                        allow_pause: bool = Form(False)):
    if not comfy.enabled():
        raise HTTPException(400, "AI restore is not set up on this server.")
    entry = _get(id)
    framing = _framing(zoom, dx, dy)
    job_id = uuid.uuid4().hex
    job = {"state": "queued", "started": time.monotonic()}
    if entry["restored"] and entry["restored"][0] == framing:
        job["state"] = "done"
    else:
        pause = False
        free = await comfy.vram_free()
        if free is not None and free < gpu.NEED_GB * 2**30:
            await comfy.free()  # ComfyUI may just be holding models from an earlier job
            await asyncio.sleep(2)
            free = await comfy.vram_free() or free
        if free is not None and free < gpu.NEED_GB * 2**30:
            units = await gpu.active_units()
            need, have = gpu.NEED_GB, round(free / 2**30, 1)
            if not units:
                raise HTTPException(409, f"Not enough free GPU memory ({have} GB free, about {need:g} GB needed).")
            if not allow_pause:
                return JSONResponse({"needs_pause": True, "units": units, "free_gb": have, "need_gb": need}, 409)
            pause = True
        job["task"] = asyncio.create_task(_run_restore(job, entry, framing, pause))
    _jobs[job_id] = job
    while len(_jobs) > 50:
        _jobs.popitem(last=False)
    return {"job": job_id}


@app.post("/api/restore/{job_id}/cancel")
async def cancel_restore(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown restore job.")
    task = job.get("task")
    if task and not task.done():
        task.cancel()
        try:
            await task  # wait until ComfyUI has dropped it and the GPU is handed back
        except asyncio.CancelledError:
            pass
    return {"state": job["state"]}


@app.get("/api/restore/{job_id}")
def restore_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown restore job.")
    end = job.get("finished", time.monotonic())
    return {"state": job["state"], "error": job.get("error"), "seconds": round(end - job["started"])}


def _safe_stem(name):
    first = (name.split() or ["card"])[0]
    stem = re.sub(r"[^\w-]", "", first.lower())
    return stem or "card"


@app.post("/api/save")
async def save(id: str = Form(...), date: str = Form(""), name: str = Form(""),
               zoom: float = Form(1.0), dx: int = Form(0), dy: int = Form(0),
               restored: bool = Form(False)):
    if SAVE_DIR is None:
        raise HTTPException(400, "Saving to a folder is not set up on this server.")
    png = await run_in_threadpool(_render, id, date, name, zoom, dx, dy, restored)
    stem = _safe_stem(name)
    out, n = SAVE_DIR / f"{stem}.png", 2
    while out.exists():  # never overwrite
        out, n = SAVE_DIR / f"{stem}-{n}.png", n + 1
    try:
        out.write_bytes(png)
    except OSError:
        raise HTTPException(500, f"Cannot write to the save folder ({SAVE_DIR}). Check it exists and is writable by user 1000.")
    return {"saved": out.name}
