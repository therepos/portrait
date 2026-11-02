"""GPU helper: pauses listed systemd services while the app restores a photo.

Runs as its own container (same image, `python -m app.broker`) with root and
the host's D-Bus socket, on a network only the app can reach. It accepts two
requests, pause and resume, and only for the units in PAUSE_UNITS.

Each pause is a lease. Units are started again when the last lease is
released, when a lease expires (the app died mid-restore), or when this
helper shuts down. A unit that was already stopped is never started.
"""
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import systemd

UNITS = [u.strip() for u in os.environ.get("PAUSE_UNITS", "").split(",") if u.strip()]
LEASE_SECONDS = int(os.environ.get("PAUSE_MAX_MINUTES", "30")) * 60
STOP_WAIT = 90

log = logging.getLogger("portrait.broker")
_lock = asyncio.Lock()
_leases: dict = {}      # lease id -> expiry (monotonic)
_stopped: list = []     # units we stopped and owe a start


async def _state(unit):
    return await asyncio.to_thread(systemd.state, unit)


async def _start_all():
    """Start every unit we stopped. Called with _lock held."""
    for unit in list(_stopped):
        for attempt in range(3):
            try:
                await asyncio.to_thread(systemd.start, unit)
                log.info("started %s again", unit)
                _stopped.remove(unit)
                break
            except Exception as e:
                log.error("could not start %s (attempt %d): %r", unit, attempt + 1, e)
                await asyncio.sleep(5)


async def expire_leases():
    """Drop leases whose restore never came back; start units if none remain."""
    async with _lock:
        now = time.monotonic()
        for lease in [k for k, exp in _leases.items() if exp < now]:
            log.warning("lease %s expired; releasing it", lease)
            del _leases[lease]
        if not _leases and _stopped:
            await _start_all()


async def _watchdog():
    while True:
        await asyncio.sleep(10)
        await expire_leases()


@asynccontextmanager
async def lifespan(app):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log.info("may pause: %s", ", ".join(UNITS) or "nothing (PAUSE_UNITS is empty)")
    task = asyncio.create_task(_watchdog())
    yield
    task.cancel()
    async with _lock:  # never leave a service stopped
        _leases.clear()
        await _start_all()


app = FastAPI(title="portrait GPU helper", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/units")
async def units():
    out = []
    for unit in UNITS:
        try:
            out.append({"unit": unit, "state": await _state(unit), "paused_by_us": unit in _stopped})
        except Exception as e:
            out.append({"unit": unit, "state": "unknown", "error": e.__class__.__name__})
    return out


@app.post("/pause")
async def pause():
    if not UNITS:
        raise HTTPException(400, "No services are listed in PAUSE_UNITS.")
    async with _lock:
        lease = uuid.uuid4().hex
        try:
            for unit in UNITS:
                if unit in _stopped:
                    continue  # already paused for another restore
                if await _state(unit) not in ("active", "activating", "reloading"):
                    continue  # stopped by someone else: leave it that way
                log.info("stopping %s to free the GPU", unit)
                await asyncio.to_thread(systemd.stop, unit)
                _stopped.append(unit)
                deadline = time.monotonic() + STOP_WAIT
                while await _state(unit) not in ("inactive", "failed"):
                    if time.monotonic() > deadline:
                        raise RuntimeError(f"{unit} did not stop within {STOP_WAIT} s")
                    await asyncio.sleep(1)
        except Exception as e:
            if not _leases:
                await _start_all()
            raise HTTPException(500, f"Could not pause: {e}")
        _leases[lease] = time.monotonic() + LEASE_SECONDS
        return {"lease": lease, "paused": list(_stopped)}


class Release(BaseModel):
    lease: str


@app.post("/resume")
async def resume(body: Release):
    async with _lock:
        _leases.pop(body.lease, None)
        if not _leases:
            await _start_all()
    return {"paused": list(_stopped)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "9000")))
