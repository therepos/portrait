"""Talks to the GPU helper container (app.broker) to pause services during a restore."""
import asyncio
import os
import time

import httpx

BROKER_URL = (os.environ.get("GPU_HELPER_URL") or "").rstrip("/")
NEED_GB = float(os.environ.get("RESTORE_VRAM_GB", "16"))  # free GPU memory a restore needs


class PauseError(RuntimeError):
    pass


def enabled():
    return bool(BROKER_URL)


async def active_units():
    """Pausable services that are running now."""
    if not enabled():
        return []
    try:
        async with httpx.AsyncClient(base_url=BROKER_URL, timeout=15) as http:
            units = (await http.get("/units")).json()
    except (httpx.HTTPError, ValueError):
        return []
    return [u["unit"] for u in units if u.get("state") in ("active", "activating", "reloading")]


async def pause():
    try:
        async with httpx.AsyncClient(base_url=BROKER_URL, timeout=180) as http:
            r = await http.post("/pause")
    except httpx.HTTPError as e:
        raise PauseError(f"Cannot reach the GPU helper ({e.__class__.__name__}).")
    if r.status_code != 200:
        raise PauseError(r.json().get("detail", "The GPU helper could not pause the service."))
    return r.json()["lease"]


async def resume(lease):
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(base_url=BROKER_URL, timeout=180) as http:
                (await http.post("/resume", json={"lease": lease})).raise_for_status()
                return
        except httpx.HTTPError:
            await asyncio.sleep(3)
    # if this never gets through, the helper's lease expiry starts the service anyway


async def wait_for_vram(free_bytes, timeout=90):
    """Wait until the GPU has enough free memory, as reported by free_bytes()."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        free = await free_bytes()
        if free is None or free >= NEED_GB * 2**30:
            return
        await asyncio.sleep(2)
