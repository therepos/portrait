"""Restore queue: runs restores one at a time, keeps their progress and results on disk.

A paused GPU service stays paused across back-to-back restores and is handed
back when the queue runs empty.
"""
import asyncio
import json
import logging
import time

from PIL import Image

from . import comfy, gpu, store

log = logging.getLogger("portrait.jobs")
STATS = store.DATA_DIR / "stats.json"

_wake = asyncio.Event()
_running: dict = {}      # item id -> asyncio task of the restore in progress
_lease = None            # GPU pause held while the queue is busy
_stats = {"load": 60.0, "step": 7.0, "finish": 15.0, "pause": 20.0}  # seconds, refined as restores run


def _load_stats():
    try:
        _stats.update(json.loads(STATS.read_text()))
    except (OSError, ValueError):
        pass


def _learn(key, seconds):
    _stats[key] = round(0.6 * _stats[key] + 0.4 * seconds, 2)
    try:
        STATS.parent.mkdir(parents=True, exist_ok=True)
        STATS.write_text(json.dumps(_stats))
    except OSError:
        pass


def estimate(item):
    """Rough seconds left for one item."""
    p = item.get("progress") or {}
    steps = p.get("steps") or 40
    phase = item["state"] if item["state"] in ("queued", "pausing") else p.get("phase", "queued")
    since = time.time() - p.get("phase_started", time.time())
    if phase == "steps":
        rate = p.get("step_rate") or _stats["step"]
        return (steps - p.get("step", 0)) * rate + _stats["finish"]
    if phase == "finishing":
        return max(3, _stats["finish"] - since)
    if phase == "loading":
        return max(5, _stats["load"] - since) + steps * _stats["step"] + _stats["finish"]
    left = _stats["load"] + steps * _stats["step"] + _stats["finish"]
    if phase == "pausing":
        left += max(3, _stats["pause"] - since)
    return left


def with_eta(items):
    """Add eta_seconds to unfinished items, counting the queue ahead of each."""
    queue = [i for i in store.all_items() if i["state"] not in store.FINAL]
    ahead, out = 0.0, {}
    for i in queue:
        own = estimate(i)
        out[i["id"]] = round(ahead + own)
        ahead += own
    for i in items:
        if i["id"] in out:
            i["eta_seconds"] = out[i["id"]]
    return items


def submit(item):
    item["state"] = "queued"
    store.save(item)
    _wake.set()


async def cancel(item_id):
    task = _running.get(item_id)
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    else:
        try:
            item = store.load(item_id)
        except KeyError:
            return
        if item["state"] not in store.FINAL:
            item["state"] = "cancelled"
            store.save(item)


def _set(item, **fields):
    item.update(fields)
    store.save(item)


async def _ensure_gpu(item):
    """Make room on the GPU if needed; returns False if the item can't run."""
    global _lease
    free = await comfy.vram_free()
    if free is None or free >= gpu.NEED_GB * 2**30:
        return True
    await comfy.free()  # ComfyUI may just be holding models from an earlier job
    await asyncio.sleep(2)
    free = await comfy.vram_free() or free
    if free >= gpu.NEED_GB * 2**30 or _lease:
        return True  # with a pause already held, there is nothing more to free
    if not gpu.enabled() or not await gpu.active_units():
        _set(item, state="error", error=f"Not enough free GPU memory ({free / 2**30:.1f} GB free, "
                                        f"about {gpu.NEED_GB:g} GB needed).")
        return False
    if not item.get("allow_pause"):
        _set(item, state="error", error="The graphics card is busy and pausing it was not allowed.")
        return False
    _set(item, state="pausing", progress={"phase": "pausing", "phase_started": time.time()})
    started = time.monotonic()
    pausing = asyncio.ensure_future(gpu.pause())
    try:
        _lease = await asyncio.shield(pausing)
    except asyncio.CancelledError:  # cancelled mid-pause: still collect the lease to give it back
        try:
            _lease = await pausing
        except gpu.PauseError:
            pass
        raise
    await gpu.wait_for_vram(comfy.vram_free)
    _learn("pause", time.monotonic() - started)
    return True


async def _restore(item):
    source = store.get_image(item, "source.png")
    target = tuple(item["target_size"])
    size = comfy.model_size(*target)
    # the model's shape at target width; the extra rows at the bottom are trimmed
    fit = (target[0], round(target[0] * size[1] / size[0]))
    send = size if source.width >= size[0] else source.size
    picture = source.resize(send, Image.LANCZOS) if send != source.size else source

    if not await _ensure_gpu(item):
        return
    _set(item, state="running", progress={"phase": "queued", "phase_started": time.time()})
    marks = {"loading": None, "steps": None, "finishing": None, "last_step": (0, None)}

    def on_progress(p):
        now = time.time()
        prog = item.get("progress") or {}
        if p["phase"] != prog.get("phase"):
            marks[p["phase"]] = now
            prog = {"phase_started": now}
        prog.update(p)
        step, at = marks["last_step"]
        if p["phase"] == "steps" and p.get("step", 0) > step:
            if at:
                rate = (now - at) / (p["step"] - step)
                prog["step_rate"] = round(0.7 * prog.get("step_rate", rate) + 0.3 * rate, 2)
            marks["last_step"] = (p["step"], now)
        _set(item, progress=prog)

    started = time.time()
    result = await comfy.restore(picture, size, on_progress)
    result = result.resize(fit, Image.LANCZOS).crop((0, 0, *target))
    store.put_image(item, "result.png", result)
    store.put_thumb(item, result)
    if marks["loading"] and marks["steps"]:
        _learn("load", marks["steps"] - marks["loading"])
    if item["progress"].get("step_rate"):
        _learn("step", item["progress"]["step_rate"])
    if marks["finishing"]:
        _learn("finish", time.time() - marks["finishing"])
    _set(item, state="done", seconds=round(time.time() - started), progress=None)


async def _run(item):
    try:
        await _restore(item)
    except asyncio.CancelledError:
        _set(item, state="cancelled", progress=None)
    except (comfy.ComfyError, gpu.PauseError) as e:
        _set(item, state="error", error=str(e), progress=None)
    except Exception as e:
        log.exception("restore %s failed", item["id"])
        _set(item, state="error", error=f"Restore failed ({e.__class__.__name__}).", progress=None)


async def _release_gpu():
    global _lease
    if _lease:
        await comfy.free()
        await gpu.resume(_lease)
        _lease = None


async def worker():
    _load_stats()
    for item in store.all_items():  # restart: run anything that was waiting or interrupted
        if item["state"] not in store.FINAL:
            submit(item)
    while True:
        queue = [i for i in store.all_items() if i["state"] == "queued" and i["kind"] == "restore"]
        if not queue:
            await _release_gpu()
            _wake.clear()
            await _wake.wait()
            continue
        item = queue[0]
        task = asyncio.create_task(_run(item))
        _running[item["id"]] = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:  # app shutting down: put it back in the queue for next start
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            try:
                _set(store.load(item["id"]), state="queued", progress=None)
            except KeyError:
                pass
            await _release_gpu()
            raise
        finally:
            _running.pop(item["id"], None)


async def cleanup():
    while True:
        for item in store.expired():
            log.info("deleting expired item %s", item["id"])
            store.delete(item["id"])
        await asyncio.sleep(3600)
