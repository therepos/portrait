"""Optional AI restore through a ComfyUI server.

The workflow is an API-format export ("Export (API)" in ComfyUI). The app only
changes three things in it: the image fed to the LoadImage node, the resize
step in front of the model (set to an exact size), and the sampler seed.
Everything else, prompt included, comes from the workflow file.
"""
import asyncio
import io
import json
import os
import random
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image

URL = (os.environ.get("COMFYUI_URL") or "").rstrip("/")
WORKFLOW = Path(os.environ.get("COMFYUI_WORKFLOW") or "/config/comfyui-workflow.json")
TIMEOUT = float(os.environ.get("COMFYUI_TIMEOUT", "900"))

# Resize nodes that pick their own output size; replaced with an exact resize
AUTO_RESIZE = {"FluxKontextImageScale", "ImageScaleToTotalPixels", "ImageScaleToMaxDimension"}
OUTPUT_NODES = {"SaveImage", "SaveImageAdvanced", "PreviewImage"}
SAMPLERS = {"KSampler", "KSamplerAdvanced", "RandomNoise"}
MULTIPLE = 16  # the model works in 16-pixel steps
MEGAPIXELS = float(os.environ.get("COMFYUI_MEGAPIXELS", "1.0"))


class ComfyError(RuntimeError):
    pass


def enabled():
    return bool(URL) and WORKFLOW.exists()


def model_size(w, h):
    """Size the model works at: about MEGAPIXELS, multiples of 16, as close as possible
    to the card's shape and never wider (the extra height is trimmed afterwards).
    Qwen Image Edit keeps the picture faithful near 1 MP and starts redrawing it when
    asked for much more."""
    target = MEGAPIXELS * 1_000_000
    best = None
    for mw in range(MULTIPLE * 32, MULTIPLE * 160, MULTIPLE):
        mh = -(-mw * h // w)            # at least the card's height for this width
        mh = -(-mh // MULTIPLE) * MULTIPLE
        if abs(mw * mh - target) > 0.1 * target:
            continue
        err = mh / mw - h / w            # extra height to trim, as a share
        if best is None or err < best[0]:
            best = (err, mw, mh)
    return best[1], best[2]


def _prepare(image_name, size):
    wf = json.loads(WORKFLOW.read_text())
    loaders = [k for k, n in wf.items() if n["class_type"] == "LoadImage"]
    if len(loaders) != 1:
        raise ComfyError("The workflow must have exactly one Load Image node.")
    wf[loaders[0]]["inputs"]["image"] = image_name
    for node in wf.values():
        if node["class_type"] in AUTO_RESIZE:
            src = node["inputs"].get("image")
            node["class_type"] = "ImageScale"
            node["inputs"] = {"image": src, "upscale_method": "lanczos",
                              "width": size[0], "height": size[1], "crop": "disabled"}
        elif node["class_type"] in SAMPLERS:
            key = "noise_seed" if "noise_seed" in node["inputs"] else "seed"
            if not isinstance(node["inputs"].get(key), list):
                node["inputs"][key] = random.randrange(2**50)
    outputs = [k for k, n in wf.items() if n["class_type"] in OUTPUT_NODES]
    if not outputs:
        raise ComfyError("The workflow has no Save Image node.")
    return wf, outputs


async def restore(picture: Image.Image, size, on_status=lambda s: None) -> Image.Image:
    """Send picture through the workflow; return the result at exactly `size`."""
    buf = io.BytesIO()
    picture.save(buf, "PNG")
    name = f"portrait_{uuid.uuid4().hex[:12]}.png"
    async with httpx.AsyncClient(base_url=URL, timeout=60) as http:
        try:
            r = await http.post("/upload/image", files={"image": (name, buf.getvalue(), "image/png")},
                                data={"overwrite": "true"})
            r.raise_for_status()
            uploaded = r.json()
            image_ref = f"{uploaded['subfolder']}/{uploaded['name']}" if uploaded.get("subfolder") else uploaded["name"]
            wf, outputs = _prepare(image_ref, size)
            r = await http.post("/prompt", json={"prompt": wf, "client_id": "portrait"})
        except httpx.HTTPError as e:
            raise ComfyError(f"Cannot reach ComfyUI ({e.__class__.__name__}).")
        if r.status_code != 200:
            detail = r.json().get("error", {}).get("message", r.text[:200]) if r.headers.get("content-type", "").startswith("application/json") else r.text[:200]
            raise ComfyError(f"ComfyUI rejected the workflow: {detail}")
        prompt_id = r.json()["prompt_id"]
        try:
            return await _wait(http, prompt_id, outputs, size, on_status)
        except (asyncio.CancelledError, ComfyError):
            await _cancel(http, prompt_id)
            raise


async def _cancel(http, prompt_id):
    """Drop the job from ComfyUI's queue, or stop it if it is already running."""
    try:
        await http.post("/queue", json={"delete": [prompt_id]})
        q = (await http.get("/queue")).json()
        if any(item[1] == prompt_id for item in q.get("queue_running", [])):
            await http.post("/interrupt", json={"prompt_id": prompt_id})
    except (httpx.HTTPError, ValueError):
        pass


async def _wait(http, prompt_id, outputs, size, on_status):
    start = time.monotonic()
    while time.monotonic() - start < TIMEOUT:
        await asyncio.sleep(2)
        hist = (await http.get(f"/history/{prompt_id}")).json().get(prompt_id)
        if not hist:
            q = (await http.get("/queue")).json()
            running = any(item[1] == prompt_id for item in q.get("queue_running", []))
            on_status("running" if running else "queued")
            continue
        status = hist.get("status", {})
        if status.get("status_str") == "error":
            msgs = [m[1].get("exception_message", "") for m in status.get("messages", []) if m[0] == "execution_error"]
            msg = msgs[0].strip() if msgs else "unknown error"
            if "out of memory" in msg.lower():
                raise ComfyError("ComfyUI ran out of GPU memory. Another program may be using the graphics card.")
            raise ComfyError("ComfyUI failed: " + msg.splitlines()[0][:200])
        for key in outputs:
            images = hist.get("outputs", {}).get(key, {}).get("images", [])
            if images:
                img = images[0]
                r = await http.get("/view", params={"filename": img["filename"],
                                                    "subfolder": img.get("subfolder", ""),
                                                    "type": img.get("type", "output")})
                r.raise_for_status()
                out = Image.open(io.BytesIO(r.content)).convert("RGB")
                return out if out.size == size else out.resize(size, Image.LANCZOS)
        if status.get("completed"):
            raise ComfyError("ComfyUI finished without producing an image.")
    raise ComfyError(f"ComfyUI took longer than {int(TIMEOUT)} s.")


async def free():
    """Ask ComfyUI to unload its models so other programs get the GPU back."""
    try:
        async with httpx.AsyncClient(base_url=URL, timeout=30) as http:
            await http.post("/free", json={"unload_models": True, "free_memory": True})
    except httpx.HTTPError:
        pass


async def vram_free():
    """Free GPU memory in bytes as ComfyUI sees it, or None if unknown."""
    try:
        async with httpx.AsyncClient(base_url=URL, timeout=10) as http:
            return (await http.get("/system_stats")).json()["devices"][0]["vram_free"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        return None
