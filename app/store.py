"""History items (restores and cards) kept on disk per browser.

Each item is a folder DATA_DIR/items/<id>/ holding item.json and its images.
Items belong to the browser that made them (a random id in a cookie) and are
deleted after RETENTION_DAYS.
"""
import json
import os
import shutil
import time
import uuid
from pathlib import Path

from PIL import Image

DATA_DIR = Path(os.environ.get("DATA_DIR") or "/tmp/portrait-data")
RETENTION_DAYS = float(os.environ.get("RETENTION_DAYS", "7"))
ITEMS = DATA_DIR / "items"
THUMB = 360  # longest side of history thumbnails

FINAL = ("done", "error", "cancelled")


def _dir(item_id):
    if not item_id.isalnum():
        raise KeyError(item_id)
    return ITEMS / item_id


def save(item):
    d = _dir(item["id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "item.json.tmp"
    tmp.write_text(json.dumps(item))
    tmp.replace(d / "item.json")


def create(owner, kind, publish=True, **fields):
    """New item. With publish=False it isn't listed until save() is called,
    so its files can be written first."""
    item = {"id": uuid.uuid4().hex, "owner": owner, "kind": kind, "created": time.time(),
            "state": "done", **fields}
    if publish:
        save(item)
    return item


def load(item_id):
    try:
        return json.loads((_dir(item_id) / "item.json").read_text())
    except (OSError, ValueError):
        raise KeyError(item_id)


def owned(item_id, owner):
    item = load(item_id)
    if item["owner"] != owner:
        raise KeyError(item_id)
    return item


def all_items():
    if not ITEMS.exists():
        return []
    out = []
    for d in ITEMS.iterdir():
        try:
            out.append(json.loads((d / "item.json").read_text()))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda i: i["created"])


def for_owner(owner):
    return [i for i in reversed(all_items()) if i["owner"] == owner]


def path(item, name):
    return _dir(item["id"]) / name


def put_image(item, name, image: Image.Image):
    p = path(item, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    image.save(p, "PNG", compress_level=6)
    return name


def put_bytes(item, name, data: bytes):
    p = path(item, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return name


def put_thumb(item, image: Image.Image):
    t = image.convert("RGB")
    t.thumbnail((THUMB, THUMB), Image.LANCZOS)
    p = path(item, "thumb.jpg")
    p.parent.mkdir(parents=True, exist_ok=True)
    t.save(p, "JPEG", quality=85)


def get_image(item, name) -> Image.Image:
    return Image.open(path(item, name)).convert("RGB")


def delete(item_id):
    shutil.rmtree(_dir(item_id), ignore_errors=True)


def expired():
    """Finished items older than the retention period."""
    cutoff = time.time() - RETENTION_DAYS * 86400
    return [i for i in all_items() if i["created"] < cutoff and i["state"] in FINAL]


def orphans(min_age=86400):
    """Item folders that never got an item.json (e.g. a crash mid-upload)."""
    if not ITEMS.exists():
        return []
    cutoff = time.time() - min_age
    return [d.name for d in ITEMS.iterdir()
            if d.is_dir() and not (d / "item.json").exists() and d.stat().st_mtime < cutoff]


DEFAULT_SETTINGS = {"layout": "card", "caption": True, "font": "builtin", "custom_layouts": []}


def get_settings(owner):
    try:
        saved = json.loads((DATA_DIR / "settings" / f"{owner}.json").read_text())
    except (OSError, ValueError):
        saved = {}
    return {**DEFAULT_SETTINGS, **{k: v for k, v in saved.items() if k in DEFAULT_SETTINGS}}


def put_settings(owner, settings):
    d = DATA_DIR / "settings"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{owner}.json.tmp"
    tmp.write_text(json.dumps(settings))
    tmp.replace(d / f"{owner}.json")
