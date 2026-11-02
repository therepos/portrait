"""Caption fonts: a built-in Helvetica-style font plus any the user adds.

Added fonts live in DATA_DIR/fonts (uploaded from the page or copied in).
Every font is sized so its capital letters are the same height, and characters
it lacks (e.g. Chinese in Helvetica) fall back to the bundled Noto Sans TC.
"""
import functools
import re
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont
from PIL import ImageFont

from . import store

ASSETS = Path(__file__).resolve().parent / "assets"
BUILTIN_ID = "builtin"
BUILTIN_FILE = ASSETS / "texgyreheros-bold.otf"
FALLBACK_FILE = ASSETS / "NotoSansTC-Bold.otf"
EXTENSIONS = (".ttf", ".otf", ".ttc")
MAX_BYTES = 30 * 1024 * 1024


def folder():
    return store.DATA_DIR / "fonts"


def _font_file(font_id):
    if font_id in ("", BUILTIN_ID, None):
        return BUILTIN_FILE
    if not re.fullmatch(r"[\w.\- ]+", font_id) or not font_id.lower().endswith(EXTENSIONS):
        return BUILTIN_FILE
    path = folder() / font_id
    return path if path.exists() else BUILTIN_FILE


@functools.lru_cache(maxsize=64)
def _display_name(path, mtime):
    try:
        family, style = ImageFont.truetype(str(path), 20).getname()
        return f"{family} {style}".strip() if style and style.lower() != "regular" else family
    except OSError:
        return None


def list_fonts():
    out = [{"id": BUILTIN_ID, "name": "Helvetica-style (built in)", "builtin": True}]
    if folder().exists():
        for p in sorted(folder().iterdir()):
            if p.suffix.lower() in EXTENSIONS:
                name = _display_name(p, p.stat().st_mtime)
                if name:
                    out.append({"id": p.name, "name": name, "builtin": False})
    return out


def add(filename, data):
    """Save an uploaded font; returns its id. Raises ValueError if it isn't a usable font."""
    stem, ext = Path(filename or "font").stem, Path(filename or "").suffix.lower()
    if ext not in EXTENSIONS:
        raise ValueError("Only .ttf, .otf or .ttc font files can be added.")
    if len(data) > MAX_BYTES:
        raise ValueError("That font file is too big.")
    safe = re.sub(r"[^\w.\- ]+", "_", stem).strip(" .") or "font"
    folder().mkdir(parents=True, exist_ok=True)
    path, n = folder() / f"{safe}{ext}", 2
    while path.exists():
        path, n = folder() / f"{safe}-{n}{ext}", n + 1
    path.write_bytes(data)
    try:
        ImageFont.truetype(str(path), 20).getbbox("H")
        _cmap(str(path), path.stat().st_mtime)
    except Exception:
        path.unlink(missing_ok=True)
        raise ValueError("That file doesn't look like a working font.")
    return path.name


def remove(font_id):
    path = _font_file(font_id)
    if path != BUILTIN_FILE:
        path.unlink(missing_ok=True)


@functools.lru_cache(maxsize=64)
def _cmap(path, mtime):
    """Characters the font can draw."""
    if path.lower().endswith(".ttc"):
        font = TTCollection(path).fonts[0]
    else:
        font = TTFont(path, lazy=True)
    return frozenset(font.getBestCmap() or {})


@functools.lru_cache(maxsize=128)
def _sized(path, cap_height):
    """The font at the size whose capital H is cap_height pixels tall."""
    probe = ImageFont.truetype(path, 100)
    left, top, right, bottom = probe.getbbox("H", anchor="ls")
    size = max(4, round(100 * cap_height / max(1, bottom - top)))
    return ImageFont.truetype(path, size)


class Caption:
    """Fonts for one caption: the chosen font plus the fallback for missing characters."""

    def __init__(self, font_id, cap_height):
        path = _font_file(font_id)
        self.main = _sized(str(path), round(cap_height))
        self.fallback = _sized(str(FALLBACK_FILE), round(cap_height))
        self.covers = _cmap(str(path), path.stat().st_mtime)

    def font_for(self, ch):
        return self.main if ch.isspace() or ord(ch) in self.covers else self.fallback
