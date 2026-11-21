"""Stickers: cut the subject out and fit it to a 512×512 transparent WebP.

Two ways in:
- cut-out: a cartoon or drawing is cut out as it is, keeping its own colours
  (enlarged first with an anime upscaler if it is small);
- cartoon: a photo is redrawn by the AI on a plain white background (see jobs.py),
  then cut out here with the white treated as background.

The cut-out mask comes from ISNet (anime), the model rembg calls "isnet-anime".
Only its mask is used: rembg's own cut-out greys the semi-transparent edges.
"""
import io
import os
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import card

ASSETS = Path(__file__).resolve().parent / "assets"
MODELS = Path(os.environ.get("STICKER_MODELS") or "/models")
MASK_MODEL = MODELS / "isnet-anime.onnx"
UPSCALE_MODEL = ASSETS / "realesrgan-x4-anime6b.onnx"

SIZE = 512          # WhatsApp sticker size
MARGIN = 16         # empty space WhatsApp asks for around the artwork
MAX_BYTES = 100 * 1024
BORDER = 8          # white outline at 512 px
SMALL = 480         # subjects smaller than this (long side) are enlarged first

_lock = threading.Lock()   # one model run at a time; each one already uses every core
_sessions = {}


class StickerError(ValueError):
    pass


def available():
    return MASK_MODEL.exists()


def _session(path):
    if path not in _sessions:
        import onnxruntime as ort
        _sessions[path] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return _sessions[path]


# ---- models ------------------------------------------------------------------

def mask(im: Image.Image) -> np.ndarray:
    """Foreground probability 0..1 at the picture's size (ISNet anime, as rembg runs it)."""
    x = np.asarray(im.convert("RGB").resize((1024, 1024), Image.LANCZOS), np.float32)
    x = x / max(float(x.max()), 1e-6) - np.array([0.485, 0.456, 0.406], np.float32)
    s = _session(MASK_MODEL)
    with _lock:
        out = s.run(None, {s.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    out = (out - out.min()) / max(float(out.max() - out.min()), 1e-6)
    return np.clip(cv2.resize(out, im.size, interpolation=cv2.INTER_LINEAR), 0, 1)


def upscale(im: Image.Image, tile=128, pad=8) -> Image.Image:
    """4× with Real-ESRGAN x4plus anime 6B, in tiles to keep memory small."""
    x = np.asarray(im.convert("RGB"), np.float32).transpose(2, 0, 1) / 255
    _, h, w = x.shape
    out = np.zeros((3, h * 4, w * 4), np.float32)
    s = _session(UPSCALE_MODEL)
    name = s.get_inputs()[0].name
    with _lock:
        for y in range(0, h, tile):
            for x0 in range(0, w, tile):
                ya, yb = max(y - pad, 0), min(y + tile + pad, h)
                xa, xb = max(x0 - pad, 0), min(x0 + tile + pad, w)
                r = s.run(None, {name: x[None, :, ya:yb, xa:xb]})[0][0]
                ty, tx = (y - ya) * 4, (x0 - xa) * 4
                th, tw = (min(y + tile, h) - y) * 4, (min(x0 + tile, w) - x0) * 4
                out[:, y * 4:y * 4 + th, x0 * 4:x0 * 4 + tw] = r[:, ty:ty + th, tx:tx + tw]
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8).transpose(1, 2, 0))


# ---- alpha -----------------------------------------------------------------------

def _keep_subject(binary, min_share=0.02, hole_share=0.003):
    """Drop specks well away from the subject and fill small holes inside it."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if n <= 1:
        raise StickerError("Could not find anything to cut out.")
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= areas.max() * min_share
    solid = keep[labels]
    holes = ~solid
    n, labels, stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 4)
    h, w = binary.shape
    limit = solid.sum() * hole_share
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area <= limit and x > 0 and y > 0 and x + bw < w and y + bh < h:
            solid[labels == i] = True
    return solid


def _firm(m, solid):
    """Soft mask to a clean alpha: firm edges, small holes filled, specks gone."""
    a = np.clip((m - 0.3) / 0.4, 0, 1)
    near = cv2.dilate(solid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    a = np.where(near, a, 0)
    a[solid & (m <= 0.5)] = 1   # filled holes
    return a


def flat_background(rgb):
    """The background colour if the picture's edges are one plain colour, else None."""
    h, w, _ = rgb.shape
    b = max(2, round(min(h, w) * 0.01))
    edges = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                            rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)]).astype(np.float32)
    colour = np.median(edges, axis=0)
    return colour if (np.abs(edges - colour).max(axis=1) < 20).mean() > 0.8 else None


def _clear_background(rgb, m, a, bg, tol=24, sure=0.93):
    """On a plain background, big patches of its colour that reach the outside are
    background even where the model kept them, such as the gap between a tail and a
    body. Thin lines of that colour (outlines, whiskers) stay because only patches
    wider than about 4% of the picture count, and patches the model is sure about
    stay too (a white cat on white)."""
    h, w, _ = rgb.shape
    d = np.abs(rgb.astype(np.float32) - bg).max(axis=2)
    size = max(5, round(min(h, w) * 0.04)) | 1
    disc = lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    flat = cv2.morphologyEx((d < tol).astype(np.uint8), cv2.MORPH_OPEN, disc(size))
    n, labels = cv2.connectedComponents(flat, connectivity=4)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    region = np.isin(labels, edge[edge > 0])
    # the opening rounds off corners and leaves a soft glow in a shade near the background;
    # take those back, but only where the model is unsure, so outlines next to the patch stay whole
    region |= cv2.dilate(region.astype(np.uint8), disc(size // 2 | 1)).astype(bool) & (d < 3 * tol) & (m < sure)
    kept = region & (m > 0.5)
    n, labels = cv2.connectedComponents(kept.astype(np.uint8), connectivity=8)
    if n > 1:
        mean_m = np.bincount(labels.ravel(), m.ravel(), n) / np.maximum(np.bincount(labels.ravel(), None, n), 1)
        region &= ~(kept & (mean_m[labels] >= sure))
    a = a.copy()
    a[region] = 0
    # a thin stroke that only edged a patch cleared here is left hanging: drop it
    newly = (region & (m > 0.5)).astype(np.uint8)
    if newly.any():
        thin_k = disc(max(3, round(min(h, w) * 0.015)) | 1)
        body = a > 0.5
        thin = body & ~cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_OPEN, thin_k).astype(bool)
        near = cv2.dilate(newly, thin_k).astype(bool)
        a[near & (thin | (a < 0.25))] = 0  # with the faint haze it leaves
        n, labels, stats, _ = cv2.connectedComponentsWithStats((a > 0.5).astype(np.uint8), 8)
        if n > 2:  # and any bits of it now floating on their own, with their faint edges
            areas = stats[:, cv2.CC_STAT_AREA]
            small = areas < areas[1:].max() * 0.02
            small[0] = False
            loose = cv2.dilate(small[labels].astype(np.uint8), disc(5)).astype(bool)
            a[loose & ~cv2.dilate((~small[labels] & (labels > 0)).astype(np.uint8), disc(3)).astype(bool)] = 0
    return a


def cut_cartoon(im: Image.Image, bg="auto") -> Image.Image:
    """Cut out a drawing, keeping its colours exactly. bg: the plain background colour,
    None if there isn't one, or "auto" to look at the picture's edges."""
    rgb = np.asarray(im.convert("RGB"))
    m = mask(im)
    a = _firm(m, _keep_subject(m > 0.5))
    if isinstance(bg, str):
        bg = flat_background(rgb)
    if bg is not None:
        a = _clear_background(rgb, m, a, bg)
    rgba = np.dstack([rgb, (a * 255 + 0.5).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


def cut_white(im: Image.Image) -> Image.Image:
    """Cut out a picture drawn on plain white. Near the outline, where the model is
    unsure (between hair strands, fine edges), white counts as background and is
    taken out of the edge colours."""
    rgb = np.asarray(im.convert("RGB")).astype(np.float32)
    m = mask(im)
    solid = _keep_subject(m > 0.5)
    a_model = _firm(m, solid)
    r = max(3, round(min(im.size) * 0.02))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    u8 = solid.astype(np.uint8)
    zone = cv2.dilate(u8, k).astype(bool) & ~cv2.erode(u8, k).astype(bool)

    # how far each pixel is from white, against the colour of the subject next to it
    darkness = 255 - rgb.min(axis=2)
    sure = (a_model > 0.99) & (darkness > 40) & ~zone
    sigma = max(2.0, r / 3)
    num = cv2.GaussianBlur(darkness * sure, (0, 0), sigma)
    den = cv2.GaussianBlur(sure.astype(np.float32), (0, 0), sigma)
    near_fg = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 255)
    a_white = np.clip(darkness / np.maximum(near_fg, 30), 0, 1)
    a_white[darkness < 8] = 0  # pure white is always background

    a = np.where(zone, a_white, a_model)
    a = np.where(cv2.dilate(u8, k).astype(bool), a, 0)
    a[a < 0.04] = 0

    # unmix: colour = a*subject + (1-a)*white
    with np.errstate(divide="ignore", invalid="ignore"):
        fg = (rgb - (1 - a[..., None]) * 255) / a[..., None]
    fg = np.where(a[..., None] > 0.02, np.clip(fg, 0, 255), rgb)
    rgba = np.dstack([(fg + 0.5).astype(np.uint8), (a * 255 + 0.5).astype(np.uint8)])
    return Image.fromarray(rgba, "RGBA")


# ---- whole jobs ------------------------------------------------------------------

def cut_out(im: Image.Image) -> Image.Image:
    """Cartoon or drawing → cut-out RGBA, enlarged first if the subject is small."""
    im = im.convert("RGB")
    m = mask(im)
    ys, xs = np.nonzero(m > 0.5)
    if not len(xs):
        raise StickerError("Could not find anything to cut out.")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if max(x1 - x0, y1 - y0) < 0.05 * max(im.size):
        raise StickerError("Could not find a clear subject to cut out.")
    pad = round(max(x1 - x0, y1 - y0) * 0.06)
    box = (max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, im.width), min(y1 + pad, im.height))
    part = im.crop(box)
    if max(part.size) < SMALL and UPSCALE_MODEL.exists():
        part = upscale(part)
    return cut_cartoon(part, flat_background(np.asarray(im)))  # judged on the whole picture


def head_crop(photo: Image.Image) -> Image.Image:
    """Head to chest of the face nearest the middle, about square."""
    faces = card.find_faces(photo)
    if not faces:
        raise card.NoFaceError("No face found in the photo.")
    cx, cy = photo.width / 2, photo.height / 2
    f = min(faces, key=lambda f: (f.cx - cx) ** 2 + (f.eye_y - cy) ** 2)
    side = 3.2 * f.width
    left, top = f.cx - side / 2, f.eye_y - 1.3 * f.width
    right, bottom = left + side, top + side
    # stay inside the photo; a crop cut short by the edge stays cut short
    left, top = max(left, 0), max(top, 0)
    right, bottom = min(right, photo.width), min(bottom, photo.height)
    return photo.crop((round(left), round(top), round(right), round(bottom)))


def compose(cut: Image.Image, border=True) -> Image.Image:
    """Fit the cut-out into 512×512 with WhatsApp's margin and an optional white outline."""
    bbox = cut.getchannel("A").point(lambda v: 255 if v > 10 else 0).getbbox()
    if not bbox:
        raise StickerError("The cut-out came out empty.")
    cut = cut.crop(bbox)
    edge = BORDER if border else 0
    room = SIZE - 2 * (MARGIN + edge)
    s = room / max(cut.size)
    art = cut.resize((max(1, round(cut.width * s)), max(1, round(cut.height * s))), Image.LANCZOS)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    canvas.alpha_composite(art, ((SIZE - art.width) // 2, (SIZE - art.height) // 2))
    if not border:
        return canvas
    a = np.asarray(canvas.getchannel("A"))
    dist = cv2.distanceTransform((a < 128).astype(np.uint8), cv2.DIST_L2, 5)
    ring = np.clip(edge + 0.5 - dist, 0, 1)
    under = Image.new("RGBA", (SIZE, SIZE), (255, 255, 255, 0))
    under.putalpha(Image.fromarray((ring * 255 + 0.5).astype(np.uint8)))
    under.alpha_composite(canvas)
    return under


def to_webp(sticker: Image.Image) -> bytes:
    """WebP under WhatsApp's 100 KB (static stickers), as sharp as fits."""
    for alpha in (100, 60):
        for q in (95, 90, 85, 80, 70, 60, 50, 40, 30, 20):
            buf = io.BytesIO()
            sticker.save(buf, "WEBP", quality=q, method=6, alpha_quality=alpha)
            if buf.tell() <= MAX_BYTES:
                return buf.getvalue()
    return buf.getvalue()
