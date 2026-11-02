"""Framing: find the face, crop to a layout, optionally add the caption strip.

The Card layout and the caption style were measured from the original GIMP
template (Pinterest 1200x1542.xcf) and a finished card; the caption scales with
the layout's size.
"""
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageOps

from . import fonts

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # HEIC support is optional
    pass

ASSETS = Path(__file__).resolve().parent / "assets"
MAX_SIDE = 4000  # uploads are shrunk to this before anything else

# Caption style at 1200 px wide (the measured card); scaled for other sizes
CAPTION = {"strip": 93, "x": 66, "baseline_from_bottom": 36, "cap_height": 31,
           "sep_top": 29, "sep_width": 3, "gap_before": 26, "gap_after": 29}
INK, SEPARATOR = ImageColor.getrgb("#000000"), ImageColor.getrgb("#b3b3b3")


@dataclass(frozen=True)
class Layout:
    id: str
    name: str
    width: int
    height: int
    face_width: float   # face box width as share of the width
    face_cx: float      # face centre, from left
    eye_y: float        # eye line, from top

    def public(self):
        return {"id": self.id, "name": self.name, "width": self.width, "height": self.height}


# Platform sizes checked against 2026 guidance; framing tuned per shape
LAYOUTS = [
    Layout("card", "Card · 1200×1541", 1200, 1541, 0.35, 0.50, 0.38),
    Layout("pinterest", "Pinterest pin · 1000×1500", 1000, 1500, 0.33, 0.50, 0.36),
    Layout("instagram", "Instagram post · 1080×1350", 1080, 1350, 0.35, 0.50, 0.38),
    Layout("story", "Story (Instagram, TikTok, Pinterest) · 1080×1920", 1080, 1920, 0.30, 0.50, 0.33),
    Layout("square", "Profile picture · 1080×1080", 1080, 1080, 0.40, 0.50, 0.42),
]
BY_ID = {l.id: l for l in LAYOUTS}
MIN_SIDE, MAX_LAYOUT_SIDE = 200, 4000


def custom(width, height, name="Custom"):
    """A layout of any size, with framing interpolated from its shape."""
    width = int(min(max(width, MIN_SIDE), MAX_LAYOUT_SIDE))
    height = int(min(max(height, MIN_SIDE), MAX_LAYOUT_SIDE))
    r = height / width
    if r >= 1:  # portrait: from square (r=1) to story (r=1.78)
        k = min(1.0, (r - 1) / 0.78)
        face, eye = 0.40 - 0.10 * k, 0.42 - 0.09 * k
    else:       # landscape: keep the face the same share of the height as a square
        face, eye = 0.40 * r, 0.42
    return Layout(f"custom:{width}x{height}", f"{name} · {width}×{height}", width, height, face, 0.50, eye)


def resolve(layout_id, width=0, height=0):
    if layout_id in BY_ID:
        return BY_ID[layout_id]
    if width and height:
        return custom(width, height)
    return BY_ID["card"]


class NoFaceError(ValueError):
    pass


@dataclass
class Face:
    cx: float     # face centre x, in photo pixels
    eye_y: float  # eye line y
    width: float  # face box width


_detector = None


def load_photo(data: bytes) -> Image.Image:
    """Open an uploaded photo, honour phone rotation, shrink if huge."""
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return im


def find_face(photo: Image.Image) -> Face:
    global _detector
    rgb = np.asarray(photo)
    s = min(1.0, 1600 / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    h, w = small.shape[:2]
    if _detector is None:
        _detector = cv2.FaceDetectorYN.create(str(ASSETS / "face_detection_yunet.onnx"), "", (w, h), 0.6)
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        raise NoFaceError("No face found in the photo.")
    f = max(faces, key=lambda r: r[2] * r[3])  # biggest face
    x, _, fw, _ = f[:4]
    return Face(cx=(x + fw / 2) / s, eye_y=(f[5] + f[7]) / 2 / s, width=fw / s)


def place(photo: Image.Image, face: Face, layout: Layout, zoom=1.0, dx=0, dy=0):
    """Area of the photo (in photo pixels) that becomes the picture, and the shift
    actually applied once the photo's edges are reached. dx/dy move the photo
    right/down by that many output pixels."""
    W, H = layout.width, layout.height
    scale = layout.face_width * W / face.width * zoom
    scale = max(scale, W / photo.width, H / photo.height)  # the picture must be fully covered
    base_x = face.cx * scale - layout.face_cx * W
    base_y = face.eye_y * scale - layout.eye_y * H
    left = min(max(base_x - dx, 0), photo.width * scale - W)
    top = min(max(base_y - dy, 0), photo.height * scale - H)
    box = (left / scale, top / scale, (left + W) / scale, (top + H) / scale)
    return box, round(base_x - left), round(base_y - top)


def crop(photo: Image.Image, box, size) -> Image.Image:
    return photo.resize(size, Image.LANCZOS, box=box)


def _draw_text(d, x, baseline, text, cap):
    """Draw text starting with its ink at x; return x of the ink's right edge."""
    pen = x - cap.font_for(text[0]).getbbox(text[0], anchor="ls")[0]
    right = x
    for ch in text:
        f = cap.font_for(ch)
        d.text((pen, baseline), ch, font=f, fill=INK, anchor="ls")
        if not ch.isspace():
            right = pen + f.getbbox(ch, anchor="ls")[2]
        pen += f.getlength(ch)
    return right


def add_caption(picture: Image.Image, date: str, name: str, font_id=fonts.BUILTIN_ID) -> Image.Image:
    """White strip along the bottom with "date | name", scaled to the picture."""
    W, H = picture.size
    s = min(W, H) / 1200
    c = {k: v * s for k, v in CAPTION.items()}
    card = picture.copy()
    d = ImageDraw.Draw(card)
    d.rectangle([0, H - round(c["strip"]), W, H], fill=(255, 255, 255))
    cap = fonts.Caption(font_id, c["cap_height"])
    baseline = H - round(c["baseline_from_bottom"])
    date, name = date.strip(), name.strip()
    x = round(c["x"])
    if date:
        x = _draw_text(d, x, baseline, date, cap) + c["gap_before"]
        if name:
            sw = max(1, round(c["sep_width"]))
            d.rectangle([x, baseline - round(c["sep_top"]), x + sw - 1, baseline - 1], fill=SEPARATOR)
            x += sw + c["gap_after"]
    if name:
        _draw_text(d, x, baseline, name, cap)
    return card


def render(photo, face, layout: Layout, zoom=1.0, dx=0, dy=0, caption=None):
    """The framed picture, and the shift actually applied (dx, dy).
    caption is None or (date, name, font_id)."""
    box, dx, dy = place(photo, face, layout, zoom, dx, dy)
    picture = crop(photo, box, (layout.width, layout.height))
    if caption:
        picture = add_caption(picture, *caption)
    return picture, dx, dy


def to_png(card: Image.Image) -> bytes:
    buf = io.BytesIO()
    card.save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def to_jpeg(card: Image.Image) -> bytes:
    """Quick preview while the user drags or zooms."""
    buf = io.BytesIO()
    card.save(buf, "JPEG", quality=85)
    return buf.getvalue()
