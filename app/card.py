"""Profile card rendering: find the face, crop to the template, add the caption.

Layout lives in a template folder (template.json plus its images and fonts).
The built-in template was measured from the original GIMP template
(Pinterest 1200x1542.xcf) and a finished card; TEMPLATE_DIR overrides it.
"""
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # HEIC support is optional
    pass

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
BUILTIN_TEMPLATE = HERE / "template"

MAX_SIDE = 4000  # uploads are shrunk to this before anything else

_detector = None


class NoFaceError(ValueError):
    pass


@dataclass
class Face:
    cx: float     # face centre x, in photo pixels
    eye_y: float  # eye line y
    width: float  # face box width


class Template:
    def __init__(self, folder: Path):
        self.folder = folder
        t = json.loads((folder / "template.json").read_text())
        self.width, self.height = t["size"]
        f = t["framing"]
        self.face_width, self.face_cx, self.eye_y = f["face_width"], f["face_center_x"], f["eye_line_y"]
        self.overlay = Image.open(self._file(t["overlay"])).convert("RGBA") if t.get("overlay") else None
        c = t["caption"]
        self.text_x, self.baseline = c["x"], c["baseline"]
        self.latin = ImageFont.truetype(str(self._file(c["font"])), c["font_size"])
        self.cjk = ImageFont.truetype(str(self._file(c.get("cjk_font", c["font"]))), c["font_size"])
        self.ink = ImageColor.getrgb(c["color"])
        s = c["separator"]
        self.sep_color = ImageColor.getrgb(s["color"])
        self.sep_width, self.sep_top = s["width"], s["top"]
        self.gap_before, self.gap_after = s["gap_before"], s["gap_after"]

    def _file(self, name):
        """Files are looked up in the template folder, then the bundled assets."""
        for base in (self.folder, BUILTIN_TEMPLATE, ASSETS):
            if (base / name).exists():
                return base / name
        raise FileNotFoundError(f"Template file not found: {name}")


TEMPLATE = Template(Path(os.environ.get("TEMPLATE_DIR") or BUILTIN_TEMPLATE))


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


def frame(photo: Image.Image, face: Face, zoom=1.0, dx=0, dy=0, extra_h=0, t=TEMPLATE):
    """Area of the photo (in photo pixels) that becomes the card.

    dx/dy move the photo right/down by that many card pixels. extra_h adds
    rows below the card, for tools that need a taller image than the card.
    """
    W, H = t.width, t.height + extra_h
    scale = t.face_width * t.width / face.width * zoom
    scale = max(scale, W / photo.width, H / photo.height)  # card must be fully covered
    left = face.cx * scale - t.face_cx * t.width - dx
    top = face.eye_y * scale - t.eye_y * t.height - dy
    left = min(max(left, 0), photo.width * scale - W)
    top = min(max(top, 0), photo.height * scale - H)
    return (left / scale, top / scale, (left + W) / scale, (top + H) / scale)


def crop(photo: Image.Image, box, size) -> Image.Image:
    return photo.resize(size, Image.LANCZOS, box=box)


def _is_cjk(ch):
    return ord(ch) >= 0x2E80


def _draw_text(d, x, text, t):
    """Draw text starting with its ink at x; return x of the ink's right edge."""
    first = t.cjk if _is_cjk(text[0]) else t.latin
    pen = x - first.getbbox(text[0], anchor="ls")[0]
    right = x
    for ch in text:
        f = t.cjk if _is_cjk(ch) else t.latin
        d.text((pen, t.baseline), ch, font=f, fill=t.ink, anchor="ls")
        if not ch.isspace():
            right = pen + f.getbbox(ch, anchor="ls")[2]
        pen += f.getlength(ch)
    return right


def finish(picture: Image.Image, date: str, name: str, t=TEMPLATE) -> Image.Image:
    """Put the overlay and caption on a card-sized (or taller) picture."""
    card = picture.convert("RGBA").crop((0, 0, t.width, t.height))
    if t.overlay:
        card.alpha_composite(t.overlay)
    card = card.convert("RGB")
    d = ImageDraw.Draw(card)
    date, name = date.strip(), name.strip()
    x = t.text_x
    if date:
        x = _draw_text(d, x, date, t) + t.gap_before
        if name:
            d.rectangle([x, t.sep_top, x + t.sep_width - 1, t.baseline - 1], fill=t.sep_color)
            x += t.sep_width + t.gap_after
    if name:
        _draw_text(d, x, name, t)
    return card


def render(photo: Image.Image, face: Face, date: str, name: str, zoom=1.0, dx=0, dy=0, t=TEMPLATE) -> Image.Image:
    box = frame(photo, face, zoom, dx, dy, t=t)
    return finish(crop(photo, box, (t.width, t.height)), date, name, t)


def to_png(card: Image.Image) -> bytes:
    buf = io.BytesIO()
    card.save(buf, "PNG", compress_level=6)
    return buf.getvalue()
