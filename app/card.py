"""Profile card rendering: find the face, crop to the template, add the caption.

Template geometry and framing were measured from the original GIMP template
(Pinterest 1200x1542.xcf) and a finished card.
"""
import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # HEIC support is optional
    pass

ASSETS = Path(__file__).resolve().parent / "assets"

# Card and caption layout
W, H = 1200, 1541
TEXT_X = 66             # left edge of the date's first letter
BASELINE = 1505         # caption baseline
FONT_PX = 42            # gives 31 px tall capitals, as in the template
INK, SEPARATOR = (0, 0, 0), (179, 179, 179)
BAR_TOP, BAR_W = 1476, 3  # grey "|" between date and name
GAP_BEFORE_BAR, GAP_AFTER_BAR = 26, 29

# Framing
FACE_W = 0.35   # face box width as share of card width
FACE_CX = 0.50  # face centre, from left
EYE_Y = 0.38    # eye line, from top

MAX_SIDE = 4000  # uploads are shrunk to this before anything else

_detector = None
_fonts = None
_strip = None


class NoFaceError(ValueError):
    pass


@dataclass
class Face:
    cx: float     # face centre x, in photo pixels
    eye_y: float  # eye line y
    width: float  # face box width


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


def crop(photo: Image.Image, face: Face, zoom=1.0, dx=0, dy=0) -> Image.Image:
    """Scale and cut the photo so the face sits where the template wants it.

    dx/dy move the photo right/down by that many card pixels.
    """
    scale = FACE_W * W / face.width * zoom
    scale = max(scale, W / photo.width, H / photo.height)  # card must be fully covered
    left = face.cx * scale - FACE_CX * W - dx
    top = face.eye_y * scale - EYE_Y * H - dy
    left = min(max(left, 0), photo.width * scale - W)
    top = min(max(top, 0), photo.height * scale - H)
    box = (left / scale, top / scale, (left + W) / scale, (top + H) / scale)
    return photo.resize((W, H), Image.LANCZOS, box=box)


def _assets():
    global _fonts, _strip
    if _fonts is None:
        _fonts = (ImageFont.truetype(str(ASSETS / "texgyreheros-bold.otf"), FONT_PX),
                  ImageFont.truetype(str(ASSETS / "NotoSansTC-Bold.otf"), FONT_PX))
        _strip = Image.open(ASSETS / "caption_strip.png").convert("RGBA")
    return _fonts, _strip


def _is_cjk(ch):
    return ord(ch) >= 0x2E80


def _draw_text(d, x, text, fonts):
    """Draw text starting with its ink at x; return x of the ink's right edge."""
    latin, cjk = fonts
    first = cjk if _is_cjk(text[0]) else latin
    pen = x - first.getbbox(text[0], anchor="ls")[0]
    right = x
    for ch in text:
        f = cjk if _is_cjk(ch) else latin
        d.text((pen, BASELINE), ch, font=f, fill=INK, anchor="ls")
        if not ch.isspace():
            right = pen + f.getbbox(ch, anchor="ls")[2]
        pen += f.getlength(ch)
    return right


def render(photo: Image.Image, face: Face, date: str, name: str, zoom=1.0, dx=0, dy=0) -> Image.Image:
    fonts, strip = _assets()
    card = crop(photo, face, zoom, dx, dy).convert("RGBA")
    card.alpha_composite(strip)
    card = card.convert("RGB")
    d = ImageDraw.Draw(card)
    date, name = date.strip(), name.strip()
    x = TEXT_X
    if date:
        x = _draw_text(d, x, date, fonts) + GAP_BEFORE_BAR
        if name:
            d.rectangle([x, BAR_TOP, x + BAR_W - 1, BASELINE - 1], fill=SEPARATOR)
            x += BAR_W + GAP_AFTER_BAR
    if name:
        _draw_text(d, x, name, fonts)
    return card


def to_png(card: Image.Image) -> bytes:
    buf = io.BytesIO()
    card.save(buf, "PNG", compress_level=6)
    return buf.getvalue()
