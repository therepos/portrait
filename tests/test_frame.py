"""Layouts, the caption switch, fonts and remembered settings."""
import shutil

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import auth, card, fonts, main, store

PHOTO = Image.new("RGB", (2000, 2600), (90, 120, 160))
FACE = card.Face(cx=1000, eye_y=900, width=500)


def white_rows(im, y0, y1):
    return all(im.getpixel((5, y)) == (255, 255, 255) for y in range(y0, y1))


def test_card_layout_matches_measured_template():
    im, _, _ = card.render(PHOTO, FACE, card.BY_ID["card"], caption=("3 Nov", "Erika", "builtin"))
    assert im.size == (1200, 1541)
    assert white_rows(im, 1448, 1541) and not white_rows(im, 1440, 1448)


def test_caption_off_has_no_strip():
    im, _, _ = card.render(PHOTO, FACE, card.BY_ID["card"], caption=None)
    assert im.getpixel((5, 1530)) == (90, 120, 160)


@pytest.mark.parametrize("layout", card.LAYOUTS, ids=lambda l: l.id)
def test_every_layout_has_its_size_and_a_scaled_caption(layout):
    im, _, _ = card.render(PHOTO, FACE, layout, caption=("1 Jan", "Ann", "builtin"))
    assert im.size == (layout.width, layout.height)
    strip = round(card.CAPTION["strip"] * min(layout.width, layout.height) / 1200)
    assert white_rows(im, layout.height - strip, layout.height)


def test_custom_size_is_clamped_and_framed():
    l = card.resolve("custom", 50, 9000)
    assert (l.width, l.height) == (card.MIN_SIDE, card.MAX_LAYOUT_SIDE)
    wide = card.resolve("custom", 1920, 1080)
    im, _, _ = card.render(PHOTO, FACE, wide)
    assert im.size == (1920, 1080)


def test_unknown_layout_falls_back_to_card():
    assert card.resolve("nope").id == "card"


def test_chinese_falls_back_when_font_lacks_it():
    cap = fonts.Caption("builtin", 31)
    assert cap.font_for("A") is cap.main
    assert cap.font_for("費") is cap.fallback


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "ALLOW_OPEN", True)
    monkeypatch.setattr(auth, "USERNAME", "")
    monkeypatch.setattr(auth, "PASSWORD", "")
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "ITEMS", tmp_path / "items")
    return TestClient(main.app)


def test_font_upload_list_remove(client, tmp_path):
    bad = client.post("/api/fonts", files={"font": ("x.ttf", b"not a font", "font/ttf")})
    assert bad.status_code == 400
    data = fonts.BUILTIN_FILE.read_bytes()
    r = client.post("/api/fonts", files={"font": ("My Font.otf", data, "font/otf")})
    assert r.status_code == 200
    font_id = r.json()["id"]
    assert any(f["id"] == font_id for f in client.get("/api/fonts").json()["fonts"])
    assert fonts.Caption(font_id, 31).main.path.endswith(font_id)
    client.delete(f"/api/fonts/{font_id}")
    assert not any(f["id"] == font_id for f in client.get("/api/fonts").json()["fonts"])


def test_font_ids_cannot_escape_the_folder(client, tmp_path):
    secret = tmp_path / "secret.ttf"
    shutil.copy(fonts.BUILTIN_FILE, secret)
    client.delete("/api/fonts/..%2Fsecret.ttf")
    assert secret.exists()
    assert fonts._font_file("../secret.ttf") == fonts.BUILTIN_FILE


def test_settings_are_remembered_and_checked(client):
    assert client.get("/api/settings").json()["layout"] == "card"
    s = client.put("/api/settings", json={"layout": "square", "caption": False, "font": "x.ttf",
                                          "custom_layouts": [{"name": "Blog", "width": 800, "height": 1000},
                                                             {"name": "Too big", "width": 99999, "height": 5}]}).json()
    assert s["layout"] == "square" and s["caption"] is False
    assert s["custom_layouts"] == [{"name": "Blog", "width": 800, "height": 1000}]
    assert client.get("/api/settings").json() == s
