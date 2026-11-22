"""Stickers: cut-out, the 512×512 WhatsApp format, prompts sent to ComfyUI, the API."""
import asyncio
import io
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app import auth, card, comfy, jobs, main, sticker, store


def disc_picture(size=400, bg=(255, 255, 255), fg=(200, 60, 90)):
    im = Image.new("RGB", (size, size), bg)
    ImageDraw.Draw(im).ellipse((100, 100, 300, 300), fill=fg)
    return im


def disc_mask(im):
    """Stand-in for the model: the disc, with a soft edge."""
    yy, xx = np.mgrid[0:im.height, 0:im.width]
    d = np.hypot(xx - im.width / 2, yy - im.height / 2)
    return np.clip((102 - d) / 4, 0, 1).astype(np.float32)


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(sticker, "mask", disc_mask)
    monkeypatch.setattr(sticker, "available", lambda: True)


def test_cartoon_cut_keeps_colours_and_clears_background(fake_model):
    cut = sticker.cut_cartoon(disc_picture())
    assert cut.getpixel((200, 200)) == (200, 60, 90, 255)
    assert cut.getpixel((10, 10))[3] == 0


def test_white_background_is_taken_out_of_the_edges(fake_model):
    cut = np.asarray(sticker.cut_white(disc_picture()))
    assert tuple(cut[200, 200]) == (200, 60, 90, 255)
    assert cut[5, 5, 3] == 0
    edge = cut[200, 299:303]       # anti-aliased rim: partly see-through, colour not whitened
    soft = edge[(edge[:, 3] > 0) & (edge[:, 3] < 255)]
    assert all(px[1] < 120 for px in soft)


def test_plain_background_kept_by_the_model_is_cleared(monkeypatch):
    """A cat on black whose tail leaves a black gap the model wrongly keeps."""
    im = Image.new("RGB", (400, 400), (0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((80, 80, 180, 360), fill=(250, 250, 250), outline=(30, 30, 30), width=4)    # body
    d.rectangle((260, 80, 330, 360), fill=(250, 250, 250), outline=(30, 30, 30), width=4)   # tail
    model = np.zeros((400, 400), np.float32)
    model[76:364, 76:334] = 1.0
    model[90:350, 190:250] = 0.8      # unsure about the gap, but keeps it
    monkeypatch.setattr(sticker, "mask", lambda _: model)
    a = np.asarray(sticker.cut_cartoon(im))[..., 3]
    assert a[200, 220] == 0                         # the gap is see-through
    assert a[200, 130] == 255 and a[200, 295] == 255
    assert a[200, 81] == 255                        # outline kept


def test_white_subject_on_white_is_not_cleared(monkeypatch):
    im = Image.new("RGB", (400, 400), (255, 255, 255))
    ImageDraw.Draw(im).ellipse((100, 100, 300, 300), outline=(20, 20, 20), width=3)
    monkeypatch.setattr(sticker, "mask", disc_mask)
    a = np.asarray(sticker.cut_cartoon(im))[..., 3]
    assert a[200, 200] == 255


def test_specks_dropped_and_small_holes_filled():
    b = np.zeros((200, 200), bool)
    b[50:150, 50:150] = True
    b[98:101, 98:101] = False      # small hole
    b[5:7, 5:7] = True             # speck
    solid = sticker._keep_subject(b)
    assert solid[99, 99] and not solid[5, 5]


def test_sticker_is_512_square_with_margin_and_outline(fake_model):
    cut = sticker.cut_cartoon(disc_picture())
    plain = np.asarray(sticker.compose(cut, border=False))
    lined = np.asarray(sticker.compose(cut, border=True))
    assert plain.shape == lined.shape == (512, 512, 4)
    for s in (plain, lined):
        assert s[:sticker.MARGIN, :, 3].max() == 0 and s[:, :sticker.MARGIN, 3].max() == 0
    # the outline is white and sits just outside the artwork
    ys, xs = np.nonzero(lined[..., 3] > 250)
    top = ys.min()
    assert tuple(lined[top, xs[ys == top][0], :3]) == (255, 255, 255)


def test_webp_fits_whatsapp_limit():
    busy = Image.fromarray(np.random.default_rng(1).integers(0, 255, (128, 128, 3), np.uint8)).resize((512, 512))
    shape = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(shape).ellipse((16, 16, 496, 496), fill=255)
    busy.putalpha(shape)
    data = sticker.to_webp(busy)
    im = Image.open(io.BytesIO(data))
    assert im.format == "WEBP" and im.size == (512, 512) and len(data) <= sticker.MAX_BYTES


def test_head_crop_uses_the_face_nearest_the_middle(monkeypatch):
    photo = Image.new("RGB", (2000, 1500))
    monkeypatch.setattr(card, "find_faces", lambda im: [card.Face(300, 400, 300), card.Face(1000, 700, 200)])
    c = sticker.head_crop(photo)
    assert c.size == (640, 640)


def test_style_prompts_replace_the_workflow_prompts():
    wf = {
        "1": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": "restore"}},
        "2": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": "cartoon"}},
        "3": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["1", 0]}},
        "4": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["2", 0]}},
        "5": {"class_type": "KSampler", "inputs": {"positive": ["3", 0], "negative": ["4", 0]}},
    }
    comfy.set_prompts(wf, "draw a sticker", "photo")
    assert wf["1"]["inputs"]["prompt"] == "draw a sticker"
    assert wf["2"]["inputs"]["prompt"] == "photo"
    with pytest.raises(comfy.ComfyError):
        comfy.set_prompts({"5": {"class_type": "KSampler", "inputs": {}}}, "a", "b")


@pytest.fixture
def client(monkeypatch, tmp_path, fake_model):
    monkeypatch.setattr(auth, "ALLOW_OPEN", True)
    monkeypatch.setattr(auth, "USERNAME", "")
    monkeypatch.setattr(auth, "PASSWORD", "")
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "ITEMS", tmp_path / "items")
    monkeypatch.setattr(sticker, "cut_out", sticker.cut_cartoon)
    return TestClient(main.app)


def png(im):
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def test_cut_out_sticker_end_to_end(client):
    r = client.post("/api/stickers", data={"mode": "cutout", "border": "true"},
                    files={"photo": ("cat.png", png(disc_picture()), "image/png")})
    assert r.status_code == 200, r.text
    item = store.load(r.json()["items"][0]["id"])
    assert item["state"] == "queued" and not jobs.on_gpu(item)
    asyncio.run(jobs._sticker(item))
    done = client.get("/api/items").json()["items"][0]
    assert done["state"] == "done" and done["kind"] == "sticker"
    f = client.get(f"/api/items/{done['id']}/file")
    assert f.headers["content-type"] == "image/webp" and Image.open(io.BytesIO(f.content)).size == (512, 512)

    # outline off and on again, from the kept cut-out
    r = client.post(f"/api/stickers/{done['id']}/border", data={"border": "false"})
    assert r.json()["border"] is False
    corner_free = Image.open(io.BytesIO(client.get(f"/api/items/{done['id']}/file").content))
    assert corner_free.mode == "RGBA"

    z = zipfile.ZipFile(io.BytesIO(client.get("/api/stickers.zip").content))
    assert len(z.namelist()) == 1 and z.namelist()[0].endswith(".webp")


def test_cartoon_needs_ai_and_a_face(client, monkeypatch):
    monkeypatch.setattr(comfy, "enabled", lambda: False)
    r = client.post("/api/stickers", data={"mode": "cartoon"},
                    files={"photo": ("me.png", png(disc_picture()), "image/png")})
    assert r.status_code == 400
    monkeypatch.setattr(comfy, "enabled", lambda: True)
    monkeypatch.setattr(card, "find_faces", lambda im: [])
    r = client.post("/api/stickers", data={"mode": "cartoon"},
                    files={"photo": ("me.png", png(disc_picture()), "image/png")})
    assert r.status_code == 422 and "cut it out" in r.json()["detail"]


def test_cartoon_queues_two_versions_with_the_chosen_style(client, monkeypatch):
    monkeypatch.setattr(comfy, "enabled", lambda: True)
    monkeypatch.setattr(card, "find_faces", lambda im: [card.Face(200, 180, 80)])

    async def roomy():
        return None
    monkeypatch.setattr(comfy, "vram_free", roomy)
    client.put("/api/settings", json={"sticker_styles": [{"name": "Mine", "prompt": "draw it", "avoid": "photo"}]})
    r = client.post("/api/stickers", data={"mode": "cartoon", "versions": "2", "style": "0"},
                    files={"photo": ("me.png", png(disc_picture()), "image/png")})
    assert r.status_code == 200, r.text
    made = [store.load(i["id"]) for i in r.json()["items"]]
    assert len(made) == 2 and all(jobs.on_gpu(i) and i["prompt"] == "draw it" and i["style"] == "Mine" for i in made)


def test_finished_stickers_never_expire(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "ITEMS", tmp_path / "items")
    old = time.time() - 400 * 86400
    keep = store.create("o", "sticker", created=old)
    gone = store.create("o", "card", created=old)
    failed = store.create("o", "sticker", created=old, state="error")
    assert {i["id"] for i in store.expired()} == {gone["id"], failed["id"]}
    assert keep["id"] not in {i["id"] for i in store.expired()}


def test_styles_are_checked_and_can_be_reset(client):
    s = client.put("/api/settings", json={"sticker_styles": [{"name": "A", "prompt": "x"}, {"name": "B", "prompt": " "}],
                                          "sticker_style": 5}).json()
    assert [st["name"] for st in s["sticker_styles"]] == ["A"] and s["sticker_style"] == 0
    s = client.put("/api/settings", json={"sticker_styles": []}).json()
    assert s["sticker_styles"][0]["name"] == "Cute anime"


def character(size=400):
    """A simple cartoon figure on a coloured background."""
    yy, xx = np.mgrid[0:size, 0:size]
    bg = np.dstack([120 + xx * 60 // size, 190 + 0 * xx, 150 + yy * 60 // size]).astype(np.uint8)
    im = Image.fromarray(bg)
    d = ImageDraw.Draw(im)
    d.ellipse((110, 230, 290, 390), fill=(250, 200, 80), outline=(30, 30, 30), width=6)
    d.ellipse((100, 60, 300, 260), fill=(255, 225, 190), outline=(30, 30, 30), width=6)
    d.ellipse((150, 130, 180, 170), fill=(30, 30, 30))
    d.ellipse((220, 130, 250, 170), fill=(30, 30, 30))
    d.arc((160, 170, 240, 220), 20, 160, fill=(200, 40, 60), width=6)
    return im


@pytest.mark.skipif(not sticker.MASK_MODEL.exists(), reason="cut-out model not present")
def test_real_model_cuts_a_drawing():
    cut = sticker.cut_out(character())  # small, so it is enlarged 4x first
    a = np.asarray(cut)[..., 3]
    assert cut.width > 900 and a[a.shape[0] // 2, a.shape[1] // 2] > 200 and a[2, 2] < 30


def test_polish_goes_through_the_ai_then_is_cut_out(client, monkeypatch):
    monkeypatch.setattr(comfy, "enabled", lambda: True)

    async def roomy():
        return None
    monkeypatch.setattr(comfy, "vram_free", roomy)
    r = client.post("/api/stickers", data={"mode": "cutout", "polish": "true"},
                    files={"photo": ("cat.png", png(disc_picture()), "image/png")})
    assert r.status_code == 200, r.text
    item = store.load(r.json()["items"][0]["id"])
    assert item["polish"] and jobs.on_gpu(item) and item["prompt"] == store.POLISH["prompt"]

    sent = {}

    async def fake_comfy(it, picture, size, prompts=None):
        sent["prompts"] = prompts
        return disc_picture().resize(size)
    monkeypatch.setattr(jobs, "_comfy", fake_comfy)
    asyncio.run(jobs._sticker(item))
    done = store.load(item["id"])
    assert done["state"] == "done" and sent["prompts"][0] == store.POLISH["prompt"]
    assert store.path(done, "drawn.png").exists()

    # a polished sticker can be tried again; a plain one can be polished
    again = client.post(f"/api/stickers/{item['id']}/again").json()
    assert again["polish"] is True
    plain = client.post("/api/stickers", data={"mode": "cutout", "polish": "false"},
                        files={"photo": ("cat.png", png(disc_picture()), "image/png")}).json()["items"][0]
    assert not jobs.on_gpu(store.load(plain["id"]))
    asyncio.run(jobs._sticker(store.load(plain["id"])))
    polished = client.post(f"/api/stickers/{plain['id']}/polish")
    assert polished.status_code == 200 and polished.json()["polish"] is True
    assert client.post(f"/api/stickers/{item['id']}/polish").status_code == 400


def test_polish_needs_ai(client, monkeypatch):
    monkeypatch.setattr(comfy, "enabled", lambda: False)
    r = client.post("/api/stickers", data={"mode": "cutout", "polish": "true"},
                    files={"photo": ("cat.png", png(disc_picture()), "image/png")})
    assert r.status_code == 400 and "Polish" in r.json()["detail"]


def test_polish_instructions_can_be_edited_and_reset(client):
    s = client.put("/api/settings", json={"polish_prompt": {"prompt": "Make it crisp", "avoid": "blur"},
                                          "sticker_polish": False}).json()
    assert s["polish_prompt"] == {"prompt": "Make it crisp", "avoid": "blur"} and s["sticker_polish"] is False
    s = client.put("/api/settings", json={"polish_prompt": {"prompt": ""}}).json()
    assert s["polish_prompt"] == store.POLISH
