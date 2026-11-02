# portrait

A small web app that turns a photo into a 1200×1541 profile card: it finds the
face, crops head-and-shoulders to a fixed framing, and adds a white caption strip
reading `date | name`. It replaces a manual GIMP routine (open template, scale,
fixed-size select, paste, edit caption, export).

Works from a phone or a desktop browser. Chinese names are supported.
Optionally, it can send the framed crop through a ComfyUI workflow (for example
Qwen-Image-Edit) to restore blurry or small photos at exactly card size.

## Run it

Any machine with Docker:

```sh
mkdir portrait && cd portrait
curl -fsSLO https://raw.githubusercontent.com/therepos/portrait/main/docker-compose.yml
mkdir -p output
docker compose up -d
```

Open `http://<machine-ip>:8080`.

Or without compose:

```sh
docker run -d --name portrait -p 8080:8000 ghcr.io/therepos/portrait:latest
```

### Save to folder (optional)

With `SAVE_DIR` set and a folder mounted there, the page shows a **Save to folder**
button that writes `<first name>.png` into it (never overwrites; repeats become
`name-2.png`). The container runs as user 1000, so the folder must be writable by
that user. Without `SAVE_DIR`, only **Download** is offered.

### Update

```sh
docker compose pull && docker compose up -d
```

## Using it

**Restore tab:** drop or pick photos. Each is restored by AI in the background (a few
minutes each, one at a time) with a progress bar and time left. Leave the page and come
back; the queue keeps running. Finished photos can be downloaded or opened with
**Create Portrait**.

**Portrait tab:**

1. Pick a photo (or come from **Create Portrait**).
2. Type the date (pre-filled with today) and name.
3. Check the preview. Use **Zoom** and the arrows if the framing is off, ⟲ to reset.
4. Optional: **Restore with AI** restores just the card area (the most detail for a
   card), then compare with the **Original / Restored** switch.
5. **Download** or **Save to folder**.

**History:** restores and cards made in this browser, with Download, Create Portrait and
Delete. Each browser only sees its own (a random id in a cookie). Items are deleted after
`RETENTION_DAYS` (default 7). Mount `/data` to keep them across container updates.

## How the framing works

A face detector ([YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet))
finds the largest face and its eyes. The photo is scaled so the face box is 35% of
the card width, the face is centred horizontally, and the eye line sits 38% from the
top. These numbers were measured from a hand-made card and live in the template
(see below).

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `SAVE_DIR` | unset | Folder inside the container for **Save to folder** |
| `DATA_DIR` | `/data` | History and restore queue; mount it to keep them across updates |
| `RETENTION_DAYS` | `7` | History items are deleted after this many days |
| `TEMPLATE_DIR` | built-in | Folder with your own `template.json` (see below) |
| `COMFYUI_URL` | unset | ComfyUI address, e.g. `http://comfyui:8188`. Enables **Restore with AI** |
| `COMFYUI_WORKFLOW` | `/config/comfyui-workflow.json` | API-format workflow file |
| `COMFYUI_TIMEOUT` | `900` | Seconds to wait for one restore |
| `COMFYUI_MEGAPIXELS` | `1.0` | Size the model works at; the result is scaled to the card |
| `GPU_HELPER_URL` | unset | Address of the GPU helper container (see below) |
| `RESTORE_VRAM_GB` | `16` | Free GPU memory a restore needs before it runs |
| `MAX_UPLOAD_MB` | `40` | Largest accepted upload |
| `MAX_PHOTOS` | `20` | Uploaded photos kept in memory for re-rendering |

## Template

The layout (card size, face framing, overlay image, caption font and position) is in
[`app/template/template.json`](app/template/template.json). To change it without
rebuilding, copy that file into a folder, edit it, mount the folder and set
`TEMPLATE_DIR`. Images and fonts it names are looked up in that folder first, then in
the built-in template and assets.

## AI restore with ComfyUI

1. In ComfyUI, open your restore workflow and use **Export (API)**.
2. Mount the file at `/config/comfyui-workflow.json` and set `COMFYUI_URL`.
   If ComfyUI runs in Docker on the same host, put both containers on one network.

The app changes only three things in the workflow: the image fed to its single
**Load Image** node, the resize step in front of the model (an automatic resize such as
`FluxKontextImageScale` is replaced by an exact resize), and the sampler seed. The
prompt and models come from your file. It sends the framed crop at the photo's own
resolution (shrunk only if larger) and has the model work at about
`COMFYUI_MEGAPIXELS` (default 1.0) in the card's shape, e.g. 896×1152 for 1200×1541.
Qwen Image Edit stays faithful to the photo near 1 MP but starts re-composing it when
asked for much more. The result is enlarged slightly to card size and the few extra rows
are trimmed. Results are saved by ComfyUI as the
workflow says, and fetched over its API.

Restoring needs enough free GPU memory for the model; if another program holds the
card, ComfyUI reports out of memory and the app shows that.

### Freeing the GPU from another service

If a systemd service on the host (for example a local LLM server) holds the GPU, a small
helper container can pause it for the length of a restore. Before each restore the app
checks free GPU memory through ComfyUI. If there isn't enough, it first lets ComfyUI drop
its own cached models; if that's still not enough, the page asks **Pause it during this
restore?** (with an *Always allow* option) before anything is stopped.

Add the helper to the same compose file:

```yaml
  portrait:
    # ...as above, plus:
    environment:
      GPU_HELPER_URL: http://portrait-gpu:9000
    networks: [default, gpu]

  portrait-gpu:
    image: ghcr.io/therepos/portrait:latest
    command: ["python", "-m", "app.broker"]
    user: "0:0"                     # root, to control systemd; no host setup needed
    environment:
      PAUSE_UNITS: llama.service    # comma-separated; only these can be paused
    volumes:
      - /run/dbus/system_bus_socket:/run/dbus/system_bus_socket
    security_opt:
      - apparmor=unconfined         # Docker's default AppArmor profile blocks D-Bus
    networks: [gpu]
    restart: unless-stopped

networks:
  gpu:
    internal: true                  # only the app can reach the helper
```

The helper accepts only *pause* and *resume*, only for `PAUSE_UNITS`, and publishes no
port. Paused services are started again when the restore finishes or fails, after
`PAUSE_MAX_MINUTES` (default 30) if the app never reports back, and when the helper
stops. A service that was already stopped is left stopped. Anything using the service
is cut off while it is paused.

## Develop

```sh
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Every push to `main` builds and publishes `ghcr.io/therepos/portrait:latest`
(amd64 and arm64) after a smoke test. Tag `vX.Y.Z` to publish a versioned image.

## Bundled assets

| File | Source | Licence |
|---|---|---|
| `texgyreheros-bold.otf` | TeX Gyre Heros (Helvetica-style) | GUST Font License |
| `NotoSansTC-Bold.otf` | Noto Sans CJK TC | SIL Open Font License 1.1 |
| `face_detection_yunet.onnx` | OpenCV Zoo YuNet | MIT |
| `caption_strip.png` | Extracted from the original GIMP template | — |
