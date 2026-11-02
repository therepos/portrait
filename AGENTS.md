# Notes for anyone changing this app

For people and coding agents. The README covers using and deploying the app; this file
covers how it is built and the rules it follows.

## What it is

A self-hosted web app with two tabs:

- **Restore**: sends photos through a ComfyUI workflow (Qwen Image Edit) to sharpen them.
  Runs from a queue on disk with live progress.
- **Frame**: finds the face and frames the photo for a layout (Card, Pinterest, Instagram,
  Story, Profile picture, or a custom size), with drag/zoom and an optional caption strip.

Python (FastAPI), one HTML page with plain JavaScript, no build step, one Docker image.

## Design rules

1. **Whatever the owner does, any user can do from the app.** No steps that only work
   because someone read a chat or the code: fonts are added from the page, layouts are
   picked from a list, settings are remembered. If something needs configuring, the app
   says so on screen (e.g. the *Setup needed* page).
2. **Safe by default.** Sign-in is required unless `ALLOW_NO_LOGIN=true` is set on purpose.
   Anything that can stop another program (the GPU helper) asks in the page first.
3. **Nothing is lost silently.** Restores survive page reloads and restarts; a paused
   service is always started again (on finish, failure, timeout, shutdown or lease expiry).
4. **Plain words in the interface.** Short labels, no jargon, errors that say what to do.
5. **Optional features stay optional.** Without ComfyUI the Restore tab says it isn't set
   up; without the GPU helper nothing is paused; without `SAVE_DIR` there is no Save button.
6. **Per-user data stays per user.** History and settings belong to the signed-in account
   (or the browser when sign-in is off) and are deleted after `RETENTION_DAYS`.

## Where things live

| File | What it does |
|---|---|
| `app/main.py` | Web server: sign-in middleware, history, restore and frame endpoints, settings |
| `app/card.py` | Face detection (YuNet), layouts and their framing numbers, caption drawing |
| `app/fonts.py` | Built-in and uploaded fonts, same-capital-height sizing, Chinese fallback |
| `app/jobs.py` | Restore queue, progress and time-left estimates, GPU pause per batch |
| `app/comfy.py` | ComfyUI client: edits the workflow (input, size, seed), live step progress |
| `app/gpu.py` / `app/broker.py` | App side / helper container that pauses systemd units over D-Bus |
| `app/auth.py` | Single-account sign-in with signed cookies and lockout |
| `app/store.py` | History items and settings on disk under `DATA_DIR` |
| `app/static/` | `index.html` (the whole UI), `login.html`, `setup.html` |
| `tests/` | pytest suite, also run in CI |

## Things worth knowing

- **Qwen Image Edit stays faithful near 1 megapixel.** Asking for much more makes it
  re-compose the photo, so restores run at about 1 MP in the target's shape and are then
  enlarged slightly (`comfy.model_size`). Sizes must be multiples of 16.
- **The prompt lives in the workflow file**, not in the code. The app only changes the
  Load Image input, the resize step and the seed.
- **The GPU can be shared with another big program** (here a local LLM). Free memory is
  checked through ComfyUI before each restore; the helper pauses listed units only.
- **The Card layout and caption style were measured** from a hand-made GIMP card
  (1200×1541, caption cap height 31 px). Other layouts scale the caption by
  `min(width, height) / 1200`.
- **Framing**: the face box is a set share of the width and the eye line a set share of the
  height, per layout; custom sizes interpolate from their shape (`card.custom`).
- A restore takes about 5 minutes on an RTX 4000 SFF (40 steps).

## Working on it

```sh
docker build -t portrait:local .
docker run --rm -v "$PWD/tests:/srv/tests:ro" portrait:local \
  sh -c "pip install -q pytest && python -m pytest -q tests"
docker run --rm -p 8080:8000 -e ALLOW_NO_LOGIN=true portrait:local   # try it
```

- Every push to `main` runs the tests and a smoke test, then publishes
  `ghcr.io/therepos/portrait:latest`; deployments with Watchtower update themselves.
  Keep `main` releasable, and run the tests before pushing.
- Test against stand-ins rather than the real ComfyUI or systemd where possible; a real
  restore pauses whatever the GPU helper is set to pause.
- Commit messages: a short summary line, then what changed and why. Commit with the
  GitHub no-reply email, not a personal address.
- Pin dependency versions in `requirements.txt`.
