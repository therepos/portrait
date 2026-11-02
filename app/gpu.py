"""Optional: stop a systemd service that holds the GPU while ComfyUI restores.

Set PAUSE_UNIT (e.g. llama.service) and mount the host's
/run/dbus/system_bus_socket. The container user must be allowed to start and
stop that unit (a polkit rule; see README). The unit is always started again
afterwards, also when the restore fails or the app shuts down, unless it was
already stopped before the app paused it.
"""
import asyncio
import logging
import os
import time

import httpx

UNIT = os.environ.get("PAUSE_UNIT") or ""
HEALTH_URL = os.environ.get("PAUSE_HEALTH_URL") or ""
BUS = os.environ.get("PAUSE_BUS", "SYSTEM").upper()
STOP_WAIT = 90     # seconds for the unit to stop and its GPU memory to clear
START_WAIT = 300   # seconds for the unit to come back healthy
FREE_VRAM_GB = float(os.environ.get("PAUSE_FREE_VRAM_GB", "15"))

log = logging.getLogger("portrait.gpu")
_lock = asyncio.Lock()
_users = 0          # restores currently relying on the pause
_paused = False     # we stopped the unit and owe it a start


class PauseError(RuntimeError):
    pass


def enabled():
    return bool(UNIT)


def _systemd(method, signature, args, conn):
    from jeepney import DBusAddress, new_method_call
    from jeepney.wrappers import unwrap_msg

    manager = DBusAddress("/org/freedesktop/systemd1", bus_name="org.freedesktop.systemd1",
                          interface="org.freedesktop.systemd1.Manager")
    return unwrap_msg(conn.send_and_get_reply(new_method_call(manager, method, signature, args)))


def _call(method, signature, args):
    from jeepney.io.blocking import open_dbus_connection

    with open_dbus_connection(bus=BUS) as conn:
        return _systemd(method, signature, args, conn)


def _state():
    from jeepney import DBusAddress, new_method_call
    from jeepney.io.blocking import open_dbus_connection
    from jeepney.wrappers import unwrap_msg

    with open_dbus_connection(bus=BUS) as conn:
        (path,) = _systemd("LoadUnit", "s", (UNIT,), conn)
        props = DBusAddress(path, bus_name="org.freedesktop.systemd1", interface="org.freedesktop.DBus.Properties")
        (value,) = unwrap_msg(conn.send_and_get_reply(
            new_method_call(props, "Get", "ss", ("org.freedesktop.systemd1.Unit", "ActiveState"))))
        return value[1]


async def state():
    return await asyncio.to_thread(_state)


async def _wait_vram(comfy_url):
    """Wait until ComfyUI sees enough free GPU memory (best effort)."""
    if not comfy_url:
        return
    async with httpx.AsyncClient(base_url=comfy_url, timeout=10) as http:
        deadline = time.monotonic() + STOP_WAIT
        while time.monotonic() < deadline:
            try:
                devices = (await http.get("/system_stats")).json()["devices"]
                if devices and devices[0]["vram_free"] >= FREE_VRAM_GB * 2**30:
                    return
            except (httpx.HTTPError, KeyError, ValueError):
                pass
            await asyncio.sleep(2)
    log.warning("GPU memory still not free after %s s; trying anyway", STOP_WAIT)


async def pause(comfy_url):
    """Stop the unit (if running) before a restore."""
    global _users, _paused
    async with _lock:
        _users += 1
        if _paused:
            return
        try:
            current = await state()
            if current not in ("active", "activating", "reloading"):
                return  # already stopped by someone else: leave it that way
            log.info("stopping %s to free the GPU", UNIT)
            await asyncio.to_thread(_call, "StopUnit", "ss", (UNIT, "replace"))
            _paused = True
            deadline = time.monotonic() + STOP_WAIT
            while await state() not in ("inactive", "failed"):
                if time.monotonic() > deadline:
                    raise PauseError(f"{UNIT} did not stop within {STOP_WAIT} s.")
                await asyncio.sleep(1)
        except PauseError:
            _users -= 1
            await _start_locked()
            raise
        except Exception as e:
            _users -= 1
            await _start_locked()
            denied = "AccessDenied" in repr(e) or "authentication" in str(e).lower()
            raise PauseError(f"Not allowed to stop {UNIT}; see the README for the permission rule." if denied
                             else f"Could not stop {UNIT} ({e.__class__.__name__}).")
    await _wait_vram(comfy_url)


async def resume():
    """Release one restore's hold; start the unit when the last one finishes."""
    global _users
    async with _lock:
        _users = max(0, _users - 1)
        if _users == 0:
            await _start_locked()


async def _start_locked():
    global _paused
    if not _paused:
        return
    log.info("starting %s again", UNIT)
    for attempt in range(3):
        try:
            await asyncio.to_thread(_call, "StartUnit", "ss", (UNIT, "replace"))
            _paused = False
            break
        except Exception as e:
            log.error("could not start %s (attempt %d): %r", UNIT, attempt + 1, e)
            await asyncio.sleep(5)
    if _paused:
        return
    if HEALTH_URL:
        asyncio.create_task(_report_health())


async def _report_health():
    deadline = time.monotonic() + START_WAIT
    async with httpx.AsyncClient(timeout=5) as http:
        while time.monotonic() < deadline:
            try:
                if (await http.get(HEALTH_URL)).status_code == 200:
                    log.info("%s is healthy again", UNIT)
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(3)
    log.error("%s did not report healthy within %s s", UNIT, START_WAIT)


async def shutdown():
    """App is stopping: never leave the unit stopped."""
    global _users
    async with _lock:
        _users = 0
        await _start_locked()
