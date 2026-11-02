"""Pause/resume of the GPU-holding service, against a fake systemd."""
import asyncio

import pytest

from app import gpu


class FakeUnit:
    def __init__(self, state="active", deny=False):
        self.state, self.deny, self.calls = state, deny, []

    def call(self, method, signature, args):
        self.calls.append(method)
        if self.deny:
            raise RuntimeError("org.freedesktop.DBus.Error.AccessDenied: Interactive authentication required.")
        self.state = {"StopUnit": "inactive", "StartUnit": "active"}[method]
        return ("/org/freedesktop/systemd1/job/1",)


@pytest.fixture
def unit(monkeypatch):
    def make(**kw):
        u = FakeUnit(**kw)
        monkeypatch.setattr(gpu, "UNIT", "llama.service")
        monkeypatch.setattr(gpu, "HEALTH_URL", "")
        monkeypatch.setattr(gpu, "_call", u.call)
        monkeypatch.setattr(gpu, "_state", lambda: u.state)
        monkeypatch.setattr(gpu, "_users", 0)
        monkeypatch.setattr(gpu, "_paused", False)
        monkeypatch.setattr(gpu, "_lock", asyncio.Lock())
        return u
    return make


def run(coro):
    return asyncio.run(coro)


def test_stops_then_starts(unit):
    u = unit()

    async def go():
        await gpu.pause(None)
        assert u.state == "inactive"
        await gpu.resume()

    run(go())
    assert u.calls == ["StopUnit", "StartUnit"] and u.state == "active"


def test_restarts_when_restore_fails(unit):
    u = unit()

    async def go():
        await gpu.pause(None)
        try:
            raise RuntimeError("ComfyUI crashed")
        except RuntimeError:
            pass
        finally:
            await gpu.resume()

    run(go())
    assert u.state == "active"


def test_leaves_already_stopped_unit_alone(unit):
    u = unit(state="inactive")

    async def go():
        await gpu.pause(None)
        await gpu.resume()

    run(go())
    assert u.calls == [] and u.state == "inactive"


def test_overlapping_restores_pause_once(unit):
    u = unit()

    async def go():
        await gpu.pause(None)
        await gpu.pause(None)
        await gpu.resume()
        assert u.state == "inactive"  # second restore still running
        await gpu.resume()

    run(go())
    assert u.calls == ["StopUnit", "StartUnit"] and u.state == "active"


def test_permission_denied_is_reported_and_unit_untouched(unit):
    u = unit(deny=True)
    with pytest.raises(gpu.PauseError, match="Not allowed"):
        run(gpu.pause(None))
    assert u.state == "active" and u.calls == ["StopUnit"] and gpu._users == 0


def test_shutdown_while_paused_starts_unit(unit):
    u = unit()

    async def go():
        await gpu.pause(None)
        await gpu.shutdown()

    run(go())
    assert u.state == "active"
