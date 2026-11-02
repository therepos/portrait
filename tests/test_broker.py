"""GPU helper: pause/resume leases against a fake systemd."""
import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from app import broker


class FakeSystemd:
    def __init__(self, states, fail_stop=False):
        self.states, self.fail_stop, self.calls = dict(states), fail_stop, []

    def state(self, unit):
        return self.states.get(unit, "inactive")

    def stop(self, unit):
        self.calls.append(("stop", unit))
        if self.fail_stop:
            raise RuntimeError("AccessDenied")
        self.states[unit] = "inactive"

    def start(self, unit):
        self.calls.append(("start", unit))
        self.states[unit] = "active"


@pytest.fixture
def helper(monkeypatch):
    def make(units=("llama.service",), **states_kw):
        fake = FakeSystemd(states_kw.get("states", {u: "active" for u in units}), states_kw.get("fail_stop", False))
        monkeypatch.setattr(broker, "UNITS", list(units))
        monkeypatch.setattr(broker, "_leases", {})
        monkeypatch.setattr(broker, "_stopped", [])
        monkeypatch.setattr(broker, "_lock", asyncio.Lock())
        for name in ("state", "stop", "start"):
            monkeypatch.setattr(broker.systemd, name, getattr(fake, name))
        return fake, TestClient(broker.app)
    return make


def test_pause_then_resume(helper):
    fake, client = helper()
    with client:
        lease = client.post("/pause").json()["lease"]
        assert fake.states["llama.service"] == "inactive"
        client.post("/resume", json={"lease": lease})
    assert fake.calls == [("stop", "llama.service"), ("start", "llama.service")]
    assert fake.states["llama.service"] == "active"


def test_already_stopped_unit_left_alone(helper):
    fake, client = helper(states={"llama.service": "inactive"})
    with client:
        lease = client.post("/pause").json()["lease"]
        client.post("/resume", json={"lease": lease})
    assert fake.calls == [] and fake.states["llama.service"] == "inactive"


def test_overlapping_restores_pause_once(helper):
    fake, client = helper()
    with client:
        a = client.post("/pause").json()["lease"]
        b = client.post("/pause").json()["lease"]
        client.post("/resume", json={"lease": a})
        assert fake.states["llama.service"] == "inactive"  # b still running
        client.post("/resume", json={"lease": b})
    assert fake.calls == [("stop", "llama.service"), ("start", "llama.service")]


def test_failed_stop_reports_and_leaves_unit_running(helper):
    fake, client = helper(fail_stop=True)
    with client:
        r = client.post("/pause")
    assert r.status_code == 500 and "Could not pause" in r.json()["detail"]
    assert fake.states["llama.service"] == "active"


def test_expired_lease_starts_unit(helper):
    fake, client = helper()
    with client:
        lease = client.post("/pause").json()["lease"]
        broker._leases[lease] = time.monotonic() - 1  # app died mid-restore
        asyncio.run(broker.expire_leases())
        assert fake.states["llama.service"] == "active"


def test_shutdown_starts_unit(helper):
    fake, client = helper()
    with client:
        client.post("/pause")
    assert fake.states["llama.service"] == "active"  # helper stopped: unit restarted


def test_only_listed_units(helper):
    fake, client = helper(units=())
    with client:
        assert client.post("/pause").status_code == 400
    assert fake.calls == []
