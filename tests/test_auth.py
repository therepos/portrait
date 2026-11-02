"""Single-account login, history ownership and the per-person queue limit."""
import pytest
from fastapi.testclient import TestClient

from app import auth, main, store


@pytest.fixture
def client(monkeypatch, tmp_path):
    def make(username="ryan", password="s3cret"):
        monkeypatch.setattr(store, "DATA_DIR", tmp_path)
        monkeypatch.setattr(store, "ITEMS", tmp_path / "items")
        monkeypatch.setattr(auth, "USERNAME", username)
        monkeypatch.setattr(auth, "PASSWORD", password)
        monkeypatch.setattr(auth, "_failures", {})

        async def no_wait(_):
            pass
        monkeypatch.setattr(auth.asyncio, "sleep", no_wait)
        return TestClient(main.app)  # no lifespan: the restore worker isn't needed here
    return make


def test_open_without_account(client):
    c = client(username="", password="")
    assert c.get("/api/items").status_code == 200
    assert c.get("/login", follow_redirects=False).status_code == 303


def test_everything_locked_until_signed_in(client):
    c = client()
    assert c.get("/api/items").status_code == 401
    r = c.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "login"
    assert c.get("/login").status_code == 200
    assert c.get("/healthz").status_code == 200


def test_wrong_then_right_password(client):
    c = client()
    assert c.post("/api/login", data={"username": "ryan", "password": "nope"}).status_code == 401
    assert c.post("/api/login", data={"username": "ryan", "password": "s3cret"}).status_code == 200
    assert c.get("/api/items").status_code == 200
    assert c.get("/api/config").json()["user"] == "ryan"


def test_sign_out(client):
    c = client()
    c.post("/api/login", data={"username": "ryan", "password": "s3cret"})
    c.post("/api/logout")
    assert c.get("/api/items").status_code == 401


def test_tampered_or_old_password_session_rejected(client, monkeypatch):
    c = client()
    c.post("/api/login", data={"username": "ryan", "password": "s3cret"})
    monkeypatch.setattr(auth, "PASSWORD", "changed")  # password change signs everyone out
    assert c.get("/api/items").status_code == 401
    c.cookies.set(auth.COOKIE, "ryan|9999999999|deadbeef")
    assert c.get("/api/items").status_code == 401


def test_lockout_after_repeated_wrong_passwords(client):
    c = client()
    for _ in range(auth.MAX_FAILURES):
        c.post("/api/login", data={"username": "ryan", "password": "nope"})
    r = c.post("/api/login", data={"username": "ryan", "password": "s3cret"})
    assert r.status_code == 429


def test_browser_history_moves_into_account(client):
    c = client(username="", password="")
    c.get("/api/items")                      # gets a browser id
    browser = c.cookies.get(main.COOKIE)
    item = store.create(browser, "card", name="old.png")
    c2 = client()
    c2.cookies.set(main.COOKIE, browser)
    c2.post("/api/login", data={"username": "ryan", "password": "s3cret"})
    ids = [i["id"] for i in c2.get("/api/items").json()["items"]]
    assert item["id"] in ids


def test_queue_limit(client, monkeypatch):
    c = client(username="", password="")
    monkeypatch.setattr(main.comfy, "enabled", lambda: True)
    c.get("/api/items")
    owner = c.cookies.get(main.COOKIE)
    for _ in range(main.MAX_QUEUED):
        store.create(owner, "restore", state="queued")
    r = c.post("/api/restores", files={"photo": ("a.png", b"x", "image/png")})
    assert r.status_code == 429 and "waiting" in r.json()["detail"]
