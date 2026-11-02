"""Optional single-account login.

Set AUTH_USERNAME and AUTH_PASSWORD to require signing in. The session is a
signed cookie (no server-side state); changing the password signs everyone out.
Without them the app only shows a setup notice, unless ALLOW_NO_LOGIN=true makes
it open to anyone who can reach it, with history kept per browser.
"""
import asyncio
import hashlib
import hmac
import os
import secrets
import time

from . import store

USERNAME = os.environ.get("AUTH_USERNAME") or ""
PASSWORD = os.environ.get("AUTH_PASSWORD") or ""
# Without a username and password the app refuses to run, unless running open is chosen on purpose
ALLOW_OPEN = (os.environ.get("ALLOW_NO_LOGIN") or "").lower() in ("1", "true", "yes")
SESSION_DAYS = 30
COOKIE = "portrait_session"
MAX_FAILURES = 10          # per address, then locked out for LOCKOUT seconds
LOCKOUT = 15 * 60

_failures: dict = {}       # address -> (count, first failure time)


def enabled():
    return bool(USERNAME and PASSWORD)


def setup_needed():
    return not enabled() and not ALLOW_OPEN


def _secret():
    """Signing key: a random value kept in the data folder, mixed with the password."""
    path = store.DATA_DIR / "secret"
    try:
        key = path.read_bytes()
    except OSError:
        key = secrets.token_bytes(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
        path.chmod(0o600)
    return hashlib.sha256(key + PASSWORD.encode()).digest()


def _sign(text):
    return hmac.new(_secret(), text.encode(), hashlib.sha256).hexdigest()


def make_session(username):
    expires = int(time.time()) + SESSION_DAYS * 86400
    body = f"{username}|{expires}"
    return f"{body}|{_sign(body)}", SESSION_DAYS * 86400


def session_user(cookie):
    """Username for a valid session cookie, else None."""
    try:
        username, expires, sig = cookie.rsplit("|", 2)
    except (AttributeError, ValueError):
        return None
    if not hmac.compare_digest(sig, _sign(f"{username}|{expires}")):
        return None
    if not expires.isdigit() or int(expires) < time.time():
        return None
    return username


def _same(a, b):
    return hmac.compare_digest(hashlib.sha256(a.encode()).digest(), hashlib.sha256(b.encode()).digest())


def _locked(key, limit):
    count, first = _failures.get(key, (0, 0))
    if time.time() - first > LOCKOUT:
        _failures.pop(key, None)
        return False
    return count >= limit


def _fail(key):
    count, first = _failures.get(key, (0, time.time()))
    if time.time() - first > LOCKOUT:
        count, first = 0, time.time()
    _failures[key] = (count + 1, first)
    return count


async def check(address, username, password):
    """True if the username and password match, False if not, None while locked out.

    Each address gets MAX_FAILURES tries per LOCKOUT period, and all addresses
    together five times that, so faked addresses don't help a guesser."""
    if _locked(address, MAX_FAILURES) or _locked("*", MAX_FAILURES * 5):
        return None
    if _same(username, USERNAME) and _same(password, PASSWORD):
        _failures.pop(address, None)
        return True
    count = _fail(address)
    _fail("*")
    await asyncio.sleep(1 + count)  # each wrong guess waits a little longer
    return False
