"""Container health check: the app answers on 8000, the GPU helper on 9000."""
import sys
import urllib.request

for port in (8000, 9000):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3)
        sys.exit(0)
    except Exception:
        pass
sys.exit(1)
