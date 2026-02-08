import threading
import time
import requests
import sys

def start_keep_alive(
    url="http://127.0.0.1:8000/health",
    interval=240  # 4 minuter
):
    def _ping_loop():
        while True:
            try:
                requests.get(url, timeout=5)
                print("KEEP-ALIVE ping", file=sys.stderr, flush=True)
            except Exception as e:
                print("KEEP-ALIVE failed:", e, file=sys.stderr, flush=True)

            time.sleep(interval)

    threading.Thread(
        target=_ping_loop,
        daemon=True
    ).start()
