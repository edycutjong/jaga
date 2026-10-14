"""A fake LAN plug for tests and the CI variant of verify_offline.py — never used on the live path.

It speaks the Shelly Gen3 local RPC the real ShellyPlug driver uses (`/rpc/Switch.Set`, `/rpc/Switch.GetStatus`)
on 127.0.0.1, records every call, and can be told to fail so ACTION FAILED is testable.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


class FakePlug:
    def __init__(self, on: bool = True, fail_off: bool = False):
        self.on = on
        self.fail_off = fail_off
        self.calls: list[tuple[str, dict]] = []
        self.lock = threading.Lock()
        self.server: ThreadingHTTPServer | None = None

    @property
    def on_calls(self) -> int:
        return sum(1 for path, q in self.calls if path == "/rpc/Switch.Set" and q.get("on") == "true")

    @property
    def off_calls(self) -> int:
        return sum(1 for path, q in self.calls if path == "/rpc/Switch.Set" and q.get("on") == "false")

    def start(self) -> str:
        plug = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                with plug.lock:
                    plug.calls.append((u.path, q))
                    if u.path == "/rpc/Switch.Set":
                        want = q.get("on") == "true"
                        if not want and plug.fail_off:
                            self.send_response(500)
                            self.end_headers()
                            return
                        was = plug.on
                        plug.on = want
                        body = {"was_on": was}
                    elif u.path == "/rpc/Switch.GetStatus":
                        body = {"id": 0, "output": plug.on}
                    else:
                        self.send_response(404)
                        self.end_headers()
                        return
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def press_button(self) -> None:
        """The plug's own button: someone switches it back ON by hand."""
        with self.lock:
            self.on = True

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()


if __name__ == "__main__":  # manual QA: python tests/fakeplug.py → prints its URL and serves until Ctrl-C
    import time

    p = FakePlug()
    print(p.start(), flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        p.stop()
