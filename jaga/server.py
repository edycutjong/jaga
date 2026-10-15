"""The overlay's local server: FastAPI on 127.0.0.1:8765 only, one static page, one SSE stream.

Request guard (architecture: Endpoints):
* every request — `Host` must be 127.0.0.1:<port> or localhost:<port>, so a DNS-rebinding page can neither
  read the camera stream nor post;
* every POST — `Origin` absent or exactly this origin, and the `X-Jaga-Session` token minted at start and
  embedded only in the served page. This matters because `POST /plug/test` is the one path that switches
  mains ON (operator-only, invariant I1).
"""
from __future__ import annotations

import asyncio
import html
import json
import math
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config, receipts

STATIC = Path(__file__).resolve().parent / "static"
SLOT_RE = re.compile(r"\d{2}")


def _default(o):
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def clean(o):
    """Non-finite floats → null, recursively (np.float64 subclasses float, so `default=` never sees it)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


def dumps(obj) -> str:
    return json.dumps(clean(obj), default=_default, allow_nan=False, ensure_ascii=False)


class Broadcaster:
    """Thread → asyncio fan-out of SSE messages. Slow clients drop their oldest message, never block."""

    def __init__(self, maxsize: int = 32):
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set[asyncio.Queue] = set()
        self.maxsize = maxsize

    def publish(self, kind: str, payload: dict) -> None:
        loop = self.loop
        if loop is None or not self.clients:
            return
        try:
            data = dumps(payload)
        except (TypeError, ValueError):
            return  # never put anything but valid JSON on the wire
        for q in list(self.clients):
            loop.call_soon_threadsafe(self._put, q, (kind, data))

    def _put(self, q: asyncio.Queue, item) -> None:
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        q.put_nowait(item)


def create_app(watcher, *, compile_fn=None, embed_text_fn=None, broadcaster: Broadcaster | None = None,
               port: int = config.PORT, token: str | None = None, info: dict | None = None) -> FastAPI:
    token = token or secrets.token_urlsafe(24)
    bc = broadcaster or Broadcaster()
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    @asynccontextmanager
    async def lifespan(_app):
        bc.loop = asyncio.get_running_loop()
        yield

    app = FastAPI(title="jaga", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.token = token
    app.state.broadcaster = bc
    app.state.info = info or {}

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = request.headers.get("host", "")
        if host not in allowed_hosts:
            return JSONResponse({"error": "forbidden host"}, status_code=403)
        if request.method not in ("GET", "HEAD"):
            origin = request.headers.get("origin")
            if origin is not None and origin != f"http://{host}":
                return JSONResponse({"error": "cross-origin request refused"}, status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse({"error": "cross-site request refused"}, status_code=403)
            given = request.headers.get("x-jaga-session", "").encode("latin-1", "replace")
            if not secrets.compare_digest(given, token.encode()):
                return JSONResponse({"error": "missing or wrong session token"}, status_code=403)
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        page = (STATIC / "index.html").read_text()
        page = page.replace("{{TOKEN}}", html.escape(token)).replace("{{VERSION}}", html.escape(config_version()))
        return HTMLResponse(page, headers={"Content-Security-Policy": CSP})

    @app.get("/api/state")
    async def state() -> Response:
        def build() -> str:  # off the event loop: it waits on the watcher lock and reads the DB
            return dumps({"snapshot": watcher.snapshot(include_frames=True), "traces": watcher.traces(),
                          "events": watcher.db.recent_events(8, watcher.session), "info": app.state.info,
                          "receipt": receipts.footer_line()})

        return Response(await asyncio.to_thread(build), media_type="application/json")

    @app.get("/stream")
    async def stream(request: Request) -> StreamingResponse:
        q: asyncio.Queue = asyncio.Queue(maxsize=bc.maxsize)

        async def gen():
            try:
                bc.clients.add(q)  # registered only once streaming starts, so `finally` always removes it
                yield "retry: 1500\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        kind, data = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"event: {kind}\ndata: {data}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                bc.clients.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})

    def _slot_free() -> str:
        """The first slot not loaded and not a frozen fixture's (a typed sentence never overwrites 01–03)."""
        from .tripwires import FIXTURES

        used = set(watcher.tripwires) | {p.stem for p in FIXTURES.glob("[0-9][0-9].json")}
        for i in range(1, 100):
            s = f"{i:02d}"
            if s not in used:
                return s
        raise HTTPException(409, "no free slot")

    @app.post("/tripwires")
    async def add_tripwire(request: Request) -> Response:
        body = await _json_or_422(request)
        sentence = " ".join(str(body.get("sentence", "")).split())
        if not sentence:
            raise HTTPException(422, "type a sentence first")
        if len(sentence) > 240:
            raise HTTPException(422, "keep it to one sentence (240 characters)")
        slot = str(body.get("slot") or _slot_free()).zfill(2)
        if not SLOT_RE.fullmatch(slot):
            raise HTTPException(422, "slot must be one or two digits")
        if body.get("slot") and _fixture_slot(slot):
            raise HTTPException(409, f"slot {slot} holds a frozen tripwire — leave the slot empty")
        if compile_fn is None or embed_text_fn is None:
            raise HTTPException(503, "compiler not configured")
        try:
            res = await asyncio.to_thread(compile_fn, sentence)
            watcher.compiler_down = False
        except Exception as e:  # TF down or no valid spec: armed tripwires keep running
            watcher.compiler_down = _unreachable(e)
            cached = watcher.db.cached_spec(sentence)
            if cached is None:
                watcher.db.log_compile(sentence, False, f"{type(e).__name__}: {str(e)[:300]}")
                return Response(dumps({"ok": False, "state": "compile-failed",
                                       "detail": "Token Factory unreachable or no valid spec — new sentences wait; "
                                                 "armed tripwires keep running",
                                       "error": str(e)[:300]}), status_code=502, media_type="application/json")
            res = cached
        spec = res["spec"]
        watcher.db.log_compile(sentence, True, "cached" if res.get("cached") else "", res)
        if not spec.get("visual", False):
            return Response(dumps({"ok": True, "state": "not-visual", "slot": slot, "spec": spec,
                                   "tf_request_id": res.get("tf_request_id"), "model": res.get("model")}),
                            media_type="application/json")
        pos = await asyncio.to_thread(embed_text_fn, spec["positives"])
        neg = await asyncio.to_thread(embed_text_fn, spec["hard_negatives"])
        from .tripwires import fresh_threshold

        await asyncio.to_thread(watcher.load_tripwire, slot, sentence, spec, pos, neg, fresh_threshold(watcher.t),
                                {k: res.get(k) for k in ("tf_request_id", "tf_response_id", "model", "created",
                                                         "temperature")} | {"compiled_live": not res.get("cached")})
        return Response(dumps({"ok": True, "state": "cached" if res.get("cached") else "compiled", "slot": slot,
                               "spec": spec, "tf_request_id": res.get("tf_request_id"), "model": res.get("model"),
                               "latency_s": res.get("latency_s")}), media_type="application/json")

    @app.post("/tripwires/{slot}/arm")
    async def arm(slot: str, request: Request) -> dict:
        body = await _json_or_empty(request)
        cap = _number(body.get("capture_s"), "capture_s", 0.0, 600.0)
        dwell = _number(body.get("dwell_s"), "dwell_s", 2.0, 600.0)
        action = body.get("action")
        if action is not None and action not in ("plug_off", "notify"):
            raise HTTPException(422, "action must be plug_off or notify")
        try:
            if action is not None or dwell is not None:
                await asyncio.to_thread(watcher.set_operating, slot, action, dwell)
            await asyncio.to_thread(watcher.arm, slot, cap)
        except (KeyError, ValueError) as e:
            raise HTTPException(404 if isinstance(e, KeyError) else 409, str(e)) from e
        return {"ok": True}

    @app.post("/tripwires/{slot}/disarm")
    async def disarm(slot: str) -> dict:
        try:
            await asyncio.to_thread(watcher.disarm, slot)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        return {"ok": True}

    @app.post("/tripwires/{slot}/example")
    async def example(slot: str) -> dict:
        try:
            return await asyncio.to_thread(watcher.add_example, slot)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/plug/test")
    async def plug_test() -> Response:
        from .act import ActionFailed

        try:
            res = await asyncio.to_thread(watcher.plug_test)
        except ActionFailed as e:
            return Response(dumps({"ok": False, "detail": str(e)[:300]}), status_code=502,
                            media_type="application/json")
        return Response(dumps({"ok": True, **res}), media_type="application/json")

    @app.post("/events/{event_id}/wrong")
    async def wrong(event_id: int) -> dict:
        ok = await asyncio.to_thread(watcher.db.mark_wrong, event_id)
        if not ok:
            raise HTTPException(404, "no such event")
        watcher.label("wrong", f"event {event_id}")
        return {"ok": True}

    @app.post("/labels")
    async def label(request: Request) -> dict:
        body = await _json_or_422(request)
        if not watcher.labels:
            raise HTTPException(409, "labels are off in this session (start jaga with --labels)")
        kind = str(body.get("kind", "")).strip()[:40]
        if not kind:
            raise HTTPException(422, "kind required")
        lid = await asyncio.to_thread(watcher.label, kind, str(body.get("note", ""))[:200])
        return {"ok": True, "id": lid}

    @app.get("/receipts", response_class=HTMLResponse)
    async def receipts_view(request: Request):
        data = await asyncio.to_thread(receipts.collect, watcher)
        if request.query_params.get("format") == "json":
            return Response(dumps(data), media_type="application/json")
        return HTMLResponse(receipts.render_html(data), headers={"Content-Security-Policy": CSP})

    return app


CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; font-src 'self'; "
       "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def config_version() -> str:
    from . import __version__

    return __version__


TRANSPORT = re.compile(r"^(APIConnectionError|APITimeoutError|ConnectError|ConnectTimeout|ReadTimeout|"
                       r"TimeoutException|TimeoutError)\b|^\w+: Error code: 50[234]\b")


def _unreachable(e: Exception) -> bool:
    """Token Factory down or unreachable — every attempt failed in transport — as opposed to a sentence that got an
    answer but no valid spec (then the compiler is up, and the header must not say otherwise)."""
    if getattr(e, "kind", "") == "no_key":
        return True
    attempts = getattr(e, "attempts", []) or []
    if not attempts:
        return bool(TRANSPORT.match(f"{type(e).__name__}: {e}"))
    return all(not a.get("response_id") and TRANSPORT.match(str(a.get("error", ""))) for a in attempts)


def _fixture_slot(slot: str) -> bool:
    from .tripwires import FIXTURES

    return (FIXTURES / f"{slot}.json").exists()


def _number(v, name: str, lo: float, hi: float) -> float | None:
    """A JSON number (or numeric string) in [lo, hi]; None stays None; anything else is a 422, never a 500."""
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise HTTPException(422, f"{name} must be a number")
    try:
        x = float(v)
    except ValueError:
        raise HTTPException(422, f"{name} must be a number") from None
    if not math.isfinite(x) or not lo <= x <= hi:
        raise HTTPException(422, f"{name} must be between {lo:g} and {hi:g}")
    return x


async def _json_or_422(request: Request) -> dict:
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(422, "the body must be JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(422, "the body must be a JSON object")
    return body


async def _json_or_empty(request: Request) -> dict:
    """An optional JSON object body: empty → {}; anything that is not a JSON object → 422."""
    raw = await request.body()
    if not raw:
        return {}
    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(422, "the body must be JSON") from None
    if not isinstance(body, dict):
        raise HTTPException(422, "the body must be a JSON object")
    return body


def run(app: FastAPI, port: int = config.PORT, on_ready=None) -> None:
    """Serve on 127.0.0.1. `on_ready` runs once the socket is bound; `app.state.server.should_exit = True` stops it.
    An open overlay tab keeps an event stream open, so shutdown waits at most 2 s for it (Ctrl-C works)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(app, host=config.HOST, port=port, log_level="warning", access_log=False,
                                           timeout_graceful_shutdown=2))
    app.state.server = server
    if on_ready:
        def wait_ready():
            while not server.started and not server.should_exit:
                time.sleep(0.1)
            if server.started:
                on_ready()

        threading.Thread(target=wait_ready, daemon=True, name="jaga-ready").start()
    server.run()


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/api/state", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False
