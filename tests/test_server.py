"""The local server's request guard and endpoints (FastAPI TestClient; no network, no model)."""
import pytest
from conftest import make_watcher, push_windows
from fastapi.testclient import TestClient

from jaga.server import create_app

HOST = {"host": "127.0.0.1:8765"}


@pytest.fixture
def app_client(home, tuning, fakeplug):
    w, emb = make_watcher(home, tuning, fakeplug.url)
    calls = []

    def fake_compile(sentence):
        calls.append(sentence)
        from conftest import SPEC

        return {"spec": {**SPEC, "action": "notify"}, "tf_request_id": "req-test", "tf_response_id": "r",
                "model": "nvidia/nemotron-3-super-120b-a12b", "created": "now", "temperature": 1.0, "latency_s": 0.1}

    app = create_app(w, compile_fn=fake_compile, embed_text_fn=emb.embed_text, token="tok-123")
    c = TestClient(app, base_url="http://127.0.0.1:8765")
    c.watcher, c.compiles, c.plug = w, calls, fakeplug
    return c


def post(c, path, body=None, **headers):
    return c.post(path, json=body or {}, headers={**HOST, "x-jaga-session": "tok-123", **headers})


def test_page_embeds_the_session_token_and_csp(app_client):
    r = app_client.get("/", headers=HOST)
    assert r.status_code == 200 and 'content="tok-123"' in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]


def test_cross_origin_post_is_refused(app_client):
    r = post(app_client, "/plug/test", origin="http://evil.example")
    assert r.status_code == 403 and app_client.plug.on_calls == 0


def test_post_without_token_is_refused(app_client):
    r = app_client.post("/plug/test", headers=HOST)
    assert r.status_code == 403 and app_client.plug.on_calls == 0


def test_dns_rebinding_host_is_refused_even_for_get(app_client):
    assert app_client.get("/api/state", headers={"host": "evil.example:8765"}).status_code == 403
    assert app_client.get("/stream", headers={"host": "attacker.test"}).status_code == 403


def test_cross_site_fetch_metadata_is_refused(app_client):
    assert post(app_client, "/labels", {"kind": "note"}, **{"sec-fetch-site": "cross-site"}).status_code == 403


def test_same_origin_with_token_is_allowed(app_client):
    app_client.watcher.labels = True  # a session started with --labels (or a soak)
    r = post(app_client, "/labels", {"kind": "note", "note": "hello"}, origin="http://127.0.0.1:8765")
    assert r.status_code == 200 and r.json()["ok"]


def test_labels_are_refused_when_the_session_has_them_off(app_client):
    r = post(app_client, "/labels", {"kind": "exit_leave"}, origin="http://127.0.0.1:8765")
    assert r.status_code == 409 and "--labels" in r.json()["detail"]


def test_arm_rejects_bad_numbers_with_422_never_500(app_client):
    for body in ([1, 2], {"capture_s": "nan"}, {"capture_s": -5}, {"capture_s": [120]}, {"dwell_s": {"x": 1}},
                 {"dwell_s": 1.0}, {"action": "plug_on"}):
        r = app_client.post("/tripwires/01/arm", json=body, headers={**HOST, "x-jaga-session": "tok-123"})
        assert r.status_code == 422, (body, r.status_code, r.text)
    assert app_client.plug.on_calls == 0


def test_an_explicit_slot_cannot_overwrite_a_frozen_tripwire(app_client):
    r = post(app_client, "/tripwires", {"sentence": "the kettle is left on", "slot": "01"})
    assert r.status_code == 409 and app_client.compiles == []


def test_compiler_down_only_when_every_attempt_failed_in_transport():
    from jaga.compile import CompileError
    from jaga.server import _unreachable

    down = CompileError("compile_failed", "x", [{"shape": "openai", "ok": False, "error": "APIConnectionError: boom"},
                                               {"shape": "json_object", "ok": False,
                                                "error": "InternalServerError: Error code: 503 - busy"}])
    answered = CompileError("compile_failed", "x", [{"shape": "openai", "ok": False, "response_id": "r1",
                                                     "error": "ValidationError: field 504 too long"}])
    assert _unreachable(down) and not _unreachable(answered)
    assert _unreachable(CompileError("no_key", "no key"))


def test_plug_test_is_the_operator_on_path(app_client):
    app_client.plug.on = False
    r = post(app_client, "/plug/test")
    assert r.status_code == 200 and r.json()["ok"]
    assert app_client.plug.on_calls == 1 and app_client.plug.on is True
    kinds = [e["kind"] for e in app_client.watcher.db.recent_events(5, "test")]
    assert "operator_on" in kinds


def test_compile_then_arm_with_user_choice(app_client):
    r = post(app_client, "/tripwires", {"sentence": "the door is left open and nobody is there"})
    assert r.status_code == 200 and r.json()["state"] == "compiled"
    slot = r.json()["slot"]
    r = post(app_client, f"/tripwires/{slot}/arm", {"capture_s": 30, "action": "notify", "dwell_s": 60})
    assert r.status_code == 200
    tw = app_client.watcher.tripwires[slot]
    assert tw.state == "capturing" and tw.dwell_s == 60 and tw.action == "notify"


def test_state_and_receipts(app_client):
    w = app_client.watcher
    w.arm("01", 0)
    push_windows(w, w.embedder, "neg", 2, 100.0)
    st = app_client.get("/api/state", headers=HOST).json()
    assert st["snapshot"]["windows"][0]["slot"] == "01" and "receipt" in st
    r = app_client.get("/receipts", headers=HOST)
    assert r.status_code == 200 and "Token Factory" in r.text and "Serverless Jobs" in r.text
    j = app_client.get("/receipts?format=json", headers=HOST).json()
    assert j["tripwires"][0]["tf_request_id"] == "test"


def test_wrong_mark_writes_a_label(app_client):
    w = app_client.watcher
    eid = w.db.add_event("trip", "test trip", w.tripwires["01"].tripwire_id, session="test")
    assert post(app_client, f"/events/{eid}/wrong").status_code == 200
    assert w.db.query("SELECT wrong FROM event WHERE id=?", (eid,))[0]["wrong"] == 1
    assert w.db.query("SELECT kind FROM label WHERE kind='wrong'")
