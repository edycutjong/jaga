"""Regression tests from the architecture and QA reviews: acknowledgement, liveness, plug I/O ordering, guards,
the I2 floor, non-finite values, the Job dispatcher, the action allow-list, thread hygiene, OFF retries that never
give up, session names, the label keys, rotation of phone video and arming without a normal state."""
import json
import math
import subprocess
import threading
import time

import numpy as np
import pytest
from conftest import Captions, make_watcher, push_windows, wait_for, write_test_video

from jaga.act import Ack, ActionFailed, Notifier, PlugDriver


class SlowPlug(PlugDriver):
    """A driver whose calls can be held open, to reproduce the races the driver lock prevents."""

    kind, label, address = "test", "Wi-Fi plug", "test"

    def __init__(self, on=True):
        self.on = on
        self.gate = threading.Event()
        self.gate.set()
        self.status_delay = 0.0
        self.calls = []
        self.fail = None

    def turn_off(self):
        self.calls.append("off")
        self.gate.wait(5)
        if self.fail:
            raise self.fail
        self.on = False
        return Ack(True, 1.0, "test ack")

    def status(self):
        seen = self.on
        time.sleep(self.status_delay)
        self.calls.append("status")
        return seen

    def _device(self):  # used by operator_on for Tuya only; never here
        raise AssertionError


def watcher_with(driver, home, tuning, **kw):
    w, emb = make_watcher(home, tuning, None, **kw)
    w.driver = driver
    w.plug.update(driver.describe())
    return w, emb


# ------------------------------------------------------------------ P0-1: per-trip acknowledgement (I3)
def test_a_cached_off_never_stands_in_for_this_trips_acknowledgement(home, tuning):
    plug = SlowPlug(on=False)  # take 1 ended with the lamp cut: the startup read caches "off"
    w, emb = watcher_with(plug, home, tuning)
    w.start()
    try:
        assert wait_for(lambda: w.plug["state"] == "off")
        plug.on = True  # the operator switched the lamp back on by hand
        plug.gate.clear()  # this trip's OFF will hang until we release it
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        row = w.snapshot()["windows"][0]
        assert row["state"] == "tripped" and row["off_acked"] is False
        assert w.plug["state"] == "unknown"  # the cached "off" was demoted at the trip
        plug.gate.set()
        assert wait_for(lambda: w.snapshot()["windows"][0]["off_acked"] is True)
        assert w.plug["state"] == "off"
    finally:
        plug.gate.set()
        w.stop()


def test_a_failed_off_is_never_acknowledged(home, tuning):
    plug = SlowPlug()
    plug.fail = ActionFailed("the plug is not reachable on the home network")
    w, emb = watcher_with(plug, home, tuning)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.plug["failed"])
        assert w.snapshot()["windows"][0]["off_acked"] is False
    finally:
        w.stop()


# ------------------------------------------------------------------ P0-2: liveness
def test_the_action_worker_survives_an_unexpected_driver_error(home, tuning):
    plug = SlowPlug()
    plug.fail = AttributeError("'NoneType' object has no attribute 'get'")
    w, emb = watcher_with(plug, home, tuning)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: any(e["kind"] == "action_failed" for e in w.db.recent_events(10, "test")))
        assert w.plug["failed"] and "unexpected error" in w.plug["detail"]
        act = next(t for t in w.threads if t.name == "jaga-act")
        assert act.is_alive()
        plug.fail = None  # the plug recovers: the OFF retry succeeds
        assert wait_for(lambda: w.plug["state"] == "off", timeout=5)
    finally:
        w.stop()


def test_failing_windows_stop_the_watcher_and_do_not_move_the_heartbeat(home, tuning):
    w, emb = make_watcher(home, tuning, None, offline=False)
    w.arm("01", 0)
    push_windows(w, emb, "neg", 2, 0.0)
    beat = w.last_window_mono

    def boom(*a, **k):
        raise RuntimeError("database is locked")

    w.db.add_win = boom
    w.start()
    try:
        for i in range(8):
            w.ring.push(10.0 + i, np.zeros((90, 160, 3), np.uint8))
        for i in range(3):
            w.ring.push(20.0 + i, np.zeros((90, 160, 3), np.uint8))
            time.sleep(0.05)
        assert wait_for(lambda: not w.watcher_ok)
        assert w.last_window_mono == beat
        assert sum(e["kind"] == "watcher_stopped" for e in w.db.recent_events(20, "test")) == 1
    finally:
        w.stop()


def test_the_watchdog_reports_a_dead_thread(home, tuning):
    w, _ = make_watcher(home, tuning, None)
    w.start()
    try:
        dead = threading.Thread(target=lambda: None, name="jaga-act-test")
        dead.start()
        dead.join()
        w.threads.append(dead)
        assert wait_for(lambda: not w.watcher_ok, timeout=4)
        assert any("thread ended" in e["detail"] for e in w.db.recent_events(10, "test"))
    finally:
        w.stop()


def test_a_non_finite_embedding_fails_the_window_instead_of_scoring_it(home, tuning):
    w, emb = make_watcher(home, tuning, None)
    w.arm("01", 0)
    emb.embed_video = lambda frames: np.full(256, np.nan, np.float32)
    from jaga.capture import Window

    with pytest.raises(ValueError, match="non-finite"):
        w.process(Window(seq=1, ts=list(range(8)), frames=np.zeros((8, 90, 160, 3), np.uint8)))


def test_queued_offs_still_run_when_the_watcher_stops(home, tuning):
    plug = SlowPlug()
    w, emb = watcher_with(plug, home, tuning)
    w.arm("01", 0)
    push_windows(w, emb, "pos", 10, 100.0)  # the trip queues an OFF; no worker is running yet
    w.start()
    w.stop()
    assert plug.calls.count("off") == 1 and plug.on is False


# ------------------------------------------------------------------ P0-3: plug I/O ordering
def test_a_status_read_from_before_the_off_never_rearms(home, tuning):
    plug = SlowPlug(on=True)
    plug.status_delay = 0.4  # a slow status read that saw the plug ON just before the trip
    w, emb = watcher_with(plug, home, tuning)
    w.start()
    try:
        w.arm("01", 0)
        time.sleep(0.15)  # a status read is in flight
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.plug["state"] == "off" and w.tripwires["01"].off_acked)
        time.sleep(1.0)
        kinds = [e["kind"] for e in w.db.recent_events(30, "test")]
        assert "rearmed" not in kinds and w.tripwires["01"].state == "tripped" and w.plug["state"] == "off"
    finally:
        w.stop()


def test_a_trip_during_the_operator_test_still_ends_off(home, tuning, fakeplug, monkeypatch):
    from jaga import operator_on

    mid, go = threading.Event(), threading.Event()

    def paused_test(driver):  # the operator test, held between its OFF and its ON
        off = driver.turn_off()
        mid.set()
        go.wait(5)
        on = operator_on.switch_on(driver)
        return {"off_ms": off.latency_ms, "on_ms": on.latency_ms}

    monkeypatch.setattr(operator_on, "plug_test", paused_test)
    w, emb = make_watcher(home, tuning, fakeplug.url)
    w.start()
    try:
        w.arm("01", 0)
        t = threading.Thread(target=w.plug_test)
        t.start()
        assert mid.wait(5)
        push_windows(w, emb, "pos", 10, 100.0)  # the trip's OFF queues behind the test's lock
        go.set()
        t.join(5)
        assert wait_for(lambda: fakeplug.on is False and w.tripwires["01"].off_acked)  # the cut is the last word
        assert fakeplug.on_calls == 1  # the operator's own ON, and nothing else
    finally:
        go.set()
        w.stop()


# ------------------------------------------------------------------ P1-1: one arm gate
def test_nothing_arms_in_the_dark(home, tuning):
    w, emb = make_watcher(home, tuning, None)
    w.arm("01", 0)
    push_windows(w, emb, "neg", 2, 100.0)
    push_windows(w, emb, "neg", 2, 102.0, value=8)  # the room goes dark
    with pytest.raises(ValueError, match="too dark"):
        w.arm("01", 0)


def test_a_plug_switched_on_in_the_dark_does_not_rearm(home, tuning, fakeplug):
    w, emb = make_watcher(home, tuning, fakeplug.url)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.tripwires["01"].off_acked)
        push_windows(w, emb, "neg", 2, 110.0, value=8)  # dark while tripped
        fakeplug.press_button()
        assert wait_for(lambda: w.tripwires["01"].state == "unarmed")
        assert w.tripwires["01"].unarmed_reason == "too dark to see"
    finally:
        w.stop()


def test_a_camera_moved_while_tripped_needs_a_new_capture(home, tuning, fakeplug):
    w, emb = make_watcher(home, tuning, fakeplug.url)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.tripwires["01"].off_acked)
        with w.lock:
            from jaga.guards import GuardResult

            g = GuardResult(0.5, False, False, 0.0, 0.0, 0, 0.0, True, 0.1, 0.2)
            w._guards_act(120.0, g)  # the camera was knocked
        fakeplug.press_button()
        assert wait_for(lambda: w.tripwires["01"].state == "unarmed")
        assert "camera" in w.tripwires["01"].unarmed_reason
    finally:
        w.stop()


# ------------------------------------------------------------------ P1-2: the I2 floor
def test_one_window_can_never_trip_whatever_the_dwell_or_fraction():
    from jaga.score import DecisionRule

    for dwell, frac in ((1.5, 0.8), (1.0, 0.8), (10, 0.4), (10, 0.1)):
        rule = DecisionRule(dwell, 60, frac)
        assert not rule.push(0.0, True, 1.0).fire, (dwell, frac)
    rule = DecisionRule(1.5, 60, 0.8)
    rule.push(0.0, True, 1.0)
    assert rule.push(1.0, True, 1.0).fire  # two windows above are enough at a 1.5-s dwell


def test_operating_dwell_has_a_floor(home, tuning):
    w, _ = make_watcher(home, tuning, None)
    with pytest.raises(ValueError):
        w.set_operating("01", dwell_s=1)
    with pytest.raises(ValueError):
        w.override_dwell("01", 1, "demo")


# ------------------------------------------------------------------ P1-4: non-finite values
def test_the_wire_never_carries_nan():
    from jaga.server import Broadcaster, dumps

    out = json.loads(dumps({"m": float("nan"), "z": np.float64("inf"), "deep": [{"x": float("-inf")}, 1.5]}))
    assert out == {"m": None, "z": None, "deep": [{"x": None}, 1.5]}
    bc = Broadcaster()
    bc.publish("window", {"m": float("nan")})  # no loop attached: must simply not raise


def test_a_nan_margin_does_not_poison_the_normal_reference():
    from jaga.score import normal_reference

    mu, sd = normal_reference([0.10, float("nan"), 0.12, 0.11])
    assert math.isfinite(mu) and math.isfinite(sd)


# ------------------------------------------------------------------ P1-5: the Job dispatcher
def test_a_cli_timeout_never_prints_a_presigned_url(monkeypatch):
    from jaga import calibrate

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd=["nebius", "--env", "JAGA_IN_URL=https://s/x?X-Amz-Signature=abc"],
                                        timeout=1)

    monkeypatch.setattr(calibrate.subprocess, "run", slow)
    with pytest.raises(RuntimeError) as e:
        calibrate.nebius(["ai", "job", "create", "--env", "JAGA_IN_URL=https://s/x?X-Amz-Signature=abc"])
    assert "X-Amz" not in str(e.value) and "abc" not in str(e.value)


def test_an_input_csv_with_extra_columns_is_refused(tmp_path):
    from jaga import calibrate

    f = tmp_path / "in.csv"
    f.write_text("item_id,tripwire,m,label,source,vec\na,01,0.1,1,clip,0.1;0.2\n")
    with pytest.raises(ValueError, match="egress"):
        calibrate.dispatch(input_csv=str(f), dry_run=True)


def test_a_result_that_fails_the_checks_is_recorded_not_adopted(home):
    from jaga import calibrate
    from jaga.db import DB

    db = DB(home / "c.db")
    db.upsert_tripwire("01", "s", {"name": "X"}, "r", "m", "now")
    nan_result = {"tripwires": {"01": {"status": "ok", "theta_m": float("nan"), "n_pos": 10}}}
    calibrate.store_thresholds("aijob-x", nan_result, db)
    calibrate.store_thresholds("aijob-y", {"tripwires": {"01": {"status": "ok", "theta_m": 0.03, "n_pos": 2}}}, db)
    assert db.latest_job_threshold("01") is None  # neither was adopted
    calibrate.store_thresholds("aijob-z", {"tripwires": {"01": {"status": "ok", "theta_m": 0.03, "n_pos": 10}}}, db)
    assert db.latest_job_threshold("01")["theta"] == pytest.approx(0.03)


def test_a_wrong_mark_reaches_the_job_through_its_window(home, tuning):
    from jaga import calibrate

    w, emb = make_watcher(home, tuning, None)
    w.arm("01", 0)
    push_windows(w, emb, "pos", 10, 100.0)
    trip = next(e for e in w.db.recent_events(10, "test") if e["kind"] == "trip")
    same_ts = w.db.query("SELECT COUNT(*) AS n FROM win WHERE ts = ?", (trip["ts"],))[0]["n"]
    assert same_ts == 1  # the trip event carries its window's timestamp
    w.db.mark_wrong(trip["id"])
    w.db.execute("UPDATE event SET ts = ts + 0.005 WHERE id = ?", (trip["id"],))  # logged a few ms after its window
    rows = calibrate.wrong_rows(w.db)
    assert len(rows) == 1 and rows[0]["label"] == 0 and rows[0]["source"] == "wrong-mark"


# ------------------------------------------------------------------ P1-6: plug_off behind an allow-list
@pytest.mark.parametrize("sentence,cut", [("nobody is at the ironing board", True), ("the stove is left on", True),
                                          ("nggak ada orang di meja setrika", True), ("the TV is on", False),
                                          ("the fridge door is open", False), ("the CPAP machine is on", False)])
def test_power_is_cut_only_for_heating_or_cooking_appliances(sentence, cut):
    from jaga.compile import apply_action_policy, may_cut

    assert may_cut(sentence) is cut
    spec, note = apply_action_policy(sentence, {"action": "plug_off", "name": ""})
    assert (spec["action"] == "plug_off") is cut and (note is None) is cut


def test_the_frozen_tripwires_keep_their_actions_under_the_policy():
    from jaga.compile import may_cut
    from jaga.tripwires import load

    for slot in ("01", "02", "03"):
        fx = load(slot)
        if fx["operating"]["action"] == "plug_off":
            assert may_cut(fx["sentence"], fx["spec"]["name"]), slot


# ------------------------------------------------------------------ P1-7: thread hygiene
def test_sources_can_be_joined_after_they_end(tmp_path):
    from jaga.capture import FileSource, Ring

    src = FileSource(Ring(), write_test_video(tmp_path / "short.mp4", 3.0), realtime=False)
    src.start()
    src.join(10)
    assert not src.is_alive() and src.status.ended  # would raise TypeError if Thread._stop were shadowed


# ------------------------------------------------------------------ P2
def test_notifier_memory_is_bounded_and_short_topics_are_refused():
    n = Notifier(os_banner=False, ntfy_topic="")
    for i in range(500):
        n.notify(f"x{i}")
    assert len(n.sent) == 200
    short = Notifier(os_banner=False, ntfy_topic="jaga")
    assert short.ntfy_topic is None and short.ntfy_refused


def test_slots_are_validated_and_a_bad_token_is_403_not_500(home, tuning, fakeplug):
    from fastapi.testclient import TestClient

    from jaga.server import create_app

    w, emb = make_watcher(home, tuning, fakeplug.url)
    app = create_app(w, compile_fn=lambda s: {}, embed_text_fn=emb.embed_text, token="tok")
    c = TestClient(app, base_url="http://127.0.0.1:8765")
    h = {"host": "127.0.0.1:8765", "x-jaga-session": "tok"}
    assert c.post("/tripwires", json={"sentence": "x", "slot": "../x"}, headers=h).status_code == 422
    r = c.post("/labels", json={"kind": "note"},
               headers={"host": "127.0.0.1:8765", "x-jaga-session": "tök".encode("latin-1")})
    assert r.status_code == 403


def test_captions_are_still_scored_like_before():
    caps = Captions(3)
    from jaga.score import margin

    assert margin(caps.window("pos"), caps.pos, caps.neg) > 0


# ------------------------------------------------------------------- QA round: OFF retries, labels, sessions, rotation
def test_a_failed_off_is_retried_for_as_long_as_it_stays_tripped(home, tuning, fakeplug, monkeypatch):
    import jaga.daemon as d

    monkeypatch.setattr(d, "SLOW_RETRY_S", 0.3)
    fakeplug.fail_off = True
    w, emb = make_watcher(home, tuning, fakeplug.url)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: any("plug still not OFF" in t for t in w.notifier.sent), timeout=10)
        n = fakeplug.off_calls
        assert n >= tuning.action_retries  # the fast budget is spent …
        assert wait_for(lambda: fakeplug.off_calls >= n + 2, timeout=5)  # … and it keeps trying
        fakeplug.fail_off = False
        assert wait_for(lambda: w.plug["state"] == "off", timeout=5)
        assert w.tripwires["01"].off_acked and fakeplug.on_calls == 0
    finally:
        w.stop()


def test_label_keys_follow_the_labels_flag_not_only_scored_sessions(home, tuning):
    from pathlib import Path

    from jaga.cli import build_parser

    w, _ = make_watcher(home, tuning)
    assert w.snapshot()["labels"] is False
    w.labels = True
    assert w.snapshot()["labels"] is True and w.snapshot()["scored"] is False
    a = build_parser().parse_args(["serve", "--labels", "--session", "U1"])
    assert a.labels
    js = (Path(__file__).resolve().parents[1] / "jaga" / "static" / "overlay.js").read_text()
    assert "S.snap.labels && SOAK_KEYS" in js and "S.snap.scored && SOAK_KEYS" not in js


def test_a_session_name_with_data_is_never_reused_but_a_failed_start_is_not_burnt(home, monkeypatch):
    from types import SimpleNamespace

    from jaga import cli
    from jaga.db import DB

    db = DB()

    def boom(*a, **k):
        raise AssertionError("the model must not load before the start checks")

    monkeypatch.setattr("jaga.embed.shared", boom)
    a = SimpleNamespace(db=None, device="cpu", source="camera", camera=0, loop=False, offline=False, port=None,
                        no_os_banner=True, session="B1", camera_pos="board", zone=None, load="01", labels=False)
    # 1. no frozen θ yet: refused before the model loads, and the name stays free
    with pytest.raises(SystemExit, match="frozen θ"):
        cli.build_watcher(a, scored=True, session="B1")
    assert not cli.session_used(db, "B1")
    # 2. a run that started and logged windows owns its name
    db.start_session("B1", True, "board", "live")
    db.add_embed(1.0, "B1", True, 0.5, 0.0, 0.0, None, None, "board", 900, "live")
    with pytest.raises(SystemExit, match="already has data"):
        cli.build_watcher(a, scored=True, session="B1")
    # 3. unscored names too (a second U1 would merge two runs)
    db.add_label("U1", "exit_leave", ts=5.0)
    with pytest.raises(SystemExit, match="already has data"):
        cli.build_watcher(SimpleNamespace(**{**vars(a), "session": "U1"}), scored=False)


def test_a_busy_port_is_refused_before_the_model_loads():
    import socket

    from jaga import cli

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        with pytest.raises(SystemExit, match="in use"):
            cli.port_free(s.getsockname()[1])
    finally:
        s.close()


def test_the_schedule_runner_hands_over_when_the_session_ends():
    import threading as th

    from jaga.soak import ScheduleRunner, Step

    class W:
        def __init__(self):
            self.labels = []

        def label(self, kind, note=""):
            self.labels.append(kind)

        def publish(self, *a):
            pass

    w, done, started = W(), th.Event(), th.Event()
    started.set()
    r = ScheduleRunner(w, [Step(0, 0.0, 0.001, "capture", "capture")], say=False, started=started,
                       on_complete=done.set)
    r.start()
    assert done.wait(10) and w.labels[-1] == "plan:session_end"


def test_portrait_phone_video_is_turned_upright():
    from jaga.frames import upright

    img = np.zeros((4, 6, 3), np.uint8)
    img[0, 0] = 255  # top-left marker

    class F:
        rotation = 90

        def to_ndarray(self, format):
            return img

    out = upright(F())
    assert out.shape == (6, 4, 3)
    assert tuple(out[-1, 0]) == (255, 255, 255)  # rotated 90° counter-clockwise: top-left → bottom-left

    class G(F):
        rotation = 0

    assert upright(G()).shape == (4, 6, 3)


def test_a_fresh_tripwire_never_shows_armed_without_its_normal_state(home, tuning):
    from jaga.score import Threshold

    w, emb = make_watcher(home, tuning, theta=Threshold("fresh-z", 3.0, version=0, source="default"))
    with pytest.raises(ValueError, match="normal state"):
        w.arm("01", 0)
    assert w.tripwires["01"].state != "armed"


def test_malformed_json_is_422_not_500(home, tuning, fakeplug):
    from fastapi.testclient import TestClient

    from jaga.server import create_app

    w, emb = make_watcher(home, tuning, fakeplug.url)
    c = TestClient(create_app(w, compile_fn=lambda s: None, embed_text_fn=emb.embed_text, token="t"),
                   base_url="http://127.0.0.1:8765")
    h = {"host": "127.0.0.1:8765", "x-jaga-session": "t", "content-type": "application/json"}
    assert c.post("/labels", content=b"{not json", headers=h).status_code == 422
    assert c.post("/tripwires", content=b"[1, 2]", headers=h).status_code == 422


def test_a_typed_sentence_never_takes_a_frozen_fixture_slot(home, tuning, fakeplug):
    from conftest import SPEC
    from fastapi.testclient import TestClient

    from jaga.server import create_app

    w, emb = make_watcher(home, tuning, fakeplug.url)

    def compile_fn(sentence):
        return {"spec": {**SPEC, "action": "notify"}, "tf_request_id": "r", "tf_response_id": "r", "model": "m",
                "created": "now", "temperature": 1.0, "latency_s": 0.1}

    c = TestClient(create_app(w, compile_fn=compile_fn, embed_text_fn=emb.embed_text, token="t"),
                   base_url="http://127.0.0.1:8765")
    r = c.post("/tripwires", json={"sentence": "the kettle is left on and nobody is there"},
               headers={"host": "127.0.0.1:8765", "x-jaga-session": "t"})
    assert r.status_code == 200, r.text
    assert r.json()["slot"] not in {"01", "02", "03"}


def test_a_dropped_camera_is_reopened_not_abandoned(monkeypatch):
    from jaga.capture import CameraSource, Ring

    class Cap:
        def __init__(self, live):
            self.live = live

        def grab(self):
            return self.live

        def retrieve(self):
            return True, np.full((90, 160, 3), 120, np.uint8)

        def get(self, prop):
            return 160.0

        def release(self):
            pass

    caps = [Cap(False), Cap(True)]  # unplugged, then back
    monkeypatch.setattr(CameraSource, "_open", lambda self: caps.pop(0) if caps else None)
    monkeypatch.setattr(CameraSource, "REOPEN_AFTER_S", 0.5)
    monkeypatch.setattr(CameraSource, "REOPEN_PAUSE_S", 0.1)
    src = CameraSource(Ring(), 0)
    src.start()
    try:
        assert wait_for(lambda: src.status.extra.get("reopens") == 1 and src.status.frames > 0, timeout=10)
        assert src.status.ok and src.status.error is None
    finally:
        src.stop()
        src.join(5)


def test_a_watcher_that_never_sees_a_window_says_so(home, tuning, monkeypatch):
    import jaga.daemon as d

    monkeypatch.setattr(d, "FIRST_WINDOW_S", 0.5)
    w, _ = make_watcher(home, tuning)
    w.start()
    try:
        assert wait_for(lambda: not w.watcher_ok, timeout=5)
        assert any("no window since start" in t for t in w.notifier.sent) or wait_for(
            lambda: any("no window since start" in t for t in w.notifier.sent), timeout=3)
    finally:
        w.stop()


def test_while_dark_a_cut_tripwire_with_its_plug_on_keeps_reminding(home, tuning, fakeplug, monkeypatch):
    import jaga.daemon as d

    monkeypatch.setattr(d, "DARK_REMINDER_S", 0.5)
    w, emb = make_watcher(home, tuning, fakeplug.url)
    w.start()
    try:
        w.arm("01", 0)
        assert wait_for(lambda: w.plug.get("state") == "on")
        push_windows(w, emb, "neg", 3, 100.0)
        push_windows(w, emb, "neg", 2, 103.0, value=10)  # lights off: unarmed, plug still ON
        assert w.tripwires["01"].unarmed_reason == "too dark to see"
        assert wait_for(lambda: sum("still too dark to see" in t for t in w.notifier.sent) >= 2, timeout=6)
        assert fakeplug.off_calls == 0 and fakeplug.on_calls == 0  # it reminds; it never acts in the dark
    finally:
        w.stop()



@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="needs ffmpeg")
def test_a_dim_light_recording_is_retimed_to_real_time(tmp_path, monkeypatch, capsys):
    """A camera that reports 30 fps but delivers ~10 (dim light) must not produce a sped-up clip."""
    from types import SimpleNamespace

    import cv2

    from jaga import cli
    from jaga.frames import probe

    class SlowCam:
        def __init__(self, *a):
            pass

        def isOpened(self):  # noqa: N802
            return True

        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 30.0, cv2.CAP_PROP_FRAME_WIDTH: 320.0, cv2.CAP_PROP_FRAME_HEIGHT: 180.0}[prop]

        def read(self):
            time.sleep(0.1)
            return True, np.full((180, 320, 3), 128, np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", SlowCam)
    a = SimpleNamespace(set="iron", label="pos", n=None, lighting="lamp", camera_pos="board", seconds=2.0,
                        countdown=0, camera=0, out=str(tmp_path))
    assert cli.cmd_record(a) == 0
    f = tmp_path / "iron" / "pos_01_lamp_board.mp4"
    assert f.exists() and not (tmp_path / "iron" / "pos_01_lamp_board.part.mp4").exists()
    assert probe(f).duration_s == pytest.approx(2.0, abs=0.4)  # untouched it would last ~0.7 s
    assert "retimed to real time" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="hyphens"):
        cli.cmd_record(SimpleNamespace(**{**vars(a), "camera_pos": "kitchen_door"}))


def test_the_optional_check_is_reachable_after_the_capture(home, tuning):
    """The "it's happening now" examples need the normal state, so the overlay offers them right after the capture."""
    w, emb = make_watcher(home, tuning, action="notify")
    w.arm("01", 3)
    push_windows(w, emb, "neg", 6, 100.0)
    row = next(r for r in w.snapshot()["windows"] if r["slot"] == "01")
    assert row["state"] == "armed" and row["armed_at"] is not None and row["examples"] == 0
    assert w.add_example("01") == {"ok": True, "examples": 1}
    js = (__import__("pathlib").Path(__file__).resolve().parents[1] / "jaga" / "static" / "overlay.js").read_text()
    assert '(w.state === "armed" || w.state === "holding") && w.armed_at' in js  # the check card follows the capture


# ------------------------------------------------------------------- one plug, many tripwires; lost acknowledgements
SPEC_COOLDOWN = 60.0  # conftest.SPEC cooldown_s


class FlakyPlug(PlugDriver):
    """turn_off fails `fail_first` times; with `lose_ack` the plug really switches off but the ack never arrives."""

    kind, label, address = "test", "Wi-Fi plug", "test"

    def __init__(self, fail_first=0, lose_ack=False):
        self.on, self.fail_first, self.lose_ack = True, fail_first, lose_ack
        self.off_calls = 0

    def turn_off(self):
        self.off_calls += 1
        if self.lose_ack:
            self.on = False
            raise ActionFailed("no acknowledgement (timeout)")
        if self.off_calls <= self.fail_first:
            raise ActionFailed("the plug is not reachable")
        self.on = False
        return Ack(True, 5.0, "test ack")

    def status(self):
        return self.on


def _two_cut_tripwires(home, tuning, driver):
    from conftest import SPEC

    w, emb = make_watcher(home, tuning)
    w.driver = driver
    w.load_tripwire("02", "nobody is at the stove", {**SPEC, "name": "STOVE UNATTENDED"}, emb.caps.pos, emb.caps.neg,
                    w.tripwires["01"].threshold.__class__("labelled-m", 0.0, version=1, source="test"),
                    {"tf_request_id": "test"})
    return w, emb


def test_one_acknowledged_off_covers_every_tripwire_on_the_plug(home, tuning):
    plug = FlakyPlug(fail_first=1)
    w, emb = _two_cut_tripwires(home, tuning, plug)
    w.start()
    try:
        w.arm("01", 0)
        w.arm("02", 0)
        push_windows(w, emb, "pos", 12, 100.0)
        assert wait_for(lambda: all(t.off_acked for t in w.tripwires.values()), timeout=5)
        assert w.retry is None and not w.plug["failed"]
        plug.on = True  # a person switches it back on: both re-arm, nothing is left tripped
        assert wait_for(lambda: all(t.state == "armed" for t in w.tripwires.values()), timeout=5)
    finally:
        w.stop()


def test_a_lost_acknowledgement_is_confirmed_by_a_status_read(home, tuning):
    from dataclasses import replace

    t = replace(tuning, action_retry_s=1.0, rearm_poll_s=0.1)
    plug = FlakyPlug(lose_ack=True)
    w, emb = make_watcher(home, t)
    w.driver = plug
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.tripwires["01"].off_acked, timeout=5)  # the plug reads OFF after the trip
        assert w.retry is None and w.plug["state"] == "off"
        calls = plug.off_calls
        plug.on = True  # back at the board, the person presses the plug's button
        assert wait_for(lambda: w.tripwires["01"].state == "armed", timeout=5)
        time.sleep(1.5)
        assert plug.off_calls == calls  # no retry cut the power the person just switched on
    finally:
        w.stop()


def test_disarming_during_a_retry_stops_it_and_clears_action_failed(home, tuning):
    plug = FlakyPlug(fail_first=10**6)
    w, emb = make_watcher(home, tuning)
    w.driver = plug
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.plug.get("failed") and w.retry is not None, timeout=5)
        w.disarm("01")
        assert w.retry is None and not w.plug["failed"]
        n = plug.off_calls
        time.sleep(0.5)
        assert plug.off_calls == n
    finally:
        w.stop()


def test_a_notify_tripwire_leaving_its_cool_down_goes_through_the_arm_gate(home, tuning):
    w, emb = make_watcher(home, tuning, action="notify", dwell_s=5)
    w.arm("01", 0)
    tw = w.tripwires["01"]
    res = push_windows(w, emb, "pos", 8, 100.0)
    assert any(r[0].fired for r in res) and tw.state == "tripped"
    for t2 in w.tripwires.values():  # the camera was knocked during the cool-down (as the guard marks it)
        t2.stale_ref = True
    push_windows(w, emb, "neg", 4, 100.0 + SPEC_COOLDOWN + 20)  # past the 60-s cool-down
    assert tw.state == "unarmed" and "camera moved" in (tw.unarmed_reason or "")


def test_a_camera_that_grabs_but_never_decodes_is_reopened(monkeypatch):
    from jaga.capture import CameraSource, Ring

    class Cap:
        def __init__(self, decodes):
            self.decodes = decodes

        def grab(self):
            return True

        def retrieve(self):
            return (True, np.full((90, 160, 3), 120, np.uint8)) if self.decodes else (False, None)

        def get(self, prop):
            return 160.0

        def release(self):
            pass

    caps = [Cap(False), Cap(True)]
    monkeypatch.setattr(CameraSource, "_open", lambda self: caps.pop(0) if caps else None)
    monkeypatch.setattr(CameraSource, "REOPEN_AFTER_S", 0.5)
    monkeypatch.setattr(CameraSource, "REOPEN_PAUSE_S", 0.1)
    src = CameraSource(Ring(), 0, fps=50.0)
    src.start()
    try:
        assert wait_for(lambda: src.status.extra.get("reopens") == 1 and src.status.frames > 0, timeout=10)
    finally:
        src.stop()
        src.join(5)


def test_the_banner_text_travels_as_an_argument_not_as_script():
    from jaga.act import banner_argv

    text = 'jaga · 01 BOARD — tripped "now" \\ end'
    argv = banner_argv(text, "jaga")
    scripts = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert all(text not in sc and "·" not in sc for sc in scripts)
    assert argv[argv.index("--") + 1:] == [text, "jaga"]


# ------------------------------------------------------------------- the Tuya driver against a fake tinytuya device
class FakeOutlet:
    """Stands in for tinytuya.OutletDevice: remembers calls, answers like the firmware variants we expect."""

    mode = "dps"  # "dps": the set reply carries dps · "bare": no dps (read back) · "error" · "raise" · "stuck"
    on = True
    calls: list = []

    def __init__(self, dev_id, ip, key, version=3.3):
        FakeOutlet.calls.append(("init", dev_id, ip, version))

    def set_socketTimeout(self, s):
        pass

    def set_socketRetryLimit(self, n):
        pass

    def turn_off(self, switch=1):
        FakeOutlet.calls.append(("turn_off", switch))
        if FakeOutlet.mode == "raise":
            raise OSError("socket timed out")
        if FakeOutlet.mode == "error":
            return {"Error": "Device Unreachable", "Err": "905"}
        if FakeOutlet.mode != "stuck":
            FakeOutlet.on = False
        return {"dps": {str(switch): False}} if FakeOutlet.mode == "dps" else {}

    def turn_on(self, switch=1):  # the trip path must never call this
        FakeOutlet.calls.append(("turn_on", switch))
        FakeOutlet.on = True

    def status(self):
        FakeOutlet.calls.append(("status",))
        return {"dps": {"1": FakeOutlet.on}}


@pytest.mark.parametrize("mode,acked", [("dps", True), ("bare", True), ("error", False), ("raise", False),
                                        ("stuck", False)])
def test_the_tuya_driver_acknowledges_only_a_confirmed_off(monkeypatch, mode, acked):
    import types

    from jaga.act import TuyaPlug

    fake = types.SimpleNamespace(OutletDevice=FakeOutlet)
    monkeypatch.setitem(__import__("sys").modules, "tinytuya", fake)
    FakeOutlet.mode, FakeOutlet.on, FakeOutlet.calls = mode, True, []
    plug = TuyaPlug("dev123", "192.168.1.50", "localkey", version="3.4")
    assert plug.status() is True
    if acked:
        ack = plug.turn_off()
        assert ack.off and plug.status() is False
    else:
        with pytest.raises(ActionFailed):
            plug.turn_off()
    assert not any(c[0] == "turn_on" for c in FakeOutlet.calls)  # I1: no ON from the driver's OFF path
    assert ("init", "dev123", "192.168.1.50", 3.4) in FakeOutlet.calls


def test_a_tuya_trip_through_the_watcher_cuts_once_and_never_switches_on(home, tuning, monkeypatch):
    import types

    from jaga.act import TuyaPlug

    monkeypatch.setitem(__import__("sys").modules, "tinytuya", types.SimpleNamespace(OutletDevice=FakeOutlet))
    FakeOutlet.mode, FakeOutlet.on, FakeOutlet.calls = "dps", True, []
    w, emb = make_watcher(home, tuning)
    w.driver = TuyaPlug("dev123", "192.168.1.50", "localkey")
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.tripwires["01"].off_acked and w.plug["state"] == "off", timeout=5)
        assert sum(1 for c in FakeOutlet.calls if c[0] == "turn_off") == 1
        assert not any(c[0] == "turn_on" for c in FakeOutlet.calls)
    finally:
        w.stop()


def test_an_off_reply_never_vouches_for_a_trip_that_came_after_the_call(home, tuning):
    """I3 with two cutting tripwires: 02 trips while 01's OFF call is in flight (the plug may have been switched on
    since). 01's acknowledgement must not mark 02 acknowledged; 02 waits for its own OFF."""
    w, emb = _two_cut_tripwires(home, tuning, None)

    class MidCallPlug(PlugDriver):
        kind, label, address = "test", "Wi-Fi plug", "test"

        def turn_off(self):
            t2 = w.tripwires["02"]  # the second trip lands during this call
            t2.state, t2.off_acked, t2.tripped_mono = "tripped", False, time.monotonic()
            return Ack(True, 5.0, "test ack")

        def status(self):
            return False

    w.driver = MidCallPlug()
    t1 = w.tripwires["01"]
    t1.state, t1.off_acked, t1.tripped_mono = "tripped", False, time.monotonic()
    assert w._turn_off(t1, 100.0, 10.0, first=True)
    assert t1.off_acked and not w.tripwires["02"].off_acked


def test_the_breaker_waits_for_every_tripped_row():
    from pathlib import Path

    js = (Path(__file__).resolve().parents[1] / "jaga" / "static" / "overlay.js").read_text()
    assert "cut.every((w) => w.off_acked)" in js


def test_huge_numbers_and_head_requests_never_500(home, tuning, fakeplug):
    from fastapi.testclient import TestClient

    from jaga.server import create_app

    w, emb = make_watcher(home, tuning, fakeplug.url)
    c = TestClient(create_app(w, compile_fn=lambda s: None, embed_text_fn=emb.embed_text, token="t"),
                   base_url="http://127.0.0.1:8765")
    h = {"host": "127.0.0.1:8765", "x-jaga-session": "t", "content-type": "application/json"}
    assert c.post("/tripwires/01/arm", content=b'{"capture_s": 1' + b"0" * 400 + b"}", headers=h).status_code == 422
    assert c.post("/tripwires/01/arm", content=b'{"capture_s": 1' + b"0" * 5000 + b"}", headers=h).status_code == 422
    r = c.post("/tripwires/07/arm", json={}, headers={"host": "127.0.0.1:8765", "x-jaga-session": "t"})
    assert r.status_code == 404 and not r.json()["detail"].startswith("'")
    for path in ("/", "/api/state", "/receipts"):
        assert c.head(path, headers={"host": "127.0.0.1:8765"}).status_code == 200
    assert "run_id" in c.get("/api/state", headers={"host": "127.0.0.1:8765"}).json()


def test_arm_must_name_a_loaded_tripwire_and_soak_b_needs_its_schedule(home, monkeypatch):
    from types import SimpleNamespace

    from jaga import cli

    monkeypatch.setattr("jaga.embed.shared", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model loaded")))
    a = SimpleNamespace(db=None, device="cpu", source="camera", camera=0, loop=False, offline=False, port=None,
                        no_os_banner=True, session="x1", camera_pos="board", zone=None, load="01", arm="07",
                        labels=False)
    with pytest.raises(SystemExit, match="arm only tripwires you --load"):
        cli.build_watcher(a)
    with pytest.raises(SystemExit, match="schedule"):
        cli.cmd_soak(SimpleNamespace(**{**vars(a), "kind": "B", "schedule": None, "session": "B9", "arm": None}))


# ------------------------------------------------------------------- QA round 4
def test_a_recording_that_ends_is_not_a_stopped_watcher(home, tuning, tmp_path):
    from jaga.act import Notifier, NullPlug
    from jaga.capture import FileSource, Ring
    from jaga.daemon import Watcher
    from jaga.db import DB
    from jaga.guards import Guards

    clip = write_test_video(tmp_path / "short.mp4", 10.0)
    ring = Ring()
    src = FileSource(ring, clip, realtime=False)
    w = Watcher(db=DB(home / "e.db"), embedder=__import__("conftest").ScriptedEmbedder(Captions()), ring=ring,
                driver=NullPlug(), notifier=Notifier(os_banner=False, ntfy_topic=""), session="ended", tuning=tuning,
                guards=Guards(tuning, hog=False), source=src, offline=True)
    w.start()
    src.start()
    try:
        assert wait_for(lambda: w.source_ended, timeout=20)
        time.sleep(1.5)  # the watchdog has looked again since
        kinds = [e["kind"] for e in w.db.recent_events(20, "ended")]
        assert "source_ended" in kinds and "watcher_stopped" not in kinds and w.watcher_ok
        assert w.snapshot()["source_ended"] is True
    finally:
        w.stop()


def test_action_failed_says_whether_a_push_went_out(home, tuning):
    w, emb = make_watcher(home, tuning)  # no banner, no ntfy topic: no channel
    w.driver = FlakyPlug(fail_first=10**6)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: any(e["kind"] == "action_failed" for e in w.db.recent_events(20, "test")), timeout=5)
        detail = next(e["detail"] for e in w.db.recent_events(20, "test") if e["kind"] == "action_failed")
        assert "no push channel is on" in detail and "push sent" not in detail
        assert w.snapshot()["push_channels"] == []
    finally:
        w.stop()


def test_a_new_trip_clears_the_previous_action_line_and_stop_warns(home, tuning, capsys):
    w, emb = make_watcher(home, tuning)
    w.driver = FlakyPlug(fail_first=10**6)
    w.plug.update(last_action="off", latency_ms=12.0)  # an earlier trip's acknowledged OFF
    w.start()
    w.arm("01", 0)
    push_windows(w, emb, "pos", 10, 100.0)
    assert wait_for(lambda: w.tripwires["01"].state == "tripped")
    assert w.plug["last_action"] in ("off-pending", "off-failed") and w.plug["latency_ms"] is None
    w.stop()
    assert "OFF for tripwire 01 is unconfirmed" in capsys.readouterr().err


def test_bad_flag_values_are_refused_before_the_model_loads(home, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from jaga import cli

    monkeypatch.setattr("jaga.embed.shared", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model loaded")))
    base = dict(db=None, device="cpu", source="camera", camera=0, loop=False, offline=False, port=None,
                no_os_banner=True, session=None, camera_pos="board", zone=None, load="01", arm=None, labels=False,
                capture_s=None, dwell=None)
    for bad, msg in ((dict(dwell=["02=30"]), "not loaded"), (dict(dwell=["01=abc"]), "slot>=<seconds"),
                     (dict(dwell=["01=1"]), "2–600"), (dict(capture_s=-5.0), "between 0 and 600"),
                     (dict(load="09"), "no frozen tripwire 09")):
        with pytest.raises(SystemExit, match=msg):
            cli.build_watcher(SimpleNamespace(**{**base, **bad}))
    with pytest.raises(SystemExit, match="no such file"):
        cli.cmd_soak(SimpleNamespace(**{**base, "kind": "B", "schedule": str(tmp_path / "nope.csv"), "session": "B9"}))


def test_path_ids_and_missing_slots_answer_plainly(home, tuning, fakeplug):
    from fastapi.testclient import TestClient

    from jaga.server import create_app

    w, emb = make_watcher(home, tuning, fakeplug.url)
    c = TestClient(create_app(w, compile_fn=lambda s: None, embed_text_fn=emb.embed_text, token="t"),
                   base_url="http://127.0.0.1:8765")
    h = {"host": "127.0.0.1:8765", "x-jaga-session": "t"}
    assert c.post("/events/9223372036854775808/wrong", headers=h).status_code == 404
    for path in ("/tripwires/09/disarm", "/tripwires/09/example"):
        r = c.post(path, headers=h)
        assert r.status_code == 404 and not r.json()["detail"].startswith("'"), r.text


def test_the_overlay_swaps_frames_only_once_decoded_and_keeps_the_choice_by_the_buttons():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "jaga" / "static"
    js, html = (root / "overlay.js").read_text(), (root / "index.html").read_text()
    css = (root / "overlay.css").read_text()
    assert "next.decode()" in js and 'decoding="async"' not in html
    foot = html[html.index('<div class="dr-foot">'):]
    assert foot.index('id="dr-choose"') < foot.index('id="dr-arm"')
    assert ".cam:not(.live) .nosignal { z-index: 4" in css
    assert "no push channel is on" in js and "push sent" not in js
