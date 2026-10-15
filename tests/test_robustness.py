"""Regression tests for the a2a round-03 architecture review: acknowledgement, liveness, plug I/O ordering,
guards, the I2 floor, non-finite values, the Job dispatcher, the action allow-list and thread hygiene."""
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
