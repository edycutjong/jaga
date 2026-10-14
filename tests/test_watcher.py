"""The live watcher end to end with scripted vectors and the fake LAN plug: trip → OFF ack → re-arm."""
import numpy as np
from conftest import make_watcher, push_windows, wait_for

from jaga.capture import Window


def states(w):
    return {s: tw.state for s, tw in w.tripwires.items()}


def test_trip_cuts_power_and_ui_shows_off_only_after_ack(home, tuning, fakeplug):
    """I3: the plug state turns 'off' only after the driver acknowledged; the trip itself is logged first."""
    w, emb = make_watcher(home, tuning, fakeplug.url)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "neg", 12, 100.0)
        assert states(w)["01"] == "armed" and w.plug["state"] != "off"
        res = push_windows(w, emb, "pos", 10, 112.0)
        assert any(r[0].fired for r in res)
        assert wait_for(lambda: w.plug["state"] == "off")
        kinds = [e["kind"] for e in w.db.recent_events(10, "test")]
        assert kinds.index("action") < kinds.index("trip")  # newest first: the trip came before the ack
        assert fakeplug.off_calls == 1 and fakeplug.on_calls == 0 and fakeplug.on is False
    finally:
        w.stop()


def test_action_failed_is_shown_pushed_and_retried_off_only(home, tuning, fakeplug):
    fakeplug.fail_off = True
    w, emb = make_watcher(home, tuning, fakeplug.url)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.plug["failed"])
        assert any("ACTION FAILED" in t for t in w.notifier.sent)
        assert w.plug["state"] != "off" and states(w)["01"] == "tripped"
        fakeplug.fail_off = False  # the plug comes back: the retry (OFF only) succeeds
        assert wait_for(lambda: w.plug["state"] == "off", timeout=5)
        assert fakeplug.on_calls == 0
    finally:
        w.stop()


def test_rearm_on_plug_on_never_calls_on_and_reuses_reference(home, tuning, fakeplug):
    """Re-arm: an observed OFF → ON (the plug's own button) re-arms with the stored θ; jaga never sends ON."""
    w, emb = make_watcher(home, tuning, fakeplug.url)
    w.start()
    try:
        w.arm("01", 0)
        tw = w.tripwires["01"]
        theta_before = (tw.threshold.theta, tw.threshold.version)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.plug["state"] == "off")
        assert tw.rule.cooldown_until > 0
        fakeplug.press_button()  # a person switches it back on by hand
        assert wait_for(lambda: tw.state == "armed", timeout=5)
        assert (tw.threshold.theta, tw.threshold.version) == theta_before
        assert tw.rule.cooldown_until < 0  # re-arming ended the plug_off cool-down
        assert fakeplug.on_calls == 0
        kinds = [e["kind"] for e in w.db.recent_events(20, "test")]
        assert "rearmed" in kinds
        res = push_windows(w, emb, "pos", 10, 110.0)  # leaves again (contiguous timeline): guarded again
        assert any(r[0].fired for r in res)
    finally:
        w.stop()


def test_no_rearm_when_the_plug_was_never_seen_off(home, tuning, fakeplug):
    fakeplug.fail_off = True
    tuning_ = tuning
    w, emb = make_watcher(home, tuning_, fakeplug.url)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "pos", 10, 100.0)
        assert wait_for(lambda: w.plug["failed"])
        assert not wait_for(lambda: w.tripwires["01"].state == "armed", timeout=0.8)
    finally:
        w.stop()


def test_notify_tripwire_notifies_and_rearms_after_cooldown(home, tuning):
    w, emb = make_watcher(home, tuning, None, action="notify", dwell_s=5)
    w.start()
    try:
        w.arm("01", 0)
        tw = w.tripwires["01"]
        tw.cooldown_s = 20.0
        tw.rule.cooldown_s = 20.0
        push_windows(w, emb, "pos", 6, 100.0)
        assert wait_for(lambda: any("tripped" in t for t in w.notifier.sent))
        push_windows(w, emb, "neg", 25, 106.0)
        assert tw.state == "armed"
    finally:
        w.stop()


def test_watchdog_reports_watcher_stopped(home, tuning):
    w, emb = make_watcher(home, tuning, None, offline=False)
    w.start()
    try:
        w.arm("01", 0)
        push_windows(w, emb, "neg", 3, 0.0)
        assert wait_for(lambda: not w.watcher_ok, timeout=4)
        kinds = [e["kind"] for e in w.db.recent_events(10, "test")]
        assert "watcher_stopped" in kinds
        push_windows(w, emb, "neg", 1, 10.0)
        assert w.watcher_ok
    finally:
        w.stop()


def test_too_dark_unarms_and_light_return_rearms(home, tuning):
    w, emb = make_watcher(home, tuning, None)
    w.arm("01", 0)
    w.tripwires["01"].threshold.mu0, w.tripwires["01"].threshold.sigma0 = 0.0, 0.05
    push_windows(w, emb, "neg", 3, 100.0)
    push_windows(w, emb, "neg", 2, 103.0, value=10)
    assert states(w)["01"] == "unarmed" and w.tripwires["01"].unarmed_reason == "too dark to see"
    push_windows(w, emb, "neg", tuning.light_return_windows + 1, 105.0)
    assert states(w)["01"] == "armed"


def test_sudden_light_change_suspends_trips(home, tuning):
    w, emb = make_watcher(home, tuning, None, action="notify", dwell_s=5)
    w.arm("01", 0)
    push_windows(w, emb, "pos", 3, 100.0, value=200)
    res = push_windows(w, emb, "pos", 4, 103.0, value=110)  # the room light dims by ~45 %
    assert not any(r[0].fired for r in res)
    kinds = [e["kind"] for e in w.db.recent_events(10, "test")]
    assert "suspend" in kinds


def test_every_window_is_logged_with_baseline_inputs(home, tuning):
    w, emb = make_watcher(home, tuning, None)
    w.arm("01", 0)
    push_windows(w, emb, "neg", 5, 100.0)
    rows = w.db.query("SELECT * FROM embed WHERE session='test'")
    assert len(rows) == 5 and all(r["luma"] is not None and r["still_s"] is not None for r in rows)
    wins = w.db.query("SELECT * FROM win")
    assert len(wins) == 5 and all(r["theta_version"] == 1 for r in wins)
    w.vstore.flush()
    from jaga import vectors

    ids, vecs = vectors.load("test")
    assert list(ids) == [r["id"] for r in rows] and vecs.shape == (5, 256)


def test_capture_sets_reference_then_arms(home, tuning):
    from jaga.score import Threshold

    w, emb = make_watcher(home, tuning, None, theta=Threshold("fresh-z", None))
    w.arm("01", 5)
    push_windows(w, emb, "neg", 8, 100.0)
    tw = w.tripwires["01"]
    assert tw.state == "armed" and tw.threshold.mu0 is not None and w.guards.has_reference


def test_snapshot_is_json_safe(home, tuning):
    from jaga.server import dumps

    w, emb = make_watcher(home, tuning, None)
    w.arm("01", 0)
    push_windows(w, emb, "neg", 2, 100.0)
    import json

    snap = json.loads(dumps(w.snapshot(include_frames=True)))  # dumps refuses NaN/inf (allow_nan=False)
    assert snap["windows"][0]["slot"] == "01" and len(snap["filmstrip_jpeg_b64"]) == 8


def test_window_failure_is_logged_not_fatal(home, tuning):
    w, emb = make_watcher(home, tuning, None)

    def boom(_):
        raise RuntimeError("model went away")

    emb.embed_video = boom
    w.start()
    try:
        w.ring.push(1.0, np.zeros((90, 160, 3), np.uint8))
        for i in range(8):
            w.ring.push(2.0 + i, np.zeros((90, 160, 3), np.uint8))
        assert wait_for(lambda: any(e["kind"] == "watcher_stopped" for e in w.db.recent_events(5, "test")))
    finally:
        w.stop()
    assert isinstance(Window, type)
