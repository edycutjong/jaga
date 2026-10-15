"""The measuring harness: frames, offline scoring, seed import, the protocol schedules, soak scoring, readiness."""
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import Captions, write_test_video

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------- frames / offline
def test_canonical_frame_keeps_aspect_and_long_side():
    from jaga.frames import canon

    f = canon(np.zeros((1080, 1920, 3), np.uint8))
    assert f.shape == (360, 640, 3)
    p = canon(np.zeros((1920, 1080, 3), np.uint8))
    assert p.shape == (640, 360, 3)


def test_window_is_eight_frames_at_one_fps(tmp_path):
    from jaga.frames import clip_window_start, read_window, window_times

    assert window_times(2.0)[:2] == [2.5, 3.5] and len(window_times(0.0)) == 8
    assert clip_window_start(12.0, "middle") == 2.0 and clip_window_start(12.0, "end") == 4.0
    v = write_test_video(tmp_path / "c.mp4", 12.0)
    frames, meta = read_window(v, "middle")
    assert frames.shape[0] == 8 and frames.shape[3] == 3 and meta["start_s"] == pytest.approx(2.0, abs=0.01)


def test_sliding_windows_step_by_the_stride(tmp_path):
    from jaga.offline import sliding_windows

    v = write_test_video(tmp_path / "long.mp4", 20.0)
    w1 = list(sliding_windows(v, stride_frames=1))
    w2 = list(sliding_windows(v, stride_frames=2))
    assert len(w1) == 13 and 6 <= len(w2) <= 7 and w1[0][1].shape[0] == 8


def test_clip_id_from_path():
    from jaga.offline import clip_id

    assert clip_id("data/clips/iron/pos_03.mp4") == "iron/pos_03"
    assert clip_id("iron/pos_03") == "iron/pos_03"


def test_fresh_mode_trips_inside_positive_and_not_on_holdout():
    from jaga.offline import fresh_mode

    caps = Captions(11)
    n_ts = np.arange(0, 120, 1.0)
    n_v = np.stack([caps.window("neg") for _ in n_ts])
    p_ts = np.arange(500, 560, 1.0)
    p_v = np.stack([caps.window("pos") for _ in p_ts])
    r = fresh_mode(n_ts, n_v, p_ts, p_v, caps.pos, caps.neg, dwell_s=30, stride_s=1.0)
    assert r["ok"] and r["tripped_correctly"] and r["holdout_trips"] == 0 and 20 <= r["time_to_trip_s"] <= 30
    assert r["n_fit"] == 60 and r["n_holdout"] == 60


# ------------------------------------------------------------------- seed import
def test_import_reads_labels_from_filenames(tmp_path, monkeypatch):
    monkeypatch.setenv("JAGA_HOME", str(tmp_path / "home"))
    seed = load_script("seed")
    monkeypatch.setattr(seed, "LOCAL_CLIPS", tmp_path / "home" / "clips")
    monkeypatch.setattr(seed, "CLIPS_CSV", tmp_path / "clips.csv")
    raw = tmp_path / "raw"
    (raw / "iron").mkdir(parents=True)
    write_test_video(raw / "iron" / "pos_01_lamp_board.mp4", 10.0)
    write_test_video(raw / "iron" / "neg_02_day_board.mp4", 10.0)
    write_test_video(raw / "iron" / "badname.mp4", 10.0)
    (raw / "iron-day2").mkdir()
    write_test_video(raw / "iron-day2" / "pos_01_dusk_board.mp4", 9.0)

    class A:
        pass

    a = A()
    a.raw, a.window, a.force = str(raw), "middle", False
    seed.cmd_import(a)
    rows = {r["clip_id"]: r for r in csv.DictReader((tmp_path / "clips.csv").open())}
    assert set(rows) == {"iron/pos_01", "iron/neg_02", "iron-day2/pos_01"}
    assert rows["iron/pos_01"]["label"] == "1" and rows["iron/neg_02"]["label"] == "0"
    assert rows["iron/pos_01"]["camera_pos"] == "board" and rows["iron/neg_02"]["lighting"] == "day"
    assert rows["iron-day2/pos_01"]["heldout"] == "1" and rows["iron/pos_01"]["published"] == "0"
    assert (tmp_path / "home" / "clips" / "iron" / "pos_01.mp4").exists()


# ------------------------------------------------------------------- protocol
def test_protocol_schedules_are_reproducible_and_meet_the_minimums():
    proto = load_script("protocol")
    for s in ("B1", "B2"):
        rows = proto.generate(s)
        committed = list(csv.DictReader((REPO / "docs" / "protocol" / f"{s}.csv").open()))
        assert [dict(r) for r in committed] == [{k: str(v) for k, v in r.items()} for r in rows]
        kinds = [r["kind"] for r in rows]
        assert kinds.count("still") >= 10 and kinds.count("exit_leave") + kinds.count("exit_inview") >= 6
        assert kinds.count("exit_inview") >= 2 and float(rows[-1]["end_min"]) >= 120
        assert rows[0]["kind"] == "capture"
        for r in rows:
            if r["kind"] == "still":
                assert float(r["end_min"]) - float(r["start_min"]) >= 3


# ------------------------------------------------------------------- soak scoring on a synthetic log
def test_soak_scoring_counts_cuts_exits_and_time_to_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("JAGA_HOME", str(tmp_path / "home"))
    bench = load_script("bench")
    from jaga.db import DB

    db = DB(tmp_path / "s.db")
    tid = db.upsert_tripwire("01", "nobody is at the ironing board", {"name": "X"}, "req", "m", "now")
    db.add_threshold(tid, 1, "labelled-m", 0.0, None, None, "loo-clips", "B1")
    db.start_session("B1", True, "board", "live")
    t0 = 1000.0
    # schedule: still 0–200 s (person present, still), exit 300–500 s (away), activity otherwise
    db.add_label("B1", "plan:still", ts=t0)
    db.add_label("B1", "plan_end:still", ts=t0 + 200)
    db.add_label("B1", "plan:exit_leave", ts=t0 + 300)
    db.add_label("B1", "plan_end:exit_leave", ts=t0 + 500)
    for i in range(700):
        ts = t0 + i
        away = 300 + 10 <= i < 500
        still = i < 200 or away
        eid = db.add_embed(ts, "B1", True, 0.5, 0.1 if still else 3.0, 0.0, 0 if away else 1, 1.0, "board", 900, "live")
        db.add_win(eid, ts, tid, 0.05 if away else -0.05, None, None, None, 1, 900, False, "armed", away)
    db.add_event("trip", "jaga", tid, session="B1", ts=t0 + 300 + 10 + 100)
    r = bench.soak_rows(db, "B1", "01", dwell_s=120, k_fraction=0.8, p8=0.5, r_thr=None)
    sysm = r["systems"]
    assert sysm["jaga (live)"]["still_cut"] == 0 and sysm["jaga (live)"]["exits_caught"] == 1
    assert sysm["jaga (live)"]["median_time_to_trip_s"] == pytest.approx(110)
    assert sysm["stillness timer"]["still_cut"] == 1  # the timer cannot tell 'still' from 'gone'
    assert sysm["stillness timer"]["exits_caught"] == 1
    assert sysm["jaga (recomputed)"]["exits_caught"] == 1 and sysm["jaga (recomputed)"]["still_cut"] == 0


# ------------------------------------------------------------------- readiness + offline
def test_readiness_flags_placeholders_and_missing_video(tmp_path):
    sub = tmp_path / "submission-text.md"
    sub.write_text("We measured <bench:cuts_jaga_B> cuts.\n")
    ck = load_script("check_submission_readiness")
    res = {n: (ok, why) for n, ok, why in ck.check(sub, set())}
    assert not res["placeholders"][0] and "submission-text.md:1" in res["placeholders"][1]
    assert not res["video"][0]
    assert res["license"][0] and res["cosmos"][0] and res["kitchen"][0]


def test_verify_offline_ci_variant_trips_with_egress_blocked():
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "verify_offline.py")], capture_output=True, text=True,
                       timeout=120)
    out = json.loads(p.stdout[p.stdout.index("{"):])
    assert p.returncode == 0, out
    assert out["tripped"] and out["plug_off_acknowledged"] and out["outbound_after_block"] is False
    assert "1.1.1.1" in out["blocked_attempts"]  # the probe really was refused by the block
    assert out["fake_plug_on_calls"] == 0
