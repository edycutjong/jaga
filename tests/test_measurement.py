"""What the measurements count: the clip, fresh-mode, Soak A/B, tuning and ablation rules in scripts/bench.py,
and which clips may fit a threshold (scripts/seed.py, jaga/calibrate.py). Synthetic logs only — no model."""
import csv
import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import Captions, write_test_video

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def bench(tmp_path, monkeypatch):
    monkeypatch.setenv("JAGA_HOME", str(tmp_path / "home"))
    b = load_script("bench")
    monkeypatch.setattr(b, "OUT", tmp_path / "out")
    monkeypatch.setattr(b, "SUMMARY", tmp_path / "out" / "summary.json")
    monkeypatch.setattr(b, "DEMO", tmp_path / "DEMO.md")
    return b


# ------------------------------------------------------------------- which clips fit θ
CLIPS = [  # set, label, heldout
    ("iron", 1, 0), ("iron", 0, 0), ("iron-inview", 1, 0), ("iron-day2", 1, 1),
]


def _clips_csv(path: Path) -> None:
    seed = load_script("seed")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seed.CLIP_COLUMNS, lineterminator="\n")
        w.writeheader()
        for i, (s, lab, held) in enumerate(CLIPS):
            w.writerow({"clip_id": f"{s}/{'pos' if lab else 'neg'}_{i:02d}", "set": s, "tripwire": "01",
                        "label": lab, "heldout": held, "lighting": "day", "camera_pos": "board"})


def test_in_view_and_held_out_clips_never_fit_theta(tmp_path, monkeypatch):
    import jaga.offline as off
    from jaga import calibrate

    _clips_csv(tmp_path / "clips.csv")
    caps = Captions(3)
    monkeypatch.setattr(off, "CLIPS_CSV", tmp_path / "clips.csv")
    monkeypatch.setattr(off, "committed_vector", lambda cid: caps.window("pos"))
    ids = {r["item_id"] for r in calibrate.clip_rows()}
    assert ids == {"iron/pos_00", "iron/neg_01"}  # not iron-inview, not iron-day2

    seed = load_script("seed")
    monkeypatch.setattr(seed, "CLIPS_CSV", tmp_path / "clips.csv")
    monkeypatch.setattr(seed, "read_clips_csv", lambda: list(csv.DictReader((tmp_path / "clips.csv").open())))
    fit = {cid for cid, _, _ in seed.clip_margins("01")}
    assert fit == {"iron/pos_00", "iron/neg_01"}
    assert {cid for cid, _, _ in seed.clip_margins("01", include_heldout=True)} == {c for c, *_ in (
        ("iron/pos_00",), ("iron/neg_01",), ("iron-inview/pos_02",), ("iron-day2/pos_03",))}


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_import_refuses_two_clips_with_one_number(tmp_path, monkeypatch):
    monkeypatch.setenv("JAGA_HOME", str(tmp_path / "home"))
    seed = load_script("seed")
    monkeypatch.setattr(seed, "LOCAL_CLIPS", tmp_path / "home" / "clips")
    monkeypatch.setattr(seed, "CLIPS_CSV", tmp_path / "clips.csv")
    raw = tmp_path / "raw"
    (raw / "iron").mkdir(parents=True)
    write_test_video(raw / "iron" / "pos_07_day_board.mp4", 9.0)
    write_test_video(raw / "iron" / "pos_07_lamp_board.mp4", 9.0)

    a = SimpleNamespace(raw=str(raw), window="middle", force=False)
    assert seed.cmd_import(a) == 1  # a collision is an error, not a silent drop
    rows = list(csv.DictReader((tmp_path / "clips.csv").open()))
    assert [r["clip_id"] for r in rows] == ["iron/pos_07"]
    # a later import of a different clip under the same number is refused too
    (raw / "iron" / "pos_07_day_board.mp4").unlink()
    assert seed.cmd_import(a) == 1


# ------------------------------------------------------------------- clips: orientation and leave-one-out
def test_frame_difference_orientation_is_fixed_per_tripwire(bench):
    assert bench.MOTION_ORIENTATION == {"01": "stillness", "02": "stillness", "03": "stillness", "04": "motion"}


def test_leave_one_out_judges_each_positive_without_itself(bench):
    pos, neg = [0.30, 0.20, 0.10], [0.05, 0.25]
    dp, dn = bench.loo_decisions(pos, neg, recall=0.9)
    # without 0.30, θ = 0.10 → 0.30 above; without 0.10, θ = 0.20 → 0.10 below (the full-data θ would say above)
    assert [a for _, a in dp] == [True, True, False]
    assert [th for th, _ in dp] == pytest.approx([0.10, 0.10, 0.20])
    assert [a for _, a in dn] == [False, True]  # negatives at the full-data θ 0.10
    c = bench.loo_counts(pos, neg)
    assert (c["tp"], c["fn"], c["fp"], c["tn"]) == (2, 1, 1, 1)


# ------------------------------------------------------------------- Soak B per exit type, other cuts, dwell keys
def _soak_b_db(tmp_path):
    from jaga.db import DB

    db = DB(tmp_path / "b.db")
    tid = db.upsert_tripwire("01", "nobody is at the ironing board", {"name": "X"}, "req", "m", "now")
    db.add_threshold(tid, 1, "labelled-m", 0.0, None, None, "loo-clips", "B1")
    db.add_threshold(tid, 2, "labelled-m", None, None, None, "nebius-job-proposed:x")  # never used for scoring
    db.start_session("B1", True, "board", "live")
    t0 = 1000.0
    for kind, a, b in (("exit_leave", 100, 300), ("exit_inview", 400, 600)):
        db.add_label("B1", f"plan:{kind}", ts=t0 + a)
        db.add_label("B1", f"plan_end:{kind}", ts=t0 + b)
    for i in range(900):
        ts = t0 + i
        away = 110 <= i < 300 or 410 <= i < 600
        eid = db.add_embed(ts, "B1", True, 0.5, 0.1 if away else 3.0, 0.0, 0 if away else 1, 1.0, "board", 900, "live")
        db.add_win(eid, ts, tid, 0.05 if away else -0.05, None, None, None, 1, 900, False, "armed", away)
    db.add_event("trip", "jaga", tid, session="B1", ts=t0 + 250)  # the leave exit
    db.add_event("trip", "jaga", tid, session="B1", ts=t0 + 700)  # during an activity: an "other" false cut
    return db


def test_soak_b_reports_recall_per_exit_type_and_other_false_cuts(bench, tmp_path):
    db = _soak_b_db(tmp_path)
    r = bench.soak_rows(db, "B1", "01", dwell_s=120, k_fraction=0.8, p8=0.5, r_thr=None)
    live = r["systems"]["jaga (live)"]
    assert live["exit_kinds"] == ["exit_leave", "exit_inview"] and live["caught_flags"] == [True, False]
    assert live["other_false_cuts"] == 1
    assert r["theta"] == 0.0  # the NULL-θ job proposal is skipped


def test_secondary_dwell_never_overwrites_the_primary_section(bench, tmp_path):
    import json

    db = _soak_b_db(tmp_path)

    a = SimpleNamespace(session=["B1"], tripwire="01", k_fraction=0.8, p8=0.5, r=None, db=str(tmp_path / "b.db"))
    a.dwell, a.secondary = 120.0, False
    bench.cmd_soak(a)
    a.dwell, a.secondary = 30.0, True
    bench.cmd_soak(a)
    s = json.loads(bench.SUMMARY.read_text())
    assert s["soak_b_dwell_120"]["primary"] and not s["soak_b_dwell_30"]["primary"]
    names = {r["system"] for r in s["soak_b_dwell_30"]["pooled"]}
    assert "jaga (live)" not in names and "jaga (recomputed)" in names  # live trips happened at 120 s only
    row = next(r for r in s["soak_b_dwell_120"]["pooled"] if r["system"] == "jaga (live)")
    assert row["recall_by_exit_type"]["exit_leave"]["caught"] == 1
    assert row["other_false_cuts"] == 1 and row["other_false_cuts_per_hour_ci95"][1] > 0
    assert "Dwell 120 s (primary)" in bench.DEMO.read_text() and "Dwell 30 s (secondary" in bench.DEMO.read_text()
    assert db.query("SELECT COUNT(*) AS n FROM embed")[0]["n"] == 900


# ------------------------------------------------------------------- Soak A
def test_soak_a_counts_false_alarms_per_armed_hour_and_positives(bench, tmp_path, monkeypatch):
    import json

    from jaga.db import DB

    db = DB(tmp_path / "a.db")
    tid = db.upsert_tripwire("03", "the door is left open and nobody is there", {"name": "DOOR"}, "r", "m", "now")
    db.add_threshold(tid, 1, "labelled-m", 0.0, None, None, "loo-clips", "A1")
    db.start_session("A1", True, "kitchen", "live")
    t0 = 0.0
    db.add_label("A1", "left_open", ts=t0 + 1000)
    db.add_label("A1", "closed", ts=t0 + 1200)
    for i in range(3600):  # one hour at 1 s; unarmed (too dark) for the last 600 s
        ts = t0 + i
        state = "capturing" if i < 120 else ("armed" if i < 3000 else "unarmed")
        door = 1000 <= i < 1200
        eid = db.add_embed(ts, "A1", True, 0.5, 2.0 if i % 997 == 0 else 0.0, None, None, None, "kitchen", 900, "live")
        db.add_win(eid, ts, tid, 0.05 if door else -0.05, None, None, None, 1, 900, False, state, door)
    db.add_event("trip", "jaga", tid, session="A1", ts=t0 + 1070)  # the left-open door
    db.add_event("trip", "jaga", tid, session="A1", ts=t0 + 2500)  # a false alarm

    r = bench.soak_a_rows(db, "A1", "03", p8=0.5, dwell_s=None)
    assert r["dwell_s"] == 60 and r["armed_hours"] == pytest.approx(2880 / 3600, abs=0.01)
    live = r["systems"]["jaga (live)"]
    assert live["false_alarms"] == 1 and live["positives_caught"] == 1
    assert r["systems"]["jaga (recomputed)"]["positives_caught"] == 1
    assert r["systems"]["jaga (recomputed)"]["false_alarms"] == 0
    # frame difference through the same rule: a still empty room trips the stillness reading over and over
    assert r["systems"]["frame difference (stillness)"]["false_alarms"] >= 3
    assert r["systems"]["frame difference (motion)"]["trips"] == 0

    bench.cmd_soak_a(SimpleNamespace(session=["A1"], tripwire="03", p8=0.5, dwell=None, db=str(tmp_path / "a.db")))
    s = json.loads(bench.SUMMARY.read_text())["soak_a"]["tripwires"]["03"]["table"]
    live = next(x for x in s if x["system"] == "jaga (live)")
    rate, lo, hi = live["per_armed_hour_ci95"]
    assert rate == pytest.approx(1 / live["armed_hours"]) and lo < rate < hi


# ------------------------------------------------------------------- fresh mode hold-out (non-vacuous)
def test_fresh_holdout_is_scored_at_a_dwell_that_fits_inside_it(bench, tmp_path, monkeypatch):
    caps = Captions(5)
    home = Path(str(tmp_path / "home"))
    (home / "normal").mkdir(parents=True)
    (home / "normal" / "iron.mp4").write_bytes(b"")
    n_ts = np.arange(0, 120, 1.0)
    # the second minute drifts toward the positives: a hold-out that must count against the sentence
    n_v = np.stack([caps.window("neg") if t < 60 else caps.window("pos") for t in n_ts])
    monkeypatch.setattr(bench, "load_summary", lambda: {"period": {"period_p50_s": 1.0}})
    monkeypatch.setattr(bench, "caption_vectors", lambda slot: ({"operating": {"dwell_s": 120}}, caps.pos, caps.neg,
                                                                None))
    monkeypatch.setattr(bench, "read_clips", lambda: [])
    import jaga.embed
    import jaga.offline

    monkeypatch.setattr(jaga.embed, "shared", lambda device="auto": None)
    monkeypatch.setattr(jaga.offline, "embed_long", lambda path, emb, stride: (n_ts, n_v))
    saved = {}
    monkeypatch.setattr(bench, "save_section", lambda name, data: saved.setdefault(name, data))
    monkeypatch.setattr(bench, "write_demo", lambda: None)

    bench.cmd_fresh(SimpleNamespace(stride=None, device="cpu"))
    row = saved["fresh"]["tripwires"]["01"]
    assert row["holdout_dwell_s"] == 30.0 and row["stride_s"] == 1
    assert row["holdout_windows_above"] == row["n_holdout"] == 60
    assert row["holdout_trips"] >= 1  # at the 120-s operating dwell this would have been 0, whatever the scores


# ------------------------------------------------------------------- P7 rule (PROTOCOL §6)
def test_p7_keeps_0_8_unless_a_logged_exit_is_missed(bench, tmp_path):
    from jaga.db import DB

    db = DB(tmp_path / "u.db")
    tid = db.upsert_tripwire("01", "nobody is at the ironing board", {"name": "X"}, "r", "m", "now")
    db.add_threshold(tid, 1, "labelled-m", 0.0, None, None, "loo-clips")
    db.start_session("U1", False, "board", "live")
    emb, t0 = [], 0.0
    for i in range(1000):
        ts = t0 + i
        away = 200 <= i < 340  # one exit, 140 s, but every 4th window reads "someone there"
        above = away and i % 4 != 0
        eid = db.add_embed(ts, "U1", False, 0.5, 0.0, 0.0, None, None, "board", 900, "live")
        db.add_win(eid, ts, tid, 0.05 if above else -0.05, None, None, None, 1, 900, False, "armed", above)
        emb.append({"id": eid, "ts": ts})
    p7, rows = bench.p7_candidate(db, "U1", "01", 120.0, emb, [t0 + 200], [t0 + 340])
    by_k = {r["k_fraction"]: r["exits_caught"] for r in rows}
    assert by_k[0.8] == 0 and by_k[0.7] == 1 and p7 == 0.7


# ------------------------------------------------------------------- DEMO wording
def test_demo_says_timed_calls_and_load_honestly(bench):
    import json

    bench.OUT.mkdir(parents=True, exist_ok=True)
    bench.SUMMARY.write_text(json.dumps({"speed": {"rows": [
        {"device": "mps", "n": 200, "p50_s": 0.86, "p95_s": 0.92, "torch": "2", "threads": 8,
         "loadavg_1m_before": None}], "input": "javelin_throw.webm", "_machine": {"loadavg_1m": 37.1}}}))
    bench.write_demo()
    t = bench.DEMO.read_text()
    assert "200 timed calls on one window" in t and "not recorded during the run" in t
    assert "load average when the result was written: 37.1" in t
    assert "data/clips/iron/pos_03.mp4" not in t  # the replay command names a clip id, once clips exist


# ------------------------------------------------------------------- QA round 2: schedule end, countdown, keys, inputs
def test_the_plug_countdown_restarts_at_the_next_step_after_its_own_cut(bench):
    cuts = bench.countdown_cuts(0.0, 3600.0, 15, resume_at=[600.0, 1200.0, 1500.0, 2400.0])
    # on at 0 → cut 900; back on at the next step (1200) → cut 2100; back on at 2400 → cut 3300
    assert cuts == [900.0, 2100.0, 3300.0]
    assert bench.countdown_cuts(0.0, 600.0, 15, resume_at=[]) == []


def test_nothing_after_the_schedule_end_is_scored(bench, tmp_path):
    db = _soak_b_db(tmp_path)
    db.add_label("B1", "plan:session_end", ts=1000.0 + 650)  # the schedule ended before the 700-s "other" trip
    r = bench.soak_rows(db, "B1", "01", dwell_s=120, k_fraction=0.8, p8=0.5, r_thr=None)
    assert r["systems"]["jaga (live)"]["other_false_cuts"] == 0 and r["hours"] < 660 / 3600
    assert r["windows"] <= 652


def test_an_unclosed_left_open_key_closes_after_fifteen_minutes_and_is_reported(bench):
    labels = [{"ts": 100.0, "kind": "left_open"}, {"ts": 5000.0, "kind": "left_open"}, {"ts": 5100.0, "kind": "closed"},
              {"ts": 6000.0, "kind": "closed"}]
    pos, unpaired = bench.intervals(labels, "left_open", "closed")
    assert pos == [(100.0, 100.0 + bench.MAX_OPEN_S), (5000.0, 5100.0)]
    assert {u["missing"] for u in unpaired} == {"closed", "left_open"}


def test_a_mistyped_session_never_overwrites_a_computed_result(bench, tmp_path):
    import json

    db = _soak_b_db(tmp_path)
    a = SimpleNamespace(session=["B1"], tripwire="01", k_fraction=0.8, p8=0.5, r=None, db=str(tmp_path / "b.db"),
                        dwell=120.0, secondary=False)
    assert bench.cmd_soak(a) == 0
    before = json.loads(bench.SUMMARY.read_text())["soak_b_dwell_120"]["pooled"]
    a.session = ["B11"]
    assert bench.cmd_soak(a) == 1
    assert json.loads(bench.SUMMARY.read_text())["soak_b_dwell_120"]["pooled"] == before
    del db


def test_soak_scoring_requires_the_frozen_inputs_and_records_them(bench, tmp_path):
    import json
    import sys

    _soak_b_db(tmp_path)
    monkey_argv = ["bench.py", "--db", str(tmp_path / "b.db"), "soak", "--session", "B1"]
    old = sys.argv
    try:
        sys.argv = monkey_argv
        with pytest.raises(SystemExit):  # no --k-fraction / --p8: refused, never defaulted
            bench.main()
        sys.argv = monkey_argv + ["--k-fraction", "0.7", "--p8", "0.4", "--r", "2.5"]
        assert bench.main() == 0
    finally:
        sys.argv = old
    sec = json.loads(bench.SUMMARY.read_text())["soak_b_dwell_120"]
    assert sec["inputs"]["P7_k_fraction"] == 0.7 and sec["inputs"]["P8"] == 0.4 and sec["inputs"]["R"] == 2.5
    assert "15 min" in sec["plug_countdown"] and "Frozen inputs: P7" in bench.DEMO.read_text()


def test_leave_one_out_with_a_single_positive_is_a_miss(bench):
    dp, _ = bench.loo_decisions([0.4], [0.1, 0.2])
    assert dp == [(float("inf"), False)]


def test_freeze_theta_keeps_the_other_tripwires_published_rows(tmp_path, monkeypatch):
    import json

    seed = load_script("seed")
    tw = tmp_path / "tw"
    tw.mkdir()
    for slot in ("01", "02"):
        (tw / f"{slot}.json").write_text(json.dumps({"slot": slot, "frozen": None}))
    monkeypatch.setattr(seed, "TRIPWIRES", tw)
    monkeypatch.setattr(seed, "DECISIONS_FROZEN", tmp_path / "decisions_frozen.csv")
    monkeypatch.setattr(seed, "clip_margins", lambda slot: [(f"{slot}/pos_01", 1, 0.3), (f"{slot}/neg_01", 0, 0.1)])
    seed.cmd_freeze_theta(SimpleNamespace(slots="01", force=False, recall=0.9, version=1, frozen_for="B1"))
    seed.cmd_freeze_theta(SimpleNamespace(slots="01,02", force=False, recall=0.9, version=1, frozen_for="B1"))
    rows = list(csv.DictReader((tmp_path / "decisions_frozen.csv").open()))
    assert sorted({r["tripwire"] for r in rows}) == ["01", "02"] and len(rows) == 4


def test_readiness_staleness_follows_the_code_each_section_measured(monkeypatch, tmp_path):
    import json

    ck = load_script("check_submission_readiness")
    out = tmp_path / "summary.json"
    out.write_text(json.dumps({"speed": {"_commit": "abc1234"}, "period": {"_commit": "abc1234"},
                               "clips": {"_commit": "abc1234"}, "fresh": {"_commit": "def5678+dirty"}}))
    calls = []

    def fake_git(*args):
        calls.append(args)
        if args[0] == "diff":  # only bench.py changed since abc1234
            return "scripts/bench.py\n" if "scripts/bench.py" in args else ""
        return ""  # git log -L: cmd_speed / cmd_period untouched

    monkeypatch.setattr(ck, "git", fake_git)
    monkeypatch.setattr(ck, "git_ok", lambda *args: (not any("def5678" in x for x in args), fake_git(*args)))
    monkeypatch.setattr(ck, "REPO", tmp_path)
    (tmp_path / "bench" / "out").mkdir(parents=True)
    (tmp_path / "bench" / "out" / "summary.json").write_text(out.read_text())
    (tmp_path / "DEMO.md").write_text("This file is written by `scripts/bench.py`\n")
    monkeypatch.setattr(ck, "collected_tests", lambda: 0)
    res = {n: (ok, why) for n, ok, why in ck.check(None, {"video"})}
    assert not res["numbers"][0]
    assert "clips@abc1234" in res["numbers"][1] and "fresh@def5678+dirty" in res["numbers"][1]
    assert "speed@" not in res["numbers"][1] and "period@" not in res["numbers"][1]


def test_demo_marks_speed_rows_measured_on_a_busy_machine(bench):
    import json

    bench.OUT.mkdir(parents=True, exist_ok=True)
    bench.SUMMARY.write_text(json.dumps({
        "speed": {"rows": [{"device": "mps", "n": 200, "p50_s": 2.7, "p95_s": 5.7, "torch": "2", "threads": 8,
                            "loadavg_1m_before": 3.8, "loadavg_1m_after": 57.3},
                           {"device": "cpu", "n": 200, "p50_s": 4.3, "p95_s": 12.1, "torch": "2", "threads": 8,
                            "loadavg_1m_before": 2.0, "loadavg_1m_after": 3.0}],
                  "input": "javelin_throw.webm", "_commit": "abc", "_machine": {"cpu_count": 10, "loadavg_1m": 9}},
        "period": {"windows": 283, "period_p50_s": 1.07, "period_p95_s": 1.34, "embed_p50_s": 0.90, "session": "p",
                   "_commit": "abc", "_machine": {}}}))
    bench.write_demo()
    t = bench.DEMO.read_text()
    assert "The 1-min load average went above the machine's 10 cores during the MPS row** (MPS 3.8 → 57.3)" in t
    assert "live loop's own embed time was p50 0.90 s" in t and "CPU row" not in t


def test_receipts_render_with_partial_bench_data(home, tuning):
    from conftest import make_watcher

    from jaga import receipts

    w, _ = make_watcher(home, tuning)
    d = receipts.collect(w)
    d["bench"] = {"speed": {"rows": [{"device": "mps", "p50_s": 1.0}]}, "period": {"period_p50_s": None}}
    html = receipts.render_html(d)
    assert "MPS fp32" in html and "Live window period" not in html


def test_the_overlay_says_lost_contact_instead_of_looking_armed():
    js = (REPO / "jaga" / "static" / "overlay.js").read_text()
    assert "LOST CONTACT" in js and "S.lost = true" in js and "LOST_AFTER_MS" in js
    css = (REPO / "jaga" / "static" / "overlay.css").read_text()
    assert "body.lost .panel" in css



def test_a_partly_mistyped_session_list_writes_nothing(bench, tmp_path):
    import json

    db = _soak_b_db(tmp_path)
    a = SimpleNamespace(session=["B1"], tripwire="01", k_fraction=0.8, p8=0.5, r=None, db=str(tmp_path / "b.db"),
                        dwell=120.0, secondary=False)
    assert bench.cmd_soak(a) == 0
    before = json.loads(bench.SUMMARY.read_text())["soak_b_dwell_120"]["sessions"]
    a.session = ["B1", "B22"]
    assert bench.cmd_soak(a) == 1
    assert json.loads(bench.SUMMARY.read_text())["soak_b_dwell_120"]["sessions"] == before
    del db


def test_readiness_fails_closed_when_git_cannot_answer(monkeypatch, tmp_path):
    import json

    ck = load_script("check_submission_readiness")
    (tmp_path / "bench" / "out").mkdir(parents=True)
    (tmp_path / "bench" / "out" / "summary.json").write_text(json.dumps({"speed": {"_commit": "0000000"}}))
    (tmp_path / "DEMO.md").write_text("This file is written by `scripts/bench.py`\n")
    monkeypatch.setattr(ck, "REPO", tmp_path)
    monkeypatch.setattr(ck, "git", lambda *a: "")
    monkeypatch.setattr(ck, "git_ok", lambda *a: (False, ""))  # unknown commit / renamed function
    monkeypatch.setattr(ck, "collected_tests", lambda: 0)
    res = {n: (ok, why) for n, ok, why in ck.check(None, {"video"})}
    assert not res["numbers"][0] and "speed@0000000" in res["numbers"][1]


# ------------------------------------------------------------------- QA round 4
def test_period_with_a_mistyped_session_writes_nothing(bench, tmp_path):
    import json

    from jaga.db import DB

    db = DB(tmp_path / "p.db")
    for i in range(50):
        db.add_embed(1000.0 + i, "P1", False, 0.5, 0.0, 0.0, None, None, "qa", 900, "live")
    assert bench.cmd_period(SimpleNamespace(db=str(tmp_path / "p.db"), session="P1", input_note="x")) == 0
    before = json.loads(bench.SUMMARY.read_text())["period"]
    assert bench.cmd_period(SimpleNamespace(db=str(tmp_path / "p.db"), session="P11", input_note="x")) == 1
    assert json.loads(bench.SUMMARY.read_text())["period"] == before
    empty = SimpleNamespace(db=str(tmp_path / "empty.db"), session=None, input_note=None)
    assert bench.cmd_period(empty) == 1  # an empty database is said, not an IndexError


def test_outside_with_a_mistyped_root_keeps_the_computed_result(bench, tmp_path):
    import json

    bench.save_section("outside", {"k": 6, "n": 8, "ci95": [0.35, 0.97], "received": 8})
    assert bench.cmd_outside(SimpleNamespace(root=str(tmp_path / "nope"), stride=1.0, device="cpu")) == 1
    assert json.loads(bench.SUMMARY.read_text())["outside"]["k"] == 6


def test_soak_a_counts_a_session_named_twice_once(bench, tmp_path):
    import json

    from jaga.db import DB

    db = DB(tmp_path / "a.db")
    tid = db.upsert_tripwire("03", "the door is left open and nobody is there", {"name": "DOOR"}, "r", "m", "now")
    for i in range(400):
        eid = db.add_embed(float(i), "A1", True, 0.5, 0.0, None, None, None, "kitchen", 900, "live")
        db.add_win(eid, float(i), tid, -0.05, None, None, None, 1, 900, False, "capturing" if i < 60 else "armed",
                   False)
    a = SimpleNamespace(session=["A1", "A1"], tripwire="03", p8=0.5, dwell=None, db=str(tmp_path / "a.db"))
    assert bench.cmd_soak_a(a) == 0
    d = json.loads(bench.SUMMARY.read_text())["soak_a"]["tripwires"]["03"]
    assert d["sessions"] == ["A1"]
    del db


def test_readiness_counts_an_unstamped_result_as_stale(monkeypatch, tmp_path):
    import json

    ck = load_script("check_submission_readiness")
    (tmp_path / "bench" / "out").mkdir(parents=True)
    (tmp_path / "bench" / "out" / "summary.json").write_text(json.dumps({
        "speed": {"_commit": "unknown"}, "soak_a": {"status": "not yet run", "_commit": "abc"}}))
    (tmp_path / "DEMO.md").write_text("This file is written by `scripts/bench.py`\n")
    monkeypatch.setattr(ck, "REPO", tmp_path)
    monkeypatch.setattr(ck, "git", lambda *a: "")
    monkeypatch.setattr(ck, "git_ok", lambda *a: (True, ""))
    monkeypatch.setattr(ck, "collected_tests", lambda: 0)
    res = {n: (ok, why) for n, ok, why in ck.check(None, {"video"})}
    assert not res["numbers"][0] and "speed@unknown" in res["numbers"][1] and "soak_a" not in res["numbers"][1]


def test_the_protocol_script_checks_without_writing_and_refuses_unknown_sessions(tmp_path):
    import subprocess
    import sys

    before = sorted(p.name for p in (REPO / "docs" / "protocol").iterdir())
    ok = subprocess.run([sys.executable, str(REPO / "scripts" / "protocol.py"), "--check"], capture_output=True,
                        text=True, timeout=60)
    bad = subprocess.run([sys.executable, str(REPO / "scripts" / "protocol.py"), "X9"], capture_output=True, text=True,
                         timeout=60)
    assert ok.returncode == 0 and "reproduced exactly" in ok.stdout and bad.returncode != 0
    assert sorted(p.name for p in (REPO / "docs" / "protocol").iterdir()) == before
