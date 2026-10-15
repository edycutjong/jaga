#!/usr/bin/env python3
"""jaga's measurements. Every number on any page, slide, caption or voice-over comes from this script.

    bench.py speed            s/clip p50/p95 of Cosmos-Embed1 on this machine (MPS and CPU, fp32), N windows
    bench.py period           the live window period N = p50/p95 of diff(embed.ts) from a logged session
    bench.py clips            per tripwire on the labelled clips: AUC (Hanley–McNeil CI), leave-one-out θ at recall 0.9
                              → precision counts; vs frame difference and positives-only; in-view, day-2, lighting
    bench.py fresh            the zero-training row: μ₀/σ₀ from the first 60 s of normal/<slot>, default θ_z;
                              the held-out 60 s scored at a 30-s dwell, plus the share of its windows above θ_z
    bench.py soak --session B1 [--session B2 …] [--dwell 30 --secondary]
                              Soak B (pre-registered, docs/PROTOCOL.md): still episodes cut, exits caught per exit
                              type, other false cuts per attended hour; one section per dwell
    bench.py soak-a --session A1
                              Soak A: false alarms per armed hour (exact Poisson) vs frame difference through the
                              same rule, positives (keys 6/7) caught, and the stored vectors re-scored in fresh mode
    bench.py hog --zone x0,y0,x1,y1
                              the HOG detection rate on the person-present iron clips (PROTOCOL §6)
    bench.py ablate           (a) no hard negatives, (b) one global θ, (c) raw sentence, (c-id) raw Indonesian,
                              (d) single frame, (e) compile variance
    bench.py outside --root DIR
                              K of N outside people's tripwires that tripped correctly first time
    bench.py tune --session U1  candidate P7/P8/R from an unscored session (for docs/PROTOCOL.md)
    bench.py demo             rewrite DEMO.md from bench/out/summary.json

Raw rows go to bench/out/*.csv; summary.json keeps each section with the commit and machine it ran on.
A section that has not run is written as "not yet run" — never estimated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from jaga import config  # noqa: E402
from jaga import stats as S  # noqa: E402
from jaga.offline import CLIPS_CSV, committed_vector  # noqa: E402
from jaga.score import margins, replay_rule, theta_at_recall, window_period  # noqa: E402

OUT = REPO / "bench" / "out"
SUMMARY = OUT / "summary.json"
DEMO = REPO / "DEMO.md"
LOCAL_CLIPS = config.jaga_home() / "clips"
JAVELIN = Path.home() / ".cache" / "jaga" / "javelin_throw.webm"


# ----------------------------------------------------------------------------------- bookkeeping
def git_head() -> str:
    """The commit a result came from; '+dirty' when tracked files differ from it (the code that ran is not it)."""
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no", "--", "jaga",
                                "scripts", "jobs"], capture_output=True, text=True, timeout=10).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def machine() -> dict:
    proc = os.popen("sysctl -n machdep.cpu.brand_string 2>/dev/null").read().strip() or platform.processor()
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {"processor": proc, "platform": platform.platform(), "python": sys.version.split()[0],
            "loadavg_1m": None if load is None else round(load[0], 2), "cpu_count": os.cpu_count()}


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}


def save_section(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_summary()
    s[name] = {**data, "_commit": git_head(), "_ran_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "_machine": machine()}
    SUMMARY.write_text(json.dumps(s, indent=2, ensure_ascii=False, default=float) + "\n")


def write_csv(name: str, rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    if not rows:
        p.write_text("")
        return p
    cols = list(rows[0])
    for r in rows[1:]:
        cols += [k for k in r if k not in cols]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return p


def pct(values, q):
    a = sorted(values)
    k = (len(a) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(a) - 1)
    return a[lo] + (a[hi] - a[lo]) * (k - lo)


# ----------------------------------------------------------------------------------- speed
def cmd_speed(a) -> int:
    import av
    import torch

    from jaga.embed import Embedder
    from jaga.frames import canon

    src = Path(a.clip) if a.clip else JAVELIN
    if not src.exists():
        raise SystemExit(f"{src} missing — run scripts/day0_cosmos.py once to fetch the card's sample video")
    with av.open(str(src)) as c:
        frames = [canon(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]
    ids = np.linspace(0, len(frames) - 1, 8, dtype=int)
    window = np.stack([frames[i] for i in ids])
    rows = []
    for dev in a.devices.split(","):
        if dev == "mps" and not torch.backends.mps.is_available():
            continue
        emb = Embedder(dev)
        for _ in range(5):
            emb.embed_video(window)
        load_before = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
        times = []
        for _ in range(a.n):
            t0 = time.perf_counter()
            emb.embed_video(window)  # preprocess + get_video_embeddings + device sync: what one live window pays
            times.append(time.perf_counter() - t0)
        info = emb.info
        rows.append({"device": dev, "dtype": "float32", "n": a.n, "p50_s": round(statistics.median(times), 4),
                     "p95_s": round(pct(times, 0.95), 4), "mean_s": round(statistics.mean(times), 4),
                     "min_s": round(min(times), 4), "max_s": round(max(times), 4), "torch": info["torch"],
                     "threads": info["torch_threads"], "input": src.name, "load_s": info["load_s"],
                     "loadavg_1m_before": None if load_before is None else round(load_before, 2),
                     "loadavg_1m_after": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None})
        print(f"{dev}: p50 {rows[-1]['p50_s']} s · p95 {rows[-1]['p95_s']} s over {a.n} windows")
        del emb
    write_csv("speed.csv", rows)
    best = min(rows, key=lambda r: r["p50_s"])
    save_section("speed", {"rows": rows, "best_device": best["device"], "input": src.name,
                           "note": "per window: preprocess + get_video_embeddings, 5 warm-up calls first; other "
                                   "processes on the machine are not stopped (see _machine.loadavg_1m)"})
    write_demo()
    return 0


def cmd_period(a) -> int:
    from jaga.db import DB

    db = DB(a.db) if a.db else DB()
    sess = a.session or db.query("SELECT session FROM embed ORDER BY id DESC LIMIT 1")[0]["session"]
    rows = db.query("SELECT ts, embed_ms, source FROM embed WHERE session=? ORDER BY ts", (sess,))
    ts = [r["ts"] for r in rows]
    p50, p95 = window_period(ts)
    emb = [r["embed_ms"] / 1000 for r in rows if r["embed_ms"]]
    src = db.query("SELECT source, camera_pos, note FROM session WHERE name=?", (sess,))
    data = {"session": sess, "windows": len(ts), "period_p50_s": p50, "period_p95_s": p95,
            "embed_p50_s": statistics.median(emb) if emb else None, "embed_p95_s": pct(emb, 0.95) if emb else None,
            "source": src[0]["source"] if src else None, "input": a.input_note or (src[0]["note"] if src else "")}
    print(json.dumps(data, indent=2))
    write_csv("period.csv", [{"ts": t} for t in ts])
    save_section("period", data)
    write_demo()
    return 0


# ----------------------------------------------------------------------------------- clips
def read_clips() -> list[dict]:
    if not CLIPS_CSV.exists():
        return []
    with CLIPS_CSV.open() as f:
        return list(csv.DictReader(f))


def caption_vectors(slot: str):
    from jaga.tripwires import frozen_threshold, load, load_vectors

    fx = load(slot)
    pos, neg = load_vectors(fx)
    return fx, pos, neg, frozen_threshold(fx)


# Frame-difference baseline orientation, fixed before any clip is scored (docs/PROTOCOL.md "Clips"):
# 01-03 are "nobody there" states — a still scene — so the classical detector is a stillness score (−motion);
# a cat on the counter (04) arrives with motion.
MOTION_ORIENTATION = {"01": "stillness", "02": "stillness", "03": "stillness", "04": "motion"}


def motion_score(clip_id: str) -> float | None:
    """Frame difference over the 8-s window: mean % of pixels changed between consecutive 1-fps frames."""
    from jaga.frames import gray_small, read_window
    from jaga.guards import changed_pct

    p = LOCAL_CLIPS / f"{clip_id}.mp4"
    if not p.exists():
        return None
    frames, _ = read_window(p, "start")
    g = [gray_small(f) for f in frames]
    return float(np.mean([changed_pct(g[i], g[i - 1], config.TUNING.motion_pixel_delta) for i in range(1, len(g))]))


def loo_decisions(scores_pos, scores_neg, recall=0.9) -> tuple[list, list]:
    """Leave-one-out at recall 0.9: each clip is judged by a θ fitted without it.

    θ depends only on the positives (the highest margin keeping recall ≥ 0.9), so a negative's own θ is the
    full-data θ — leaving a negative out does not change it. Returns (θ, above) per positive and per negative."""
    sp, sn = list(scores_pos), list(scores_neg)
    pos = []
    for i, s in enumerate(sp):
        others = sp[:i] + sp[i + 1:]
        th = theta_at_recall(others, recall) if others else math.inf  # nothing left to fit θ on: a miss
        pos.append((th, s >= th))
    th_all = theta_at_recall(sp, recall) if sp else math.inf
    return pos, [(th_all, s >= th_all) for s in sn]


def loo_counts(scores_pos, scores_neg, recall=0.9) -> dict:
    pos, neg = loo_decisions(scores_pos, scores_neg, recall)
    tp = sum(a for _, a in pos)
    fp = sum(a for _, a in neg)
    fn, tn = len(pos) - tp, len(neg) - fp
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if not pos else tp / len(pos)}


def system_row(name, sp, sn) -> dict:
    if not sp or not sn:
        return {"system": name, "n_pos": len(sp), "n_neg": len(sn), "status": "not enough clips"}
    a = S.auc(sp, sn)
    ci = S.hanley_mcneil(a, len(sp), len(sn))
    return {"system": name, "n_pos": len(sp), "n_neg": len(sn), "auc": round(a, 4),
            "auc_ci95": None if ci is None else [round(ci[0], 4), round(ci[1], 4)], **loo_counts(sp, sn)}


def cmd_clips(a) -> int:
    clips = read_clips()
    if not clips:
        save_section("clips", {"status": "not yet run", "why": "no labelled clips in fixtures/clips.csv"})
        write_demo()
        print("no labelled clips yet")
        return 0
    out_rows, sections, loo_rows = [], {}, []
    for slot in sorted({c["tripwire"].zfill(2) for c in clips}):
        fx, pos, neg, th = caption_vectors(slot)
        main = [c for c in clips if c["tripwire"].zfill(2) == slot and c["heldout"] != "1"
                and c["set"] not in ("iron-inview",)]
        recs = []
        for c in main:
            v = committed_vector(c["clip_id"])
            if v is None:
                continue
            m = float(margins(v[None], pos, neg)[0])
            mp = float(margins(v[None], pos, np.zeros((0, 256)))[0])
            mo = motion_score(c["clip_id"])
            recs.append({"clip_id": c["clip_id"], "label": int(c["label"]), "lighting": c["lighting"], "m": m,
                         "pos_only": mp, "motion_pct": mo})
        if not recs:
            continue
        orient = MOTION_ORIENTATION.get(slot, "stillness")
        P = [r for r in recs if r["label"] == 1]
        N = [r for r in recs if r["label"] == 0]
        rows = [system_row("jaga", [r["m"] for r in P], [r["m"] for r in N]),
                system_row("positives only (no hard negatives)", [r["pos_only"] for r in P],
                           [r["pos_only"] for r in N])]
        paired = None
        if P and N:
            jp, jn = loo_decisions([r["m"] for r in P], [r["m"] for r in N])
            for r, (th_r, above) in zip(P + N, jp + jn, strict=True):
                r["jaga_theta_loo"], r["jaga_above_loo"] = th_r, int(above)
                loo_rows.append({"tripwire": slot, "clip_id": r["clip_id"], "label": r["label"], "m": f"{r['m']:.9f}",
                                 "theta_loo": f"{th_r:.9f}", "above_loo": int(above)})
            if all(r["motion_pct"] is not None for r in recs):
                sign = -1.0 if orient == "stillness" else 1.0
                rows.append(system_row(f"frame difference ({orient})", [sign * r["motion_pct"] for r in P],
                                       [sign * r["motion_pct"] for r in N]))
                fp_, fn_ = loo_decisions([sign * r["motion_pct"] for r in P], [sign * r["motion_pct"] for r in N])
                fd = [int(a) for _, a in fp_ + fn_]
                truth = [1] * len(P) + [0] * len(N)
                jj = [r["jaga_above_loo"] for r in P + N]
                b = sum(1 for x, y, t in zip(jj, fd, truth, strict=True) if (x == t) and (y != t))
                c = sum(1 for x, y, t in zip(jj, fd, truth, strict=True) if (y == t) and (x != t))
                paired = {"jaga_right_baseline_wrong": b, "baseline_right_jaga_wrong": c,
                          "sign_test_p": S.sign_test(b, c)}
        lighting = {}
        for r in recs:
            d = lighting.setdefault(r["lighting"], {"pos": 0, "neg": 0, "pos_above": 0, "neg_above": 0})
            k = "pos" if r["label"] else "neg"
            d[k] += 1
            d[f"{k}_above"] += r.get("jaga_above_loo", 0)
        extra = {}
        if th is not None:
            for set_name, label in (("iron-inview", "in view, away from the board"), ("iron-day2", "held-out day 2")):
                sub = [c for c in clips if c["set"] == set_name and c["tripwire"].zfill(2) == slot]
                vs = [(c, committed_vector(c["clip_id"])) for c in sub]
                vs = [(c, v) for c, v in vs if v is not None]
                if vs:
                    ms = [(int(c["label"]), float(margins(v[None], pos, neg)[0])) for c, v in vs]
                    extra[set_name] = {"label": label, "theta_m": th.theta, "theta_version": th.version,
                                       **S.confusion([m for y, m in ms if y == 1], [m for y, m in ms if y == 0],
                                                     th.theta)}
        sections[slot] = {"sentence": fx["sentence"], "systems": rows, "lighting": lighting,
                          "paired_vs_frame_difference": paired, "frame_difference_orientation": orient, **extra}
        for r in recs:
            out_rows.append({"tripwire": slot, **r})
    write_csv("clips.csv", out_rows)
    write_csv("decisions_loo.csv", loo_rows)
    save_section("clips", {"tripwires": sections})
    write_demo()
    print(json.dumps({k: [(r["system"], r.get("auc"), r.get("precision")) for r in v["systems"]]
                      for k, v in sections.items()}, indent=1))
    return 0


# ----------------------------------------------------------------------------------- fresh mode
def cmd_fresh(a) -> int:
    from jaga.embed import shared
    from jaga.offline import embed_long, fresh_mode

    normal_dir = config.jaga_home() / "normal"
    tags = {"01": "iron", "02": "fridge", "03": "door", "04": "cat"}
    rows, done = [], {}
    measured = (load_summary().get("period") or {}).get("period_p50_s")
    stride = max(1, round(a.stride if a.stride else (measured or 1.0)))  # the live window period, as shipped
    for slot, tag in tags.items():
        cands = sorted(normal_dir.glob(f"{tag}.*")) if normal_dir.exists() else []
        if not cands:
            continue
        fx, pos, neg, _ = caption_vectors(slot)
        emb = shared(a.device)
        n_ts, n_v = embed_long(cands[0], emb, stride)
        P = [c for c in read_clips() if c["tripwire"].zfill(2) == slot and c["label"] == "1" and c["heldout"] != "1"
             and c["set"] != "iron-inview"]
        pv = [committed_vector(c["clip_id"]) for c in P]
        pv = [v for v in pv if v is not None]
        # positives are single 8-s clips: per-window recall at θ_z (no dwell on one window)
        # the held-out half is 60 s: score it at the 30-s dwell (a longer dwell could never trip inside it)
        r = fresh_mode(n_ts, n_v, [], np.zeros((0, 256)), pos, neg, dwell_s=30.0, stride_s=float(stride))
        if not r.get("ok"):
            continue
        n_m = margins(n_v, pos, neg) if len(n_v) else np.zeros(0)
        hold_z = (n_m[n_ts - n_ts[0] >= 60.0] - r["mu0"]) / r["sigma0"]
        zp = [(float(margins(v[None], pos, neg)[0]) - r["mu0"]) / r["sigma0"] for v in pv]
        hold = r["n_holdout"]
        row = {"tripwire": slot, "normal_capture": cands[0].name, "stride_s": stride, "n_fit": r["n_fit"],
               "n_holdout": hold, "theta_z": r["theta_z"], "holdout_dwell_s": 30.0, "holdout_trips": r["holdout_trips"],
               "holdout_windows_above": int((hold_z >= r["theta_z"]).sum()), "z_hold_max": r["z_hold_max"],
               "pos_clips": len(zp), "pos_clips_above": sum(z >= r["theta_z"] for z in zp), "indicative": hold < 12}
        rows.append(row)
        done[slot] = row
    write_csv("fresh.csv", rows)
    save_section("fresh", {"tripwires": done} if done else {"status": "not yet run",
                                                             "why": "no normal-state captures (normal/<tripwire>)"})
    write_demo()
    print(json.dumps(done, indent=1, default=float))
    return 0


# ----------------------------------------------------------------------------------- soak
def label_times(db, session: str) -> dict:
    rows = db.query("SELECT ts, kind, note FROM label WHERE session=? ORDER BY ts", (session,))
    return rows


def schedule_intervals(labels: list[dict]) -> list[dict]:
    """Planned steps (plan:<kind> … plan_end:<kind>) with the nearest key press within ±60 s replacing a time."""
    starts = [r for r in labels if r["kind"].startswith("plan:") and r["kind"] not in ("plan:session_start",
                                                                                      "plan:session_end")]
    ends = [r for r in labels if r["kind"].startswith("plan_end:")]
    keys = {k: [r["ts"] for r in labels if r["kind"] == k] for k in
            ("still_start", "still_end", "exit_leave", "exit_inview", "returned")}

    def near(kind, t):
        c = [x for x in keys.get(kind, []) if abs(x - t) <= 60]
        return min(c, key=lambda x: abs(x - t)) if c else t

    out = []
    for s in starts:
        kind = s["kind"].split(":", 1)[1]
        e = next((x for x in ends if x["kind"] == f"plan_end:{kind}" and x["ts"] >= s["ts"]), None)
        t0, t1 = s["ts"], (e["ts"] if e else None)
        if kind == "still":
            t0, t1 = near("still_start", t0), (near("still_end", t1) if t1 else None)
        elif kind in ("exit_leave", "exit_inview"):
            t0, t1 = near(kind, t0), (near("returned", t1) if t1 else None)
        out.append({"kind": kind, "start": t0, "end": t1})
    return out


def session_end(db, session: str) -> float | None:
    return db.query("SELECT MAX(ts) AS t FROM label WHERE session=? AND kind='plan:session_end'", (session,))[0]["t"]


def countdown_cuts(t_on: float, t_end: float, minutes: float, resume_at: list[float]) -> list[float]:
    """A plug countdown timer: it cuts `minutes` after each switch-on; after its own cut the person switches it back
    on at the next scheduled step (the same counterfactual as the other baselines)."""
    cuts = []
    while t_on + minutes * 60 <= t_end:
        c = t_on + minutes * 60
        cuts.append(c)
        later = [r for r in resume_at if r > c]
        if not later:
            break
        t_on = later[0]
    return cuts


def soak_rows(db, session: str, slot: str, dwell_s: float, k_fraction: float, p8: float, r_thr: float | None,
              live_dwell: bool = True):
    emb = db.query("SELECT * FROM embed WHERE session=? ORDER BY ts", (session,))
    if not emb:
        return None
    tw = db.tripwire(slot)
    if tw is None:
        print(f"{session}: tripwire {slot} is not in this database")
        return None
    p50, _ = window_period([e["ts"] for e in emb])
    stride = p50 or 1.0
    end = session_end(db, session)
    if end is not None:  # the schedule's end closes the session: anything later is not part of it
        emb = [e for e in emb if e["ts"] <= end + stride]
    wins = {w["embed_id"]: w for w in db.query("SELECT * FROM win WHERE tripwire_id=? ORDER BY ts", (tw["id"],))}
    ts = [e["ts"] for e in emb]
    labels = label_times(db, session)
    steps = schedule_intervals(labels)
    step_starts = sorted(s["start"] for s in steps)
    suspends = [e["ts"] for e in db.query("SELECT ts FROM event WHERE session=? AND kind='suspend'", (session,))]
    th = db.query("SELECT theta FROM threshold WHERE tripwire_id=? AND theta IS NOT NULL "
                  "AND source NOT LIKE 'nebius-job%' AND mode='labelled-m' ORDER BY version DESC LIMIT 1", (tw["id"],))
    theta = th[0]["theta"] if th else None
    flags = {
        "jaga (recomputed)": [bool(wins.get(e["id"], {}).get("margin") is not None and theta is not None
                                   and wins[e["id"]]["margin"] >= theta) for e in emb],
        "stillness timer": [e["motion_pct"] is not None and e["motion_pct"] < p8 for e in emb],
        "HOG person-in-zone": [e["person_in_zone"] == 0 for e in emb],
    }
    if r_thr is not None:
        flags["reference difference ∧ no person"] = [e["ref_diff_pct"] is not None and e["ref_diff_pct"] >= r_thr
                                                     and e["person_in_zone"] == 0 for e in emb]
    trips = {name: replay_rule(ts, f, dwell_s, 600.0, stride, suspend_at=suspends, k_fraction=k_fraction,
                               pause_after_trip=True, resume_at=step_starts) for name, f in flags.items()}
    if live_dwell:  # the live trips happened at the session's own dwell; a secondary dwell is recomputed only
        live = [e["ts"] for e in db.query("SELECT ts FROM event WHERE session=? AND kind='trip' AND tripwire_id=?",
                                          (session, tw["id"])) if e["ts"] <= ts[-1]]
        trips = {"jaga (live)": live, **trips}
    still = [s for s in steps if s["kind"] == "still" and s["end"]]
    exits = [s for s in steps if s["kind"] in ("exit_leave", "exit_inview") and s["end"]]
    hours = (ts[-1] - ts[0]) / 3600 if len(ts) > 1 else 0
    res = {}
    for name, tt in trips.items():
        cut = [any(s["start"] <= t <= s["end"] + stride for t in tt) for s in still]
        caught, ttt, kinds = [], [], []
        for x in exits:
            hit = [t for t in tt if x["start"] < t <= x["end"] + stride]
            caught.append(bool(hit))
            kinds.append(x["kind"])
            if hit:
                ttt.append(hit[0] - x["start"])
        other = [t for t in tt if not any(s["start"] <= t <= s["end"] + stride for s in still)
                 and not any(x["start"] < t <= x["end"] + stride for x in exits)]
        res[name] = {"still_cut": sum(cut), "still_n": len(still), "still_cut_flags": cut,
                     "exits_caught": sum(caught), "exits_n": len(exits), "caught_flags": caught, "exit_kinds": kinds,
                     "median_time_to_trip_s": S.median(ttt), "trips": len(tt), "other_false_cuts": len(other)}
    timers = {}
    for T in (15, 30, 60):
        cuts = countdown_cuts(ts[0], ts[-1], T, step_starts)
        mid = sum(1 for c in cuts if not any(x["start"] < c <= x["end"] for x in exits))
        # how long the power stayed on while nobody was there: exit start → the countdown's cut or the return
        on_after = [(min([c for c in cuts if x["start"] < c <= x["end"]] or [x["end"]]) - x["start"]) / 60
                    for x in exits]
        timers[f"{T} min"] = {"cuts": len(cuts), "mid_session_cuts": mid, "minutes_on_after_exit": on_after,
                              "minutes_on_after_exit_median": S.median(on_after)}
    return {"session": session, "windows": len(emb), "hours": round(hours, 3), "stride_s": stride, "theta": theta,
            "dwell_s": dwell_s, "systems": res, "plug_countdown": timers, "still_episodes": len(still),
            "exits": len(exits)}


def cmd_soak(a) -> int:
    from jaga.db import DB

    db = DB(a.db) if a.db else DB()
    results = {}
    for sess in a.session:
        r = soak_rows(db, sess, a.tripwire, a.dwell, a.k_fraction, a.p8, a.r, live_dwell=not a.secondary)
        if r is None:
            print(f"{sess}: no windows logged")
            continue
        results[sess] = r
    key = f"soak_b_dwell_{a.dwell:.0f}"  # one section per dwell: the 30-s row never overwrites the 120-s one
    if not results:
        print(f"no windows for session(s) {', '.join(a.session)} — nothing written", file=sys.stderr)
        if not load_summary().get(key):
            save_section(key, {"status": "not yet run", "why": "no scored Soak B session in the database"})
            write_demo()
        return 1
    pooled = {}
    hours = sum(r["hours"] for r in results.values())
    for r in results.values():
        for name, v in r["systems"].items():
            p = pooled.setdefault(name, {"still_cut_flags": [], "caught_flags": [], "kinds": [], "ttt": [], "other": 0})
            p["still_cut_flags"] += v["still_cut_flags"]
            p["caught_flags"] += v["caught_flags"]
            p["kinds"] += v["exit_kinds"]
            p["other"] += v["other_false_cuts"]
            if v["median_time_to_trip_s"] is not None:
                p["ttt"].append(v["median_time_to_trip_s"])
    table = []
    for name, p in pooled.items():
        n, k = len(p["still_cut_flags"]), sum(p["still_cut_flags"])
        e, c = len(p["caught_flags"]), sum(p["caught_flags"])
        lo, hi = S.clopper_pearson(k, n) if n else (None, None)
        elo, ehi = S.clopper_pearson(c, e) if e else (None, None)
        by_kind = {}
        for kind in ("exit_leave", "exit_inview"):
            flags = [f for f, kk in zip(p["caught_flags"], p["kinds"], strict=True) if kk == kind]
            by_kind[kind] = {"caught": sum(flags), "n": len(flags),
                             "ci95": list(S.clopper_pearson(sum(flags), len(flags))) if flags else None}
        rate = list(S.poisson_rate(p["other"], hours)) if hours > 0 else None
        table.append({"system": name, "still_episodes": n, "still_cut": k, "cut_ci95": [lo, hi], "exits": e,
                      "exits_caught": c, "recall_ci95": [elo, ehi], "recall_by_exit_type": by_kind,
                      "median_time_to_trip_s": S.median(p["ttt"]), "other_false_cuts": p["other"],
                      "other_false_cuts_per_hour_ci95": rate, "attended_hours": round(hours, 3)})
    j = pooled.get("jaga (live)") or pooled.get("jaga (recomputed)")
    t = pooled.get("stillness timer")
    paired = None
    if j and t and len(j["still_cut_flags"]) == len(t["still_cut_flags"]):
        b = sum(1 for x, y in zip(j["still_cut_flags"], t["still_cut_flags"], strict=True) if x and not y)
        c = sum(1 for x, y in zip(j["still_cut_flags"], t["still_cut_flags"], strict=True) if y and not x)
        paired = {"jaga_only": b, "timer_only": c, "sign_test_p": S.sign_test(b, c)}
    target = None
    jr = next((r for r in table if r["system"] == "jaga (live)"), None) or \
        next((r for r in table if r["system"] == "jaga (recomputed)"), None)
    tr = next((r for r in table if r["system"] == "stillness timer"), None)
    if jr and tr and jr["exits"] and not a.secondary:  # the targets were pre-registered for the primary dwell only
        target = {"fewer_than_half": jr["still_cut"] < tr["still_cut"] / 2,
                  "recall_ge_0.8": jr["exits_caught"] / jr["exits"] >= 0.8}
        target["both_hold"] = all(target.values())
    countdown = {}
    for T in ("15 min", "30 min", "60 min"):
        per = [r["plug_countdown"][T] for r in results.values()]
        mins = [m for x in per for m in x["minutes_on_after_exit"]]
        countdown[T] = {"cuts": sum(x["cuts"] for x in per), "exits": len(mins),
                        "mid_session_cuts": sum(x["mid_session_cuts"] for x in per),
                        "minutes_on_after_exit_median": S.median(mins)}
    write_csv(f"{key}.csv", table)
    save_section(key, {"sessions": results, "pooled": table, "paired_vs_stillness": paired, "targets": target,
                       "plug_countdown": countdown, "dwell_s": a.dwell, "primary": not a.secondary,
                       "inputs": {"P7_k_fraction": a.k_fraction, "P8": a.p8, "R": a.r,
                                  "note": "frozen values passed on the command line (docs/PROTOCOL.md)"}})
    write_demo()
    print(json.dumps({"pooled": table, "paired": paired, "targets": target}, indent=1, default=float))
    return 0


# ----------------------------------------------------------------------------------- Soak A (supporting)
ARMED_STATES = ("armed", "holding")


MAX_OPEN_S = 900.0  # docs/PROTOCOL.md: a left-open key with no closing key counts as closed 15 min later


def intervals(labels: list[dict], start_kind: str, end_kind: str) -> tuple[list[tuple[float, float]], list[dict]]:
    """Pair each start key with the next end key (keys 6 = left open, 7 = closed). A start with no end closes at the
    next start or MAX_OPEN_S later, whichever is first; an end with no start is ignored. Both are reported."""
    out, unpaired, open_t = [], [], None
    for r in labels:
        if r["kind"] == start_kind:
            if open_t is not None:
                out.append((open_t, min(r["ts"], open_t + MAX_OPEN_S)))
                unpaired.append({"ts": open_t, "missing": end_kind})
            open_t = r["ts"]
        elif r["kind"] == end_kind:
            if open_t is None:
                unpaired.append({"ts": r["ts"], "missing": start_kind})
            else:
                out.append((open_t, r["ts"]))
                open_t = None
    if open_t is not None:
        out.append((open_t, open_t + MAX_OPEN_S))
        unpaired.append({"ts": open_t, "missing": end_kind})
    return out, unpaired


def armed_seconds(ts: list[float], armed: list[bool], cap_s: float) -> float:
    """Time the tripwire could trip: each armed window counts until the next window, capped (a gap is not armed)."""
    return float(sum(min(ts[i + 1] - ts[i], cap_s) for i in range(len(ts) - 1) if armed[i]))


def soak_a_rows(db, session: str, slot: str, p8: float, dwell_s: float | None) -> dict | None:
    from jaga.tripwires import load

    tw = db.tripwire(slot)
    if tw is None:
        return None
    emb = db.query("SELECT * FROM embed WHERE session=? ORDER BY ts", (session,))
    wins = {w["embed_id"]: w for w in db.query("SELECT * FROM win WHERE tripwire_id=? ORDER BY ts", (tw["id"],))}
    emb = [e for e in emb if e["id"] in wins]
    if len(emb) < 2:
        return None
    fx = load(slot)
    op = fx["operating"]
    dwell = float(dwell_s or op["dwell_s"])
    cooldown = float(op.get("cooldown_s", 600))
    cut = op["action"] == "plug_off"
    ts = [e["ts"] for e in emb]
    p50, p95 = window_period(ts)
    stride = p50 or 1.0
    armed = [wins[e["id"]]["state"] in ARMED_STATES for e in emb]
    hours = armed_seconds(ts, armed, 2 * (p95 or stride)) / 3600
    labels = label_times(db, session)
    positives, unpaired = intervals(labels, "left_open", "closed")
    suspends = [e["ts"] for e in db.query("SELECT ts FROM event WHERE session=? AND kind='suspend'", (session,))]
    resumes = [e["ts"] for e in db.query("SELECT ts FROM event WHERE session=? AND kind='rearmed' AND tripwire_id=?",
                                         (session, tw["id"]))]
    th = db.query("SELECT theta FROM threshold WHERE tripwire_id=? AND theta IS NOT NULL "
                  "AND source NOT LIKE 'nebius-job%' AND mode='labelled-m' ORDER BY version DESC LIMIT 1", (tw["id"],))
    theta = th[0]["theta"] if th else None
    a_ts = [t for t, ok in zip(ts, armed, strict=True) if ok]
    a_emb = [e for e, ok in zip(emb, armed, strict=True) if ok]

    def run(flags):
        return replay_rule(a_ts, flags, dwell, cooldown, stride, suspend_at=suspends,
                           k_fraction=config.TUNING.k_fraction, pause_after_trip=cut, resume_at=resumes)

    systems = {"jaga (live)": [e["ts"] for e in db.query(
        "SELECT ts FROM event WHERE session=? AND kind='trip' AND tripwire_id=?", (session, tw["id"]))]}
    if theta is not None:
        systems["jaga (recomputed)"] = run([wins[e["id"]]["margin"] is not None and wins[e["id"]]["margin"] >= theta
                                            for e in a_emb])
    mp = [e["motion_pct"] for e in a_emb]
    systems["frame difference (motion)"] = run([m is not None and m >= p8 for m in mp])
    systems["frame difference (stillness)"] = run([m is not None and m < p8 for m in mp])
    fresh = None
    try:  # the stored vectors re-scored in fresh mode at the frozen P6: μ₀/σ₀ from the capture's first 60 s
        from jaga import vectors as V
        from jaga.score import normal_reference
        from jaga.tripwires import load_vectors

        ids, vecs = V.load(session)
        vmap = {int(i): v for i, v in zip(ids, vecs, strict=True)}
        pos, neg = load_vectors(fx)
        cap = [e for e in emb if wins[e["id"]]["state"] == "capturing" and e["id"] in vmap]
        fit = [e for e in cap if e["ts"] - cap[0]["ts"] < 60.0] if cap else []
        if len(fit) >= 2:
            mu0, sigma0 = normal_reference(margins(np.stack([vmap[e["id"]] for e in fit]), pos, neg),
                                           config.TUNING.sigma0_floor)
            have = [e for e in a_emb if e["id"] in vmap]
            m = margins(np.stack([vmap[e["id"]] for e in have]), pos, neg) if have else np.zeros(0)
            zf = {e["id"]: (float(x) - mu0) / sigma0 >= config.TUNING.theta_z for e, x in zip(have, m, strict=True)}
            systems[f"jaga fresh mode (θ_z = {config.TUNING.theta_z:g}, offline)"] = run(
                [zf.get(e["id"], False) for e in a_emb])
            fresh = {"mu0": mu0, "sigma0": sigma0, "n_fit": len(fit)}
    except FileNotFoundError:
        fresh = {"status": "no vectors file for this session"}
    res = {}
    for name, tt in systems.items():
        caught = [any(t0 < t <= t1 + stride for t in tt) for t0, t1 in positives]
        false = [t for t in tt if not any(t0 < t <= t1 + stride for t0, t1 in positives)]
        res[name] = {"trips": len(tt), "false_alarms": len(false), "positives_caught": sum(caught),
                     "positives": len(positives), "caught_flags": caught}
    return {"session": session, "tripwire": slot, "sentence": fx["sentence"], "windows": len(emb),
            "armed_hours": round(hours, 3), "stride_s": stride, "dwell_s": dwell, "theta": theta, "p8": p8,
            "systems": res, "fresh_reference": fresh, "unpaired_keys": unpaired}


def cmd_soak_a(a) -> int:
    from jaga.db import DB

    db = DB(a.db) if a.db else DB()
    per = {}
    for sess in a.session:
        for slot in a.tripwire.split(","):
            r = soak_a_rows(db, sess, slot.strip().zfill(2), a.p8, a.dwell)
            if r is not None:
                per.setdefault(r["tripwire"], []).append(r)
    if not per:
        print(f"no windows for session(s) {', '.join(a.session)} — nothing written", file=sys.stderr)
        if not load_summary().get("soak_a"):
            save_section("soak_a", {"status": "not yet run", "why": "no Soak A session with logged windows"})
            write_demo()
        return 1
    out, rows = {}, []
    for slot, rs in per.items():
        hours = sum(r["armed_hours"] for r in rs)
        table = []
        for name in rs[0]["systems"]:
            fa = sum(r["systems"][name]["false_alarms"] for r in rs if name in r["systems"])
            flags = [f for r in rs if name in r["systems"] for f in r["systems"][name]["caught_flags"]]
            row = {"tripwire": slot, "system": name, "armed_hours": round(hours, 3), "false_alarms": fa,
                   "per_armed_hour_ci95": list(S.poisson_rate(fa, hours)) if hours > 0 else None,
                   "positives_caught": sum(flags), "positives": len(flags),
                   "caught_ci95": list(S.clopper_pearson(sum(flags), len(flags))) if flags else None}
            table.append(row)
            rows.append(row)
        out[slot] = {"sentence": rs[0]["sentence"], "sessions": [r["session"] for r in rs], "table": table,
                     "dwell_s": rs[0]["dwell_s"], "p8": a.p8, "fresh_reference": [r["fresh_reference"] for r in rs],
                     "unpaired_keys": [u for r in rs for u in r["unpaired_keys"]]}
    write_csv("soak_a.csv", rows)
    save_section("soak_a", {"tripwires": out})
    write_demo()
    print(json.dumps(out, indent=1, default=float))
    return 0


# ----------------------------------------------------------------------------------- HOG detection rate
def cmd_hog(a) -> int:
    """PROTOCOL §6: the share of the person-present iron clips (the hard negatives) on which person_in_zone = 1,
    on the newest frame of the clip's window, exactly as the live guard reads it."""
    from jaga.config import TUNING
    from jaga.frames import read_window
    from jaga.guards import Guards

    g = Guards(TUNING, zone=a.zone, hog=True)
    clips = [c for c in read_clips() if c["set"] == "iron" and c["label"] == "0" and c["heldout"] != "1"]
    rows = []
    for c in clips:
        path = LOCAL_CLIPS / f"{c['clip_id']}.mp4"
        if not path.exists():
            rows.append({"clip_id": c["clip_id"], "person_in_zone": None, "status": "clip file not on this laptop"})
            continue
        frames, _ = read_window(path, "start")  # the stored file IS the 8-s window
        rows.append({"clip_id": c["clip_id"], "person_in_zone": g.person_in_zone(frames[-1]), "status": "scored"})
    scored = [r for r in rows if r["status"] == "scored"]
    k = sum(r["person_in_zone"] for r in scored)
    n = len(scored)
    write_csv("hog.csv", rows)
    data = ({"k": k, "n": n, "rate": k / n, "ci95": list(S.clopper_pearson(k, n)), "zone": list(a.zone)} if n else
            {"status": "not yet run", "why": "no person-present iron clips on this laptop"})
    save_section("hog", data)
    write_demo()
    print(json.dumps(data, indent=1, default=float))
    return 0


# ----------------------------------------------------------------------------------- ablate
def cmd_ablate(a) -> int:
    from jaga.tripwires import load, load_vectors

    clips = [c for c in read_clips() if c["heldout"] != "1" and c["set"] != "iron-inview"]
    if not clips:
        save_section("ablate", {"status": "not yet run", "why": "no labelled clips"})
        write_demo()
        return 0
    arms, rows = {}, []
    vecs = {c["clip_id"]: committed_vector(c["clip_id"]) for c in clips}
    slots = sorted({c["tripwire"].zfill(2) for c in clips})
    pooled_pos = []
    emb = None
    for slot in slots:
        fx, pos, neg, _ = caption_vectors(slot)
        sub = [c for c in clips if c["tripwire"].zfill(2) == slot and vecs[c["clip_id"]] is not None]
        if not sub:
            continue
        V = np.stack([vecs[c["clip_id"]] for c in sub])
        y = np.array([int(c["label"]) for c in sub])
        variants = {"full jaga": margins(V, pos, neg), "(a) no hard negatives": margins(V, pos, np.zeros((0, 256)))}
        if a.text:
            from jaga.embed import shared

            emb = emb or shared(a.device)
            raw = emb.embed_text([fx["sentence"]])
            variants["(c) raw sentence + compiled negatives"] = margins(V, raw, neg)
            if slot == "01":
                idfx = load(REPO / "fixtures" / "tripwires" / "ablation" / "01-id.json")
                ipos, ineg = load_vectors(idfx)
                variants["(c-id) Super(Indonesian sentence)"] = margins(V, ipos, ineg)
                variants["(c-id) raw Indonesian sentence"] = margins(V, emb.embed_text([idfx["sentence"]]), ineg)
        for k in range(1, 6):
            vp = REPO / "fixtures" / "tripwires" / "variance" / f"{slot}-k{k}.vec.npz"
            if vp.exists():
                with np.load(vp) as z:
                    variants[f"(e) compile draw {k}"] = margins(V, z["pos"], z["neg"])
        pooled_pos += list(variants["full jaga"][y == 1])
        arms[slot] = {}
        right = {}  # per clip: is the arm's leave-one-out decision correct?
        for name, m in variants.items():
            r = system_row(name, list(m[y == 1]), list(m[y == 0]))
            dp, dn = loo_decisions(list(m[y == 1]), list(m[y == 0]))
            right[name] = [bool(x) for _, x in dp] + [not x for _, x in dn]
            if name != "full jaga":
                b = sum(1 for j, x in zip(right["full jaga"], right[name], strict=True) if j and not x)
                c = sum(1 for j, x in zip(right["full jaga"], right[name], strict=True) if x and not j)
                r["paired_vs_full"] = {"full_right_arm_wrong": b, "arm_right_full_wrong": c,
                                       "sign_test_p": S.sign_test(b, c)}
            arms[slot][name] = r
            rows.append({"tripwire": slot, **{k: v for k, v in r.items() if k not in ("auc_ci95", "paired_vs_full")}})
    if pooled_pos:  # (b) one global θ across tripwires: precision/false positives only (AUC is threshold-free)
        g = theta_at_recall(pooled_pos, 0.9)
        for slot in arms:
            fx, pos, neg, _ = caption_vectors(slot)
            sub = [c for c in clips if c["tripwire"].zfill(2) == slot and vecs[c["clip_id"]] is not None]
            V = np.stack([vecs[c["clip_id"]] for c in sub])
            y = np.array([int(c["label"]) for c in sub])
            m = margins(V, pos, neg)
            conf = S.confusion(list(m[y == 1]), list(m[y == 0]), g)
            full = [bool(x) for _, x in sum(loo_decisions(list(m[y == 1]), list(m[y == 0])), [])]
            truth = [True] * int((y == 1).sum()) + [False] * int((y == 0).sum())
            glob = [bool(v >= g) for v in list(m[y == 1]) + list(m[y == 0])]
            b = sum(1 for f, x, t in zip(full, glob, truth, strict=True) if f == t and x != t)
            c = sum(1 for f, x, t in zip(full, glob, truth, strict=True) if x == t and f != t)
            arms[slot]["(b) one global θ"] = {"theta": g, **conf, "paired_vs_full": {
                "full_right_arm_wrong": b, "arm_right_full_wrong": c, "sign_test_p": S.sign_test(b, c)}}
    write_csv("ablate.csv", rows)
    save_section("ablate", {"tripwires": arms, "single_frame_d": "run `bench.py ablate-frame` (re-embeds every clip)"})
    write_demo()
    print(json.dumps({s: {k: (v.get("auc"), v.get("precision")) for k, v in d.items()} for s, d in arms.items()},
                     indent=1, default=float))
    return 0


def cmd_ablate_frame(a) -> int:
    """(d) single frame: the same model fed 8 copies of the clip's last frame (temporal context only)."""
    from jaga.embed import shared
    from jaga.frames import read_window

    clips = [c for c in read_clips() if c["heldout"] != "1" and c["set"] != "iron-inview"]
    emb = shared(a.device)
    res = {}
    for slot in sorted({c["tripwire"].zfill(2) for c in clips}):
        fx, pos, neg, _ = caption_vectors(slot)
        sp, sn = [], []
        for c in clips:
            p = LOCAL_CLIPS / f"{c['clip_id']}.mp4"
            if c["tripwire"].zfill(2) != slot or not p.exists():
                continue
            frames, _ = read_window(p, "start")
            v = emb.embed_video(np.stack([frames[-1]] * 8))
            (sp if c["label"] == "1" else sn).append(float(margins(v[None], pos, neg)[0]))
        res[slot] = system_row("(d) single frame ×8", sp, sn)
    s = load_summary().get("ablate", {})
    s["single_frame"] = res
    save_section("ablate", {k: v for k, v in s.items() if not k.startswith("_")})
    write_demo()
    print(json.dumps(res, indent=1, default=float))
    return 0


# ----------------------------------------------------------------------------------- outside people
def cmd_outside(a) -> int:
    from jaga.compile import compile_sentence
    from jaga.embed import shared
    from jaga.offline import embed_long, fresh_mode

    root = Path(a.root).expanduser()
    people = sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    rows = []
    emb = None
    for i, p in enumerate(people, 1):
        consent = list(p.glob("consent.*"))
        sentence_f = p / "sentence.txt"
        normal = next(iter(sorted(p.glob("normal.*"))), None)
        positive = next(iter(sorted(p.glob("positive.*"))), None)
        pid = f"P{i}"
        if not (consent and sentence_f.exists() and normal and positive):
            rows.append({"person": pid, "status": "incomplete (needs consent, sentence.txt, normal.*, positive.*)"})
            continue
        emb = emb or shared(a.device)
        r = compile_sentence(sentence_f.read_text().strip()).to_dict()
        spec = r["spec"]
        if not spec["visual"]:
            rows.append({"person": pid, "status": "not visual", "tf_request_id": r["tf_request_id"]})
            continue
        pos, neg = emb.embed_text(spec["positives"]), emb.embed_text(spec["hard_negatives"])
        stride = max(1, round(a.stride))
        n_ts, n_v = embed_long(normal, emb, stride)
        p_ts, p_v = embed_long(positive, emb, stride)
        fm = fresh_mode(n_ts, n_v, p_ts, p_v, pos, neg, dwell_s=30.0, stride_s=float(stride))
        rows.append({"person": pid, "status": "scored" if fm.get("ok") else fm.get("detail"),
                     "tf_request_id": r["tf_request_id"], "tripped_correctly": fm.get("tripped_correctly"),
                     "time_to_trip_s": fm.get("time_to_trip_s"), "holdout_trips": fm.get("holdout_trips"),
                     "n_fit": fm.get("n_fit"), "n_holdout": fm.get("n_holdout")})
    scored = [r for r in rows if r.get("status") == "scored"]
    k = sum(1 for r in scored if r["tripped_correctly"])
    n = len(scored)
    ci = S.clopper_pearson(k, n) if n else (None, None)
    write_csv("outside.csv", rows)
    save_section("outside", {"k": k, "n": n, "ci95": ci, "received": len(people)} if people else
                 {"status": "not yet run", "why": "no outside people's clips received"})
    write_demo()
    print(f"{k} of {n} tripped correctly first time (95% CI {ci})")
    return 0


# ----------------------------------------------------------------------------------- tune (unscored only)
P7_CANDIDATES = (0.8, 0.7, 0.6)


def p7_candidate(db, session: str, slot: str, dwell_s: float, emb: list[dict], exits: list[float],
                 rets: list[float]) -> tuple[float | None, list[dict]]:
    """PROTOCOL §6: 0.8 unless a logged exit is missed at 0.8; then the largest of {0.7, 0.6} that catches every
    logged exit; if neither does, 0.6 and the miss is disclosed."""
    tw = db.tripwire(slot)
    if tw is None or not emb or not exits:
        return None, []
    wins = {w["embed_id"]: w for w in db.query("SELECT * FROM win WHERE tripwire_id=?", (tw["id"],))}
    th = db.query("SELECT theta FROM threshold WHERE tripwire_id=? AND theta IS NOT NULL "
                  "AND source NOT LIKE 'nebius-job%' AND mode='labelled-m' ORDER BY version DESC LIMIT 1", (tw["id"],))
    rows = [e for e in emb if e["id"] in wins]
    ts = [e["ts"] for e in rows]
    if th:
        flags = [wins[e["id"]]["margin"] is not None and wins[e["id"]]["margin"] >= th[0]["theta"] for e in rows]
    else:  # no θ v1 yet: the live per-window decision
        flags = [bool(wins[e["id"]]["above"]) for e in rows]
    stride = window_period(ts)[0] or 1.0
    on = [e["ts"] for e in db.query("SELECT ts FROM event WHERE session=? AND kind='rearmed'", (session,))]
    out = []
    for k in P7_CANDIDATES:
        trips = replay_rule(ts, flags, dwell_s, 600.0, stride, k_fraction=k, pause_after_trip=True,
                            resume_at=on + list(exits))
        caught = sum(1 for t0 in exits if any(t0 < t <= min([r for r in rets if r > t0] or [math.inf]) + stride
                                              for t in trips))
        out.append({"k_fraction": k, "exits_caught": caught, "exits": len(exits)})
    full = [r["k_fraction"] for r in out if r["exits_caught"] == r["exits"]]
    if out[0]["exits_caught"] == out[0]["exits"]:
        return 0.8, out
    return (max(k for k in full if k < 0.8) if [k for k in full if k < 0.8] else 0.6), out


def cmd_tune(a) -> int:
    from jaga.db import DB

    db = DB(a.db) if a.db else DB()
    sess = db.query("SELECT * FROM session WHERE name=?", (a.session,))
    if not sess:
        raise SystemExit(f"no session {a.session}")
    if sess[0]["scored"]:
        raise SystemExit(f"{a.session} is a scored session: tuning on it is not allowed (hold-out)")
    emb = db.query("SELECT * FROM embed WHERE session=? ORDER BY ts", (a.session,))
    labels = label_times(db, a.session)
    empty = []
    exits = [r["ts"] for r in labels if r["kind"] in ("exit_leave", "exit_inview")]
    rets = [r["ts"] for r in labels if r["kind"] == "returned"]
    for t0 in exits:
        t1 = min([r for r in rets if r > t0] or [math.inf])
        empty += [e for e in emb if t0 + 15 <= e["ts"] < t1]  # 15 s after leaving: the window has refilled
    p8 = math.ceil(max(e["motion_pct"] for e in empty if e["motion_pct"] is not None) * 100) / 100 if empty else None
    cap_end = emb[0]["ts"] + config.TUNING.capture_s if emb else None
    capw = [e["ref_diff_pct"] for e in emb if cap_end and e["ts"] <= cap_end and e["ref_diff_pct"] is not None]
    r = float(np.percentile(capw, 99)) if capw else None
    p7, p7_rows = p7_candidate(db, a.session, a.tripwire, a.dwell, emb, exits, rets)
    data = {"session": a.session, "empty_windows": len(empty), "P7_candidate": p7, "P7_check": p7_rows,
            "P8_candidate": p8, "R_candidate": r, "exits_logged": len(exits),
            "note": "candidates only; the values used are committed in docs/PROTOCOL.md before B1"}
    save_section("tune", data)
    print(json.dumps(data, indent=2))
    return 0


# ----------------------------------------------------------------------------------- DEMO.md
def fmt(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{d}f}"
    return str(x)


def write_demo() -> None:
    s = load_summary()
    L = ["# DEMO — what jaga has measured, and how to reproduce it", "",
         "This file is written by `scripts/bench.py` from `bench/out/summary.json`. Do not edit it by hand.",
         "A section that has not run says so; nothing here is estimated.", ""]

    def stamp(sec):
        m = sec.get("_machine", {})
        return (f"*ran {sec.get('_ran_at', '?')} at commit `{sec.get('_commit', '?')}` on {m.get('processor', '?')}"
                f" ({m.get('platform', '?')}); 1-min load average when the result was written: "
                f"{m.get('loadavg_1m', '?')}*")

    def ci(pair, d=2):
        return "—" if not pair or pair[0] is None else f"{fmt(pair[0], d)}–{fmt(pair[1], d)}"

    L += ["## 1. Speed — Cosmos-Embed1-224p on this laptop", ""]
    sp = s.get("speed")
    if sp and sp.get("rows"):
        L += ["| device | precision | timed calls | p50 s/clip | p95 s/clip | torch | threads | "
              "1-min load average before → after |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sp["rows"]:
            load = (f"{r['loadavg_1m_before']} → {r['loadavg_1m_after']}" if r.get("loadavg_1m_before") is not None
                    else "not recorded during the run")
            L.append(f"| {r['device'].upper()} | fp32 | {r['n']} | {r['p50_s']} | {r['p95_s']} | {r['torch']} | "
                     f"{r['threads']} | {load} |")
        n = sp["rows"][0]["n"]
        cores = (sp.get("_machine") or {}).get("cpu_count") or 0
        busy = [r["device"].upper() for r in sp["rows"] if cores and max(r.get("loadavg_1m_before") or 0,
                                                                         r.get("loadavg_1m_after") or 0) > cores]
        L += ["", f"{n} timed calls on one window (8 frames → `preprocess` + `get_video_embeddings`), after 5 warm-up "
              f"calls; input `{sp['input']}` (the model card's own sample video). Other processes on the machine were "
              f"not stopped; the load column shows how busy it was. {stamp(sp)}", ""]
        if busy:
            ctx = [f"**The {' and '.join(busy)} row{'s' if len(busy) > 1 else ''} ran while the 1-min load average "
                   f"exceeded the machine's {cores} cores** — other processes were competing for it, so read "
                   f"{'them' if len(busy) > 1 else 'it'} as a loaded machine, not an idle one."]
            pe0 = s.get("period") or {}
            if pe0.get("embed_p50_s") and pe0.get("_commit") == sp.get("_commit"):
                ctx.append(f"At the same commit the live loop's own embed time was p50 {pe0['embed_p50_s']:.2f} s over "
                           f"{pe0['windows']} windows (below).")
            day0 = REPO / "evidence" / "day0" / "cosmos-mps.json"
            if day0.exists():
                d0 = json.loads(day0.read_text())
                q = d0.get("s_per_clip_preprocess_plus_embed") or {}
                if q.get("p50") is not None:
                    ctx.append(f"Day-0 stage COSMOS measured MPS p50 {q['p50']} s over {d0.get('runs')} calls "
                               f"(`evidence/day0/cosmos-mps.json`).")
            L += [" ".join(ctx), ""]
        L += [f"Reproduce: `python scripts/bench.py speed --n {n}`", ""]
    else:
        L += ["Not yet run. `python scripts/bench.py speed`", ""]
    pe = s.get("period")
    L += ["### Live window period N (every dwell clamp and the watchdog use it)", ""]
    if pe and pe.get("windows"):
        note = ("; " + pe["input"]) if pe.get("input") else ""
        L += [f"p50 **{fmt(pe['period_p50_s'], 2)} s**, p95 {fmt(pe['period_p95_s'], 2)} s over {pe['windows']} "
              f"windows of session `{pe['session']}` (source: {pe.get('source')}{note}). "
              f"{stamp(pe)}", ""]
    else:
        L += ["Not yet run. `python scripts/bench.py period --session <name>`", ""]

    L += ["## 2. Labelled clips (one 8-s clip = one window)", ""]
    cl = s.get("clips")
    if cl and cl.get("tripwires"):
        for slot, d in cl["tripwires"].items():
            L += [f"### Tripwire {slot} · “{d['sentence']}”", "",
                  "| system | clips (pos / neg) | AUC (95 % CI) | at recall 0.9 (leave-one-out θ): TP / FP "
                  "| precision |",
                  "|---|---|---|---|---|"]
            for r in d["systems"]:
                if "auc" not in r:
                    L.append(f"| {r['system']} | {r['n_pos']} / {r['n_neg']} | {r.get('status')} | | |")
                    continue
                c = (f"{r['auc_ci95'][0]:.3f}–{r['auc_ci95'][1]:.3f}" if r.get("auc_ci95")
                     else "interval undefined at AUC 0 or 1")
                L.append(f"| {r['system']} | {r['n_pos']} / {r['n_neg']} | {r['auc']:.3f} ({c}) | "
                         f"{r['tp']} / {r['fp']} | {fmt(r['precision'])} |")
            pv = d.get("paired_vs_frame_difference")
            if pv:
                L.append(f"\nPaired on the same clips (each at its own leave-one-out θ): jaga right and frame "
                         f"difference wrong on {pv['jaga_right_baseline_wrong']}, the reverse on "
                         f"{pv['baseline_right_jaga_wrong']}; exact sign test p = {fmt(pv['sign_test_p'], 3)}. "
                         f"Frame difference is scored as {d.get('frame_difference_orientation')}, fixed in "
                         f"`docs/PROTOCOL.md` before any clip.")
            for key in ("iron-inview", "iron-day2"):
                if key in d:
                    e = d[key]
                    L.append(f"\n{e['label']}: at frozen θ v{e['theta_version']} → TP {e['tp']}, FN {e['fn']}, "
                             f"FP {e['fp']}, TN {e['tn']}")
            L.append("")
        L += [f"n is small (about 10 + 10 per tripwire) and the intervals are wide. {stamp(cl)}", "",
              "Reproduce without the model: `python scripts/bench.py clips` (reads the committed vectors).", ""]
    else:
        L += ["Not yet run: no labelled clips have been recorded yet (`fixtures/clips.csv`).", ""]

    L += ["## 3. Fresh sentence, no labelled clips (the zero-training path)", ""]
    fr = s.get("fresh")
    if fr and fr.get("tripwires"):
        for slot, r in fr["tripwires"].items():
            L.append(f"- {slot}: μ₀/σ₀ from {r['n_fit']} normal windows (stride {r['stride_s']} s); held-out normal "
                     f"windows {r['n_holdout']}, of which above θ_z {r['theta_z']}: {r.get('holdout_windows_above')}; "
                     f"trips on them at a {fmt(r.get('holdout_dwell_s'), 0)}-s dwell: {r['holdout_trips']}; positive "
                     f"clips above θ_z: {r['pos_clips_above']} of {r['pos_clips']}"
                     + (" (indicative: < 12 windows)" if r['indicative'] else ""))
        L += ["", stamp(fr), ""]
    else:
        L += ["Not yet run: needs the 2-minute normal-state captures (`normal/<tripwire>`).", ""]

    L += ["## 4. Soak B — someone standing still at the board (pre-registered: docs/PROTOCOL.md)", ""]
    keys = sorted((k for k in s if k.startswith("soak_b_dwell_")), key=lambda k: (not s[k].get("primary", True), k))
    shown = False
    for key in keys:
        so = s[key]
        if not so.get("pooled"):
            continue
        shown = True
        L += [f"### Dwell {fmt(so['dwell_s'], 0)} s" + (" (primary)" if so.get("primary", True) else
                                                        " (secondary, recomputed from the same logs)"), "",
              "| system | still episodes cut (95 % CI) | exits caught (95 % CI) | leaves the room "
              "| steps away, in view | median time-to-trip | other false cuts per attended hour (95 % CI) |",
              "|---|---|---|---|---|---|---|"]
        for r in so["pooled"]:
            bk = r.get("recall_by_exit_type", {})
            lv, iv = bk.get("exit_leave", {}), bk.get("exit_inview", {})
            rate = r.get("other_false_cuts_per_hour_ci95")
            L.append(f"| {r['system']} | {r['still_cut']} of {r['still_episodes']} ({ci(r['cut_ci95'])}) | "
                     f"{r['exits_caught']} of {r['exits']} ({ci(r['recall_ci95'])}) | "
                     f"{lv.get('caught', '—')} of {lv.get('n', '—')} | {iv.get('caught', '—')} of {iv.get('n', '—')} | "
                     f"{fmt(r['median_time_to_trip_s'], 0)} s | "
                     + (f"{r['other_false_cuts']} in {fmt(r['attended_hours'], 2)} h → {fmt(rate[0], 2)} "
                        f"({fmt(rate[1], 2)}–{fmt(rate[2], 2)})" if rate else "—") + " |")
        cd = so.get("plug_countdown") or {}
        if cd:
            L += ["", "| plug countdown timer | cuts | mid-session cuts (person present) | minutes on after an exit "
                  "begins (median) |", "|---|---|---|---|"]
            for T, v in cd.items():
                L.append(f"| {T} | {v['cuts']} | {v['mid_session_cuts']} | "
                         f"{fmt(v['minutes_on_after_exit_median'], 1)} |")
        inp = so.get("inputs") or {}
        if inp:
            L.append(f"\nFrozen inputs: P7 (k fraction) = {fmt(inp.get('P7_k_fraction'), 2)}, "
                     f"P8 = {fmt(inp.get('P8'), 2)} % of pixels, R = {fmt(inp.get('R'), 2)} "
                     f"(docs/PROTOCOL.md, Frozen values).")
        if so.get("paired_vs_stillness"):
            p = so["paired_vs_stillness"]
            L.append(f"\nPaired on the same episodes: jaga-only cuts {p['jaga_only']}, timer-only cuts "
                     f"{p['timer_only']}, exact sign test p = {fmt(p['sign_test_p'], 3)}.")
        if so.get("targets"):
            t = so["targets"]
            L.append(f"\nPre-registered targets: fewer than half the timer's cuts: {t['fewer_than_half']}; "
                     f"recall on exits ≥ 0.8: {t['recall_ge_0.8']}.")
        L += ["", stamp(so), ""]
    if not shown:
        L += ["Not yet run: Soak B1/B2 have not been recorded. Protocol and schedules: `docs/PROTOCOL.md`.", ""]
    hg = s.get("hog")
    if hg and hg.get("n"):
        L += [f"HOG person-in-zone baseline: detected the person on {hg['k']} of {hg['n']} person-present iron clips "
              f"(95 % CI {ci(hg['ci95'])}). {stamp(hg)}", ""]

    L += ["## 4b. Soak A — one night, tripwires 02 and 03 (supporting)", ""]
    sa = s.get("soak_a")
    if sa and sa.get("tripwires"):
        for slot, d in sa["tripwires"].items():
            L += [f"### Tripwire {slot} · “{d['sentence']}” · dwell {fmt(d['dwell_s'], 0)} s", "",
                  "| system | armed hours | false alarms | per armed hour (95 % CI) | positives caught (95 % CI) |",
                  "|---|---|---|---|---|"]
            for r in d["table"]:
                rate = r.get("per_armed_hour_ci95")
                L.append(f"| {r['system']} | {fmt(r['armed_hours'], 2)} | {r['false_alarms']} | "
                         + (f"{fmt(rate[0], 2)} ({fmt(rate[1], 2)}–{fmt(rate[2], 2)})" if rate else "—")
                         + f" | {r['positives_caught']} of {r['positives']} ({ci(r.get('caught_ci95'))}) |")
            L.append("")
        p8 = next(iter(sa["tripwires"].values()))["p8"]
        unp = [u for d in sa["tripwires"].values() for u in d.get("unpaired_keys", [])]
        L += [f"Frame difference runs through the same rule at P8 = {fmt(p8, 2)} % of pixels changed, in both "
              f"orientations." + (f" {len(unp)} key press(es) had no partner (a left-open with no close counts as "
                                  f"closed after 15 min, per the protocol)." if unp else "") + f" {stamp(sa)}", ""]
    else:
        L += ["Not yet run: the Soak A night has not been recorded.", ""]

    L += ["## 5. Self-ablation", ""]
    ab = s.get("ablate")
    if ab and ab.get("tripwires"):
        for slot, arms in ab["tripwires"].items():
            L += [f"### Tripwire {slot}", "", "| arm | AUC | precision at recall 0.9 | paired vs full jaga: full right "
                  "/ arm right (sign test p) |", "|---|---|---|---|"]
            for name, r in arms.items():
                pv = r.get("paired_vs_full")
                pt = (f"{pv['full_right_arm_wrong']} / {pv['arm_right_full_wrong']} (p = {fmt(pv['sign_test_p'], 3)})"
                      if pv else "—")
                L.append(f"| {name} | {fmt(r.get('auc'))} | {fmt(r.get('precision'))} | {pt} |")
            L.append("")
        L += [stamp(ab), ""]
    else:
        L += ["Not yet run: needs the labelled clips.", ""]

    L += ["## 6. Outside people", ""]
    ou = s.get("outside")
    if ou and ou.get("n") is not None and "k" in ou:
        L += [f"{ou['k']} of {ou['n']} tripwires typed by people outside the team tripped correctly the first time "
              f"(95 % CI {fmt(ou['ci95'][0], 2)}–{fmt(ou['ci95'][1], 2)}); {ou['received']} received. Their videos "
              f"are not published; only these scores are. {stamp(ou)}", ""]
    else:
        L += ["Not yet run: no outside people's clips received yet.", ""]

    L += ["## Reproduce", "",
          "```bash",
          "pip install -e .",
          "python scripts/bench.py speed      # downloads Cosmos-Embed1 once (public on Hugging Face on 2026-10-15)",
          "python scripts/bench.py period --session <name>   # from any logged session",
          "```", "",
          "Once the labelled clips are recorded and their vectors committed, these run without the model:", "",
          "```bash",
          "python scripts/bench.py clips      # AUC / precision from the committed vectors",
          "jaga score --replay iron/pos_03 --tripwire fixtures/tripwires/01.json   # a clip id from fixtures/clips.csv",
          "```", ""]
    DEMO.write_text("\n".join(L))


def _zone_arg(v: str) -> tuple[float, float, float, float]:
    from jaga.cli import _zone

    return _zone(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--db", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("speed")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--devices", default="mps,cpu")
    p.add_argument("--clip", default=None)
    p.set_defaults(fn=cmd_speed)
    p = sub.add_parser("period")
    p.add_argument("--session", default=None)
    p.add_argument("--input-note", default=None)
    p.set_defaults(fn=cmd_period)
    p = sub.add_parser("clips")
    p.set_defaults(fn=cmd_clips)
    p = sub.add_parser("fresh")
    p.add_argument("--stride", type=float, default=None, help="default: the measured live window period (bench period)")
    p.set_defaults(fn=cmd_fresh)
    p = sub.add_parser("soak")
    p.add_argument("--session", action="append", required=True)
    p.add_argument("--tripwire", default="01")
    p.add_argument("--dwell", type=float, default=120.0)
    p.add_argument("--k-fraction", type=float, required=True, help="P7 as frozen in docs/PROTOCOL.md")
    p.add_argument("--p8", type=float, required=True, help="P8 as frozen in docs/PROTOCOL.md (stillness timer)")
    p.add_argument("--r", type=float, default=None,
                   help="R as frozen in docs/PROTOCOL.md (without it the reference-difference baseline is omitted)")
    p.add_argument("--secondary", action="store_true",
                   help="a secondary dwell (e.g. --dwell 30): recomputed for every system, live trips not used")
    p.set_defaults(fn=cmd_soak)
    p = sub.add_parser("ablate")
    p.add_argument("--text", action="store_true", help="also arms (c) and (c-id), which embed text with the model")
    p.set_defaults(fn=cmd_ablate)
    p = sub.add_parser("ablate-frame")
    p.set_defaults(fn=cmd_ablate_frame)
    p = sub.add_parser("outside")
    p.add_argument("--root", required=True)
    p.add_argument("--stride", type=float, default=1.0)
    p.set_defaults(fn=cmd_outside)
    p = sub.add_parser("tune")
    p.add_argument("--session", required=True)
    p.add_argument("--tripwire", default="01")
    p.add_argument("--dwell", type=float, default=120.0)
    p.set_defaults(fn=cmd_tune)
    p = sub.add_parser("soak-a")
    p.add_argument("--session", action="append", required=True)
    p.add_argument("--tripwire", default="02,03")
    p.add_argument("--dwell", type=float, default=None, help="default: the tripwire's operating dwell")
    p.add_argument("--p8", type=float, default=config.TUNING.still_motion_pct,
                   help="frame-difference threshold (%% of pixels changed): the P8 frozen for Soak B")
    p.set_defaults(fn=cmd_soak_a)
    p = sub.add_parser("hog")
    p.add_argument("--zone", required=True, type=_zone_arg, help="x0,y0,x1,y1 fractions (the frozen zone box)")
    p.set_defaults(fn=cmd_hog)
    p = sub.add_parser("demo")
    p.set_defaults(fn=lambda a: write_demo() or 0)
    a = ap.parse_args()
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
