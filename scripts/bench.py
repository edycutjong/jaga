#!/usr/bin/env python3
"""jaga's measurements. Every number on any page, slide, caption or voice-over comes from this script.

    bench.py speed            s/clip p50/p95 of Cosmos-Embed1 on this machine (MPS and CPU, fp32), N windows
    bench.py period           the live window period N = p50/p95 of diff(embed.ts) from a logged session
    bench.py clips            per tripwire on the labelled clips: AUC (Hanley–McNeil CI), leave-one-out θ at recall 0.9
                              → precision counts; vs frame difference and positives-only; in-view, day-2, lighting
    bench.py fresh            the zero-training row: μ₀/σ₀ from the first 60 s of normal/<slot>, default θ_z
    bench.py soak --session B1 [--session B2 …]
                              Soak B (pre-registered, docs/PROTOCOL.md) and Soak A rows
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
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
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
        times = []
        for _ in range(a.n):
            t0 = time.perf_counter()
            emb.embed_video(window)  # preprocess + get_video_embeddings + device sync: what one live window pays
            times.append(time.perf_counter() - t0)
        info = emb.info
        rows.append({"device": dev, "dtype": "float32", "n": a.n, "p50_s": round(statistics.median(times), 4),
                     "p95_s": round(pct(times, 0.95), 4), "mean_s": round(statistics.mean(times), 4),
                     "min_s": round(min(times), 4), "max_s": round(max(times), 4), "torch": info["torch"],
                     "threads": info["torch_threads"], "input": src.name, "load_s": info["load_s"]})
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


def loo_counts(scores_pos, scores_neg, recall=0.9) -> dict:
    """Leave-one-out θ at recall 0.9: each clip is judged by a θ fitted without it."""
    sp, sn = list(scores_pos), list(scores_neg)
    tp = fp = fn = tn = 0
    for i, s in enumerate(sp):
        others = sp[:i] + sp[i + 1:]
        th = theta_at_recall(others, recall) if others else s
        tp, fn = (tp + 1, fn) if s >= th else (tp, fn + 1)
    th_all = theta_at_recall(sp, recall) if sp else math.inf
    for s in sn:
        fp, tn = (fp + 1, tn) if s >= th_all else (fp, tn + 1)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if not sp else tp / len(sp)}


def system_row(name, sp, sn) -> dict:
    if not sp or not sn:
        return {"system": name, "n_pos": len(sp), "n_neg": len(sn), "status": "not enough clips"}
    a = S.auc(sp, sn)
    lo, hi = S.hanley_mcneil(a, len(sp), len(sn))
    return {"system": name, "n_pos": len(sp), "n_neg": len(sn), "auc": round(a, 4), "auc_ci95": [round(lo, 4),
            round(hi, 4)], **loo_counts(sp, sn)}


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
        absence = fx["sentence"].lower().startswith("nobody")
        P = [r for r in recs if r["label"] == 1]
        N = [r for r in recs if r["label"] == 0]
        rows = [system_row("jaga", [r["m"] for r in P], [r["m"] for r in N]),
                system_row("positives only (no hard negatives)", [r["pos_only"] for r in P],
                           [r["pos_only"] for r in N])]
        if all(r["motion_pct"] is not None for r in recs):
            sign = -1.0 if absence else 1.0  # an absent person is stillness: invert frame difference
            rows.append(system_row("frame difference" + (" (stillness)" if absence else ""),
                                   [sign * r["motion_pct"] for r in P], [sign * r["motion_pct"] for r in N]))
        # paired: jaga vs frame difference at their own leave-one-out operating points
        th_j = theta_at_recall([r["m"] for r in P], 0.9)
        for r in recs:
            r["jaga_above"] = int(r["m"] >= th_j)
            loo_rows.append({"tripwire": slot, **{k: r[k] for k in ("clip_id", "label", "m", "jaga_above")}})
        lighting = {}
        for r in recs:
            d = lighting.setdefault(r["lighting"], {"pos": 0, "neg": 0, "pos_above": 0, "neg_above": 0})
            k = "pos" if r["label"] else "neg"
            d[k] += 1
            d[f"{k}_above"] += r["jaga_above"]
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
        sections[slot] = {"sentence": fx["sentence"], "systems": rows, "lighting": lighting, **extra}
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
    tags = {"01": "iron", "02": "fridge", "03": "door"}
    rows, done = [], {}
    stride = max(1, round(a.stride))
    for slot, tag in tags.items():
        cands = sorted(normal_dir.glob(f"{tag}.*")) if normal_dir.exists() else []
        if not cands:
            continue
        fx, pos, neg, _ = caption_vectors(slot)
        emb = shared(a.device)
        n_ts, n_v = embed_long(cands[0], emb, stride)
        P = [c for c in read_clips() if c["tripwire"].zfill(2) == slot and c["label"] == "1" and c["heldout"] != "1"]
        pv = [committed_vector(c["clip_id"]) for c in P]
        pv = [v for v in pv if v is not None]
        # positives are single 8-s clips: per-window recall at θ_z (no dwell on one window)
        r = fresh_mode(n_ts, n_v, [], np.zeros((0, 256)), pos, neg, dwell_s=float(fx["operating"]["dwell_s"]),
                       stride_s=float(stride))
        if not r.get("ok"):
            continue
        zp = [(float(margins(v[None], pos, neg)[0]) - r["mu0"]) / r["sigma0"] for v in pv]
        hold = r["n_holdout"]
        row = {"tripwire": slot, "normal_capture": cands[0].name, "stride_s": stride, "n_fit": r["n_fit"],
               "n_holdout": hold, "theta_z": r["theta_z"], "holdout_trips": r["holdout_trips"],
               "z_hold_max": r["z_hold_max"], "pos_clips": len(zp),
               "pos_clips_above": sum(z >= r["theta_z"] for z in zp),
               "indicative": hold < 12}
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


def soak_rows(db, session: str, slot: str, dwell_s: float, k_fraction: float, p8: float, r_thr: float | None):
    emb = db.query("SELECT * FROM embed WHERE session=? ORDER BY ts", (session,))
    if not emb:
        return None
    tw = db.tripwire(slot)
    wins = {w["embed_id"]: w for w in db.query("SELECT * FROM win WHERE tripwire_id=? ORDER BY ts", (tw["id"],))}
    ts = [e["ts"] for e in emb]
    p50, _ = window_period(ts)
    stride = p50 or 1.0
    labels = label_times(db, session)
    steps = schedule_intervals(labels)
    step_starts = sorted(s["start"] for s in steps)
    suspends = [e["ts"] for e in db.query("SELECT ts FROM event WHERE session=? AND kind='suspend'", (session,))]
    th = db.query("SELECT theta FROM threshold WHERE tripwire_id=? AND source NOT LIKE 'nebius-job:%' "
                  "ORDER BY version DESC LIMIT 1", (tw["id"],))
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
    live = [e["ts"] for e in db.query("SELECT ts FROM event WHERE session=? AND kind='trip' AND tripwire_id=?",
                                      (session, tw["id"]))]
    trips = {"jaga (live)": live, **trips}
    still = [s for s in steps if s["kind"] == "still" and s["end"]]
    exits = [s for s in steps if s["kind"] in ("exit_leave", "exit_inview") and s["end"]]
    res = {}
    for name, tt in trips.items():
        cut = [any(s["start"] <= t <= s["end"] + stride for t in tt) for s in still]
        caught, ttt = [], []
        for x in exits:
            hit = [t for t in tt if x["start"] < t <= x["end"] + stride]
            caught.append(bool(hit))
            if hit:
                ttt.append(hit[0] - x["start"])
        res[name] = {"still_cut": sum(cut), "still_n": len(still), "still_cut_flags": cut,
                     "exits_caught": sum(caught), "exits_n": len(exits), "caught_flags": caught,
                     "median_time_to_trip_s": S.median(ttt), "trips": len(tt)}
    hours = (ts[-1] - ts[0]) / 3600 if len(ts) > 1 else 0
    timers = {}
    on_times = [ts[0]] + [e["ts"] for e in db.query("SELECT ts FROM event WHERE session=? AND kind='rearmed'",
                                                     (session,))]
    for T in (15, 30, 60):
        cuts = [t + T * 60 for t in on_times if t + T * 60 <= ts[-1]]
        mid = sum(1 for c in cuts if not any(x["start"] < c <= x["end"] for x in exits))
        after = [min([c - x["start"] for c in cuts if c > x["start"]] or [T * 60.0]) / 60 for x in exits]
        timers[f"{T} min"] = {"mid_session_cuts": mid, "minutes_on_after_exit_median": S.median(after)}
    return {"session": session, "windows": len(emb), "hours": round(hours, 3), "stride_s": stride, "theta": theta,
            "dwell_s": dwell_s, "systems": res, "plug_countdown": timers, "still_episodes": len(still),
            "exits": len(exits)}


def cmd_soak(a) -> int:
    from jaga.db import DB

    db = DB(a.db) if a.db else DB()
    results = {}
    for sess in a.session:
        r = soak_rows(db, sess, a.tripwire, a.dwell, a.k_fraction, a.p8, a.r)
        if r is None:
            print(f"{sess}: no windows logged")
            continue
        results[sess] = r
    if not results:
        save_section("soak", {"status": "not yet run", "why": "no scored Soak B session in the database"})
        write_demo()
        return 0
    pooled = {}
    for r in results.values():
        for name, v in r["systems"].items():
            p = pooled.setdefault(name, {"still_cut_flags": [], "caught_flags": [], "ttt": []})
            p["still_cut_flags"] += v["still_cut_flags"]
            p["caught_flags"] += v["caught_flags"]
            if v["median_time_to_trip_s"] is not None:
                p["ttt"].append(v["median_time_to_trip_s"])
    table = []
    for name, p in pooled.items():
        n, k = len(p["still_cut_flags"]), sum(p["still_cut_flags"])
        e, c = len(p["caught_flags"]), sum(p["caught_flags"])
        lo, hi = S.clopper_pearson(k, n) if n else (None, None)
        elo, ehi = S.clopper_pearson(c, e) if e else (None, None)
        table.append({"system": name, "still_episodes": n, "still_cut": k, "cut_ci95": [lo, hi], "exits": e,
                      "exits_caught": c, "recall_ci95": [elo, ehi], "median_time_to_trip_s": S.median(p["ttt"])})
    j = pooled.get("jaga (live)")
    t = pooled.get("stillness timer")
    paired = None
    if j and t and len(j["still_cut_flags"]) == len(t["still_cut_flags"]):
        b = sum(1 for x, y in zip(j["still_cut_flags"], t["still_cut_flags"], strict=True) if x and not y)
        c = sum(1 for x, y in zip(j["still_cut_flags"], t["still_cut_flags"], strict=True) if y and not x)
        paired = {"jaga_only": b, "timer_only": c, "sign_test_p": S.sign_test(b, c)}
    target = None
    jr = next((r for r in table if r["system"] == "jaga (live)"), None)
    tr = next((r for r in table if r["system"] == "stillness timer"), None)
    if jr and tr and jr["exits"]:
        target = {"fewer_than_half": jr["still_cut"] < tr["still_cut"] / 2,
                  "recall_ge_0.8": jr["exits_caught"] / jr["exits"] >= 0.8}
        target["both_hold"] = all(target.values())
    write_csv("soak.csv", table)
    save_section("soak", {"sessions": results, "pooled": table, "paired_vs_stillness": paired,
                          "targets": target, "dwell_s": a.dwell})
    write_demo()
    print(json.dumps({"pooled": table, "paired": paired, "targets": target}, indent=1, default=float))
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
        for name, m in variants.items():
            r = system_row(name, list(m[y == 1]), list(m[y == 0]))
            arms[slot][name] = r
            rows.append({"tripwire": slot, **{k: v for k, v in r.items() if k != "auc_ci95"}})
    if pooled_pos:  # (b) one global θ across tripwires: precision/false positives only (AUC is threshold-free)
        g = theta_at_recall(pooled_pos, 0.9)
        for slot in arms:
            fx, pos, neg, _ = caption_vectors(slot)
            sub = [c for c in clips if c["tripwire"].zfill(2) == slot and vecs[c["clip_id"]] is not None]
            V = np.stack([vecs[c["clip_id"]] for c in sub])
            y = np.array([int(c["label"]) for c in sub])
            m = margins(V, pos, neg)
            arms[slot]["(b) one global θ"] = {"theta": g, **S.confusion(list(m[y == 1]), list(m[y == 0]), g)}
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
    data = {"session": a.session, "empty_windows": len(empty), "P8_candidate": p8, "R_candidate": r,
            "exits_logged": len(exits), "note": "candidates only; the values used are committed in docs/PROTOCOL.md "
                                                 "before B1"}
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
                f" ({m.get('platform', '?')}); 1-min load average {m.get('loadavg_1m', '?')}*")

    L += ["## 1. Speed — Cosmos-Embed1-224p on this laptop", ""]
    sp = s.get("speed")
    if sp and sp.get("rows"):
        L += ["| device | precision | windows | p50 s/clip | p95 s/clip | torch | threads |",
              "|---|---|---|---|---|---|---|"]
        for r in sp["rows"]:
            L.append(f"| {r['device'].upper()} | fp32 | {r['n']} | {r['p50_s']} | {r['p95_s']} | {r['torch']} | "
                     f"{r['threads']} |")
        L += ["", f"One window = 8 frames → `preprocess` + `get_video_embeddings`, after 5 warm-up calls; input "
              f"`{sp['input']}` (the model card's own sample video). {stamp(sp)}", "",
              "Reproduce: `python scripts/bench.py speed --n 200`", ""]
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
                L.append(f"| {r['system']} | {r['n_pos']} / {r['n_neg']} | {r['auc']:.3f} ({r['auc_ci95'][0]:.3f}–"
                         f"{r['auc_ci95'][1]:.3f}) | {r['tp']} / {r['fp']} | {fmt(r['precision'])} |")
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
            L.append(f"- {slot}: μ₀/σ₀ from {r['n_fit']} normal windows; held-out normal windows {r['n_holdout']}, "
                     f"trips on them {r['holdout_trips']}; positive clips above θ_z {r['theta_z']}: "
                     f"{r['pos_clips_above']} of {r['pos_clips']}" + (" (indicative: < 12 windows)" if r['indicative']
                                                                      else ""))
        L += ["", stamp(fr), ""]
    else:
        L += ["Not yet run: needs the 2-minute normal-state captures (`normal/<tripwire>`).", ""]

    L += ["## 4. Soak B — someone standing still at the board (pre-registered: docs/PROTOCOL.md)", ""]
    so = s.get("soak")
    if so and so.get("pooled"):
        L += ["| system | still episodes cut (95 % CI) | exits caught (95 % CI) | median time-to-trip |",
              "|---|---|---|---|"]
        for r in so["pooled"]:
            ci, rc = r["cut_ci95"], r["recall_ci95"]
            L.append(f"| {r['system']} | {r['still_cut']} of {r['still_episodes']} ({fmt(ci[0], 2)}–{fmt(ci[1], 2)}) | "
                     f"{r['exits_caught']} of {r['exits']} ({fmt(rc[0], 2)}–{fmt(rc[1], 2)}) | "
                     f"{fmt(r['median_time_to_trip_s'], 0)} s |")
        if so.get("paired_vs_stillness"):
            p = so["paired_vs_stillness"]
            L.append(f"\nPaired on the same episodes: jaga-only cuts {p['jaga_only']}, timer-only cuts "
                     f"{p['timer_only']}, exact sign test p = {fmt(p['sign_test_p'], 3)}.")
        if so.get("targets"):
            t = so["targets"]
            L.append(f"\nPre-registered targets: fewer than half the timer's cuts: {t['fewer_than_half']}; "
                     f"recall on exits ≥ 0.8: {t['recall_ge_0.8']}.")
        L += ["", stamp(so), ""]
    else:
        L += ["Not yet run: Soak B1/B2 have not been recorded. Protocol and schedules: `docs/PROTOCOL.md`.", ""]

    L += ["## 5. Self-ablation", ""]
    ab = s.get("ablate")
    if ab and ab.get("tripwires"):
        for slot, arms in ab["tripwires"].items():
            L += [f"### Tripwire {slot}", "", "| arm | AUC | precision at recall 0.9 |", "|---|---|---|"]
            for name, r in arms.items():
                L.append(f"| {name} | {fmt(r.get('auc'))} | {fmt(r.get('precision'))} |")
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
          "jaga score --replay data/clips/iron/pos_03.mp4 --tripwire fixtures/tripwires/01.json   # no model download",
          "python scripts/bench.py clips      # AUC / precision from the committed vectors",
          "python scripts/bench.py speed      # needs the gated Cosmos-Embed1 weights (accept the licence on HF)",
          "```", ""]
    DEMO.write_text("\n".join(L))


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
    p.add_argument("--stride", type=float, default=1.0)
    p.set_defaults(fn=cmd_fresh)
    p = sub.add_parser("soak")
    p.add_argument("--session", action="append", required=True)
    p.add_argument("--tripwire", default="01")
    p.add_argument("--dwell", type=float, default=120.0)
    p.add_argument("--k-fraction", type=float, default=config.TUNING.k_fraction)
    p.add_argument("--p8", type=float, default=config.TUNING.still_motion_pct)
    p.add_argument("--r", type=float, default=None, help="reference-difference threshold R (docs/PROTOCOL.md)")
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
    p.set_defaults(fn=cmd_tune)
    p = sub.add_parser("demo")
    p.set_defaults(fn=lambda a: write_demo() or 0)
    a = ap.parse_args()
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
