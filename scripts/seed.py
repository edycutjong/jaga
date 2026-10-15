#!/usr/bin/env python3
"""Deterministic fixtures for jaga: frozen tripwires, labelled clips, clip vectors, frozen thresholds.

Order matters (leakage guard): each tripwire is compiled ONCE from its sentence alone and frozen before any
clip is scored; its spec is never edited afterwards. θ is computed from clips only after that.

    seed.py freeze-specs              compile 01/02/03 once with Nemotron 3 Super, save request IDs, embed captions
    seed.py freeze-ablation           ablation (c-id): Super's compile of the Indonesian sentence, frozen once
    seed.py variance --k 5            K extra compiles per sentence (compile variance; never used for θ)
    seed.py import --raw DIR          labelled recordings <set>/<pos|neg>_<NN>_<lighting>_<camera_pos>.<ext>
                                      → 8-s windows at 1 fps, fixtures/clips.csv (labels from the filename)
    seed.py embed-clips               Cosmos-Embed1 window vector per clip → fixtures/vecs/<set>/<clip>.npy
    seed.py freeze-theta --version 1  θ_m = highest margin with clip recall ≥ 0.9 → fixtures/tripwires/<slot>.json
    seed.py verify                    re-score every clip and compare with fixtures/decisions_frozen.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jaga import config  # noqa: E402
from jaga.offline import CLIPS_CSV, DECISIONS_FROZEN, VECS  # noqa: E402

TRIPWIRES = REPO / "fixtures" / "tripwires"
VARIANCE = TRIPWIRES / "variance"
LOCAL_CLIPS = config.jaga_home() / "clips"  # every imported clip, local only
PUBLIC_CLIPS = REPO / "data" / "clips"  # only clips listed in PUBLISH.txt

# The three events to be measured. 02 is the fridge; cat clips, if recorded, get their own tripwire 04.
SENTENCES = {
    "01": ("iron", "nobody is at the ironing board"),
    "02": ("fridge", "the fridge door is left open and nobody is there"),
    "03": ("door", "the door is left open and nobody is there"),
}
CAT_SENTENCE = ("cat", "the cat is on the kitchen counter")
SET_TO_SLOT = {"iron": "01", "iron-inview": "01", "iron-day2": "01", "fridge": "02", "cat": "04", "door": "03"}
HELDOUT_SETS = {"iron-day2"}
TEST_ONLY_SETS = {"iron-inview"}  # scored at the frozen θ, never used to fit it (a separate recall, per exit type)
NAME_RE = re.compile(r"^(?P<label>pos|neg)_(?P<n>\d{1,3})_(?P<lighting>day|lamp|dusk)_(?P<pos>[a-z0-9-]+)$", re.I)
CLIP_COLUMNS = ["clip_id", "set", "tripwire", "label", "recorded_at", "lighting", "camera_pos", "heldout",
                "window_start_s", "source_duration_s", "source_fps", "published", "notes"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ----------------------------------------------------------------------------------- freeze-specs
def compile_frozen(slot: str, event: str, sentence: str, embedder, out: Path) -> dict:
    from jaga.compile import compile_sentence
    from jaga.tripwires import save_vectors

    r = compile_sentence(sentence).to_dict()
    spec = r["spec"]
    if not spec["visual"]:
        raise SystemExit(f"{slot}: Super said the sentence is not visual: {spec['reason']}")
    fx = {"slot": slot, "event": event, "sentence": sentence, "spec": spec,
          "tf_request_id": r["tf_request_id"], "tf_response_id": r["tf_response_id"], "model": r["model"],
          "temperature": r["temperature"], "top_p": r["top_p"], "max_tokens": r["max_tokens"], "shape": r["shape"],
          "created": r["created"], "latency_s": r["latency_s"], "usage": r["usage"],
          "frozen_at": now(),
          "leakage_guard": "compiled once from the sentence alone, before any clip was scored; never edited",
          "caption_vectors": f"{out.stem}.vec.npz — Cosmos-Embed1 text_proj (256-d), positives and hard negatives "
                             "in spec order",
          "frozen": None}
    out.write_text(json.dumps(fx, indent=2, ensure_ascii=False) + "\n")
    fx["_path"] = str(out)
    pos = embedder.embed_text(spec["positives"])
    neg = embedder.embed_text(spec["hard_negatives"])
    save_vectors(fx, pos, neg)
    return fx


def cmd_freeze_specs(a) -> int:
    from jaga.embed import shared

    TRIPWIRES.mkdir(parents=True, exist_ok=True)
    emb = shared(a.device)
    sentences = dict(SENTENCES)
    slots = [s.zfill(2) for s in a.slots.split(",")]
    if a.cat:  # cat clips instead of fridge: a new tripwire 04; the frozen 02 is never recompiled (leakage guard)
        sentences["04"] = CAT_SENTENCE
        slots = ["04"]
    for slot in slots:
        out = TRIPWIRES / f"{slot}.json"
        if out.exists():
            print(f"{slot}: already frozen ({out.name}) — never recompiled (leakage guard)")
            continue
        event, sentence = sentences[slot]
        fx = compile_frozen(slot, event, sentence, emb, out)
        s = fx["spec"]
        print(f"{slot} “{sentence}” → {s['name']} · {len(s['positives'])} pos / {len(s['hard_negatives'])} neg · "
              f"dwell {s['dwell_s']} s → {s['action']} · request {fx['tf_request_id']}")
    return 0


ABLATION = TRIPWIRES / "ablation"
INDONESIAN_01 = "nggak ada orang di meja setrika"  # "nobody is at the ironing board", as typed in Jakarta


def cmd_freeze_ablation(a) -> int:
    """Ablation (c-id): Super's compile of the raw Indonesian sentence, frozen once before any clip is scored."""
    from jaga.embed import shared

    ABLATION.mkdir(parents=True, exist_ok=True)
    out = ABLATION / "01-id.json"
    if out.exists():
        print(f"{out.name}: already frozen")
        return 0
    fx = compile_frozen("01-id", "iron", INDONESIAN_01, shared(a.device), out)
    fx = json.loads(out.read_text())
    fx["use"] = "ablation (c-id) only: Super(Indonesian sentence) vs the raw Indonesian sentence embedded directly"
    out.write_text(json.dumps(fx, indent=2, ensure_ascii=False) + "\n")
    print(f"01-id “{INDONESIAN_01}” → {fx['spec']['name']} · request {fx['tf_request_id']}"
          f" · language {fx['spec']['language_in']}")
    return 0


def cmd_variance(a) -> int:
    from jaga.compile import compile_sentence
    from jaga.embed import shared
    from jaga.tripwires import load

    emb = shared(a.device)
    VARIANCE.mkdir(parents=True, exist_ok=True)
    for slot in [s.zfill(2) for s in a.slots.split(",")]:
        base = load(slot)
        for k in range(1, a.k + 1):
            out = VARIANCE / f"{slot}-k{k}.json"
            if out.exists():
                continue
            r = compile_sentence(base["sentence"]).to_dict()
            d = {"slot": slot, "draw": k, "sentence": base["sentence"], "spec": r["spec"],
                 "tf_request_id": r["tf_request_id"], "model": r["model"], "temperature": r["temperature"],
                 "top_p": r["top_p"], "created": r["created"], "use": "compile variance only (never used for θ)"}
            out.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
            import numpy as np

            np.savez(out.with_suffix("").with_suffix(".vec.npz"), pos=emb.embed_text(r["spec"]["positives"]),
                     neg=emb.embed_text(r["spec"]["hard_negatives"]))
            print(f"{slot} draw {k}: {r['spec']['name']} · request {r['tf_request_id']}")
    return 0


# ----------------------------------------------------------------------------------- import
def read_clips_csv() -> list[dict]:
    if not CLIPS_CSV.exists():
        return []
    with CLIPS_CSV.open() as f:
        return list(csv.DictReader(f))


def write_clips_csv(rows: list[dict]) -> None:
    CLIPS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["clip_id"])
    with CLIPS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLIP_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CLIP_COLUMNS})


def recorded_at(path: Path, meta_time: str | None) -> str:
    if meta_time:
        return meta_time
    st = path.stat()
    t = getattr(st, "st_birthtime", st.st_mtime)
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))


def write_window_mp4(src: Path, start_s: float, dst: Path) -> None:
    """The 8-s window as a small H.264 file (480p) for local re-scoring and, if allowed, publication."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("ffmpeg not found (needed to cut the 8-s windows)")
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(src), "-t", "8",
                    "-an", "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-pix_fmt", "yuv420p", str(dst)], check=True)


def cmd_import(a) -> int:
    from jaga.frames import clip_window_start, probe

    raw = Path(a.raw).expanduser()
    if not raw.exists():
        raise SystemExit(f"{raw} does not exist")
    publish = set()
    pl = raw / "PUBLISH.txt"
    if pl.exists():
        publish = {ln.strip() for ln in pl.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    rows = {r["clip_id"]: r for r in read_clips_csv()}
    added = collisions = 0
    seen_now: dict[str, str] = {}
    for set_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        if set_dir.name in ("normal", "outside"):
            continue
        if set_dir.name not in SET_TO_SLOT:
            print(f"skip {set_dir.name}/ (unknown set)")
            continue
        for f in sorted(set_dir.iterdir()):
            if f.suffix.lower() not in (".mov", ".mp4", ".m4v", ".webm", ".mkv"):
                continue
            m = NAME_RE.match(f.stem)
            if not m:
                print(f"skip {f.name}: expected <pos|neg>_<NN>_<day|lamp|dusk>_<camera_pos>")
                continue
            cid = f"{set_dir.name}/{m['label'].lower()}_{int(m['n']):02d}"
            if cid in seen_now:
                print(f"ERROR {f.name}: {cid} is also {seen_now[cid]} — every clip needs its own number within a set "
                      "and label; rename one and import again")
                collisions += 1
                continue
            seen_now[cid] = f.name
            if cid in rows and not a.force:
                old = rows[cid]
                if (old.get("lighting"), old.get("camera_pos")) != (m["lighting"].lower(), m["pos"].lower()):
                    print(f"ERROR {f.name}: {cid} was already imported from a {old.get('lighting')} clip at "
                          f"{old.get('camera_pos')} — give this one its own number")
                    collisions += 1
                continue  # the same clip, imported before
            info = probe(f)
            if info.duration_s < 8.0:
                print(f"skip {f.name}: {info.duration_s:.1f} s is shorter than the 8-s window")
                continue
            start = clip_window_start(info.duration_s, a.window)
            dst = LOCAL_CLIPS / f"{cid}.mp4"
            write_window_mp4(f, start, dst)
            pub = f"{set_dir.name}/{f.name}" in publish or f.name in publish
            if pub:  # only clips the human listed in PUBLISH.txt are ever committed as video
                (PUBLIC_CLIPS / set_dir.name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(dst, PUBLIC_CLIPS / f"{cid}.mp4")
            rows[cid] = {"clip_id": cid, "set": set_dir.name, "tripwire": SET_TO_SLOT[set_dir.name],
                         "label": 1 if m["label"].lower() == "pos" else 0,
                         "recorded_at": recorded_at(f, info.creation_time),
                         "lighting": m["lighting"].lower(), "camera_pos": m["pos"].lower(),
                         "heldout": 1 if set_dir.name in HELDOUT_SETS else 0, "window_start_s": f"{start:.3f}",
                         "source_duration_s": f"{info.duration_s:.2f}", "source_fps": f"{info.fps:.2f}",
                         "published": 1 if pub else 0, "notes": f"window={a.window}; label from the filename"}
            added += 1
            print(f"{cid}: {'POS' if rows[cid]['label'] == 1 else 'neg'} · window {start:.1f}–{start + 8:.1f} s"
                  + (" · published" if pub else ""))
    normal = raw / "normal"
    if normal.exists():
        dst_dir = config.jaga_home() / "normal"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(normal.rglob("*")):
            if f.suffix.lower() in (".mov", ".mp4", ".m4v"):
                tag = f.parent.name if f.parent != normal else f.stem
                shutil.copy2(f, dst_dir / f"{tag}{f.suffix.lower()}")
                print(f"normal/{tag}: copied for the fresh-mode row ({probe(f).duration_s:.0f} s)")
    write_clips_csv(list(rows.values()))
    print(f"{added} clip(s) imported · {len(rows)} in fixtures/clips.csv" + (f" · {collisions} name collision(s)"
                                                                            if collisions else ""))
    return 1 if collisions else 0


# ----------------------------------------------------------------------------------- embed / freeze / verify
def cmd_embed_clips(a) -> int:
    import numpy as np

    from jaga.embed import shared
    from jaga.offline import embed_clip

    emb = shared(a.device)
    n = 0
    for r in read_clips_csv():
        out = VECS / f"{r['clip_id']}.npy"
        if out.exists() and not a.force:
            continue
        src = LOCAL_CLIPS / f"{r['clip_id']}.mp4"
        if not src.exists():
            print(f"missing local clip {src}")
            continue
        vec, _ = embed_clip(src, emb, "start")  # the stored file IS the 8-s window
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, vec.astype(np.float32))
        n += 1
    print(f"embedded {n} clip(s) on {emb.device} fp32 → fixtures/vecs/")
    (VECS / "MACHINE.json").parent.mkdir(parents=True, exist_ok=True)
    (VECS / "MACHINE.json").write_text(json.dumps({**emb.info, "processor": platform.processor(),
                                                   "embedded_at": now()}, indent=2) + "\n")
    return 0


def clip_margins(slot: str, include_heldout: bool = False) -> list[tuple[str, int, float]]:
    from jaga.offline import committed_vector
    from jaga.score import margin
    from jaga.tripwires import load, load_vectors

    pos, neg = load_vectors(load(slot))
    out = []
    for r in read_clips_csv():
        if r["tripwire"].zfill(2) != slot or (r["heldout"] == "1" and not include_heldout):
            continue
        if r["set"] in TEST_ONLY_SETS and not include_heldout:
            continue
        v = committed_vector(r["clip_id"])
        if v is not None:
            out.append((r["clip_id"], int(r["label"]), margin(v, pos, neg)))
    return out


def cmd_freeze_theta(a) -> int:
    from jaga.score import theta_at_recall

    rows_out = []
    for slot in [s.zfill(2) for s in a.slots.split(",")]:
        p = TRIPWIRES / f"{slot}.json"
        fx = json.loads(p.read_text())
        if fx.get("frozen") and not a.force:
            print(f"{slot}: θ already frozen (v{fx['frozen']['version']})")
            continue
        ms = clip_margins(slot)
        pos = [m for _, y, m in ms if y == 1]
        if not pos:
            print(f"{slot}: no positive clip margins yet")
            continue
        theta = theta_at_recall(pos, a.recall)
        fx["frozen"] = {"theta_m": round(theta, 6), "version": a.version, "recall_target": a.recall,
                        "source": "clips: highest margin with recall ≥ target (full data)", "n_pos": len(pos),
                        "n_neg": sum(1 for _, y, _ in ms if y == 0), "set_at": now(), "frozen_for": a.frozen_for}
        p.write_text(json.dumps(fx, indent=2, ensure_ascii=False) + "\n")
        for cid, y, m in ms:
            rows_out.append({"clip_id": cid, "tripwire": slot, "label": y, "margin": f"{m:.6f}",
                             "theta_m": f"{theta:.6f}", "theta_version": a.version, "above": int(m >= theta)})
        print(f"{slot}: θ_m {theta:+.6f} (v{a.version}) from {len(pos)} positive clips")
    if rows_out:
        with DECISIONS_FROZEN.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows_out)
    return 0


def cmd_verify(a) -> int:
    from jaga.offline import committed_vector, published_decision, score_vector
    from jaga.tripwires import frozen_threshold, load, load_vectors

    bad = n = 0
    for r in read_clips_csv():
        slot = r["tripwire"].zfill(2)
        fx = load(slot)
        th = frozen_threshold(fx)
        v = committed_vector(r["clip_id"])
        pub = published_decision(r["clip_id"], slot)
        if th is None or v is None or pub is None:
            continue
        pos, neg = load_vectors(fx)
        s = score_vector(v, pos, neg, th.theta)
        n += 1
        same = int(pub["above"]) == int(s["above"]) and abs(float(pub["margin"]) - s["m"]) < 1e-4
        if not same:
            bad += 1
            print(f"MISMATCH {r['clip_id']}: published {pub['margin']} / {pub['above']} vs {s['m']:.6f} / {s['above']}")
    print(f"verify: {n - bad}/{n} clips match fixtures/decisions_frozen.csv")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze-specs")
    p.add_argument("--slots", default="01,02,03")
    p.add_argument("--cat", action="store_true", help="freeze the cat variant as tripwire 04 (only if cat clips exist)")
    p.set_defaults(fn=cmd_freeze_specs)
    p = sub.add_parser("freeze-ablation")
    p.set_defaults(fn=cmd_freeze_ablation)
    p = sub.add_parser("variance")
    p.add_argument("--slots", default="01,02,03")
    p.add_argument("--k", type=int, default=5)
    p.set_defaults(fn=cmd_variance)
    p = sub.add_parser("import")
    p.add_argument("--raw", default=os.environ.get("JAGA_RAW_CLIPS", str(Path.home() / ".jaga" / "raw-clips")))
    p.add_argument("--window", default="middle", choices=["start", "middle", "end"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_import)
    p = sub.add_parser("embed-clips")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_embed_clips)
    p = sub.add_parser("freeze-theta")
    p.add_argument("--slots", default="01,02,03")
    p.add_argument("--version", type=int, default=1)
    p.add_argument("--recall", type=float, default=0.9)
    p.add_argument("--frozen-for", default=None, help="the scored session(s) this θ is frozen for, e.g. B1,B2,A1")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_freeze_theta)
    p = sub.add_parser("verify")
    p.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
