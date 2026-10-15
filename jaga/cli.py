"""jaga — type what to watch for; NVIDIA Cosmos-Embed1 watches a webcam and a smart plug acts.

    jaga serve      the live watcher + overlay on http://127.0.0.1:8765 (camera, or --source <file>)
    jaga score      score a recorded clip with a frozen tripwire (--replay: committed vectors, no model)
    jaga compile    compile one sentence into a TripwireSpec with Nemotron 3 Super (Token Factory)
    jaga record     guided recorder for a labelled clip (label chosen before recording)
    jaga soak       a scored session (Soak A / Soak B) on its pre-registered schedule
    jaga label      log a label into the running session
    jaga calibrate  dispatch the calibration Serverless Job on Nebius AI Cloud
    jaga plug       plug status, or switch it OFF (there is no ON here: see /plug/test)
    jaga info       versions and where local state lives
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _zone(s: str | None):
    if not s:
        return None
    v = tuple(float(x) for x in s.split(","))
    if len(v) != 4 or not (0 <= v[0] < v[2] <= 1 and 0 <= v[1] < v[3] <= 1):
        raise argparse.ArgumentTypeError("zone is x0,y0,x1,y1 as fractions of the frame, e.g. 0.3,0.2,0.9,1")
    return v


def _slots(s: str | None) -> list[str]:
    return [x.strip().zfill(2) for x in (s or "").split(",") if x.strip()]


# ------------------------------------------------------------------------------------------- serve
def session_used(db, name: str) -> bool:
    """A session name with windows or labels is taken (a start that failed before its first window is not)."""
    return bool(db.query("SELECT 1 FROM embed WHERE session=? LIMIT 1", (name,))
                or db.query("SELECT 1 FROM label WHERE session=? LIMIT 1", (name,)))


def port_free(port: int | None) -> None:
    """Refuse to start (before the model loads) when the overlay's port is taken — usually another jaga."""
    if not port:
        return
    import socket

    from . import config

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as uvicorn binds: a just-closed jaga does not count
    try:
        s.bind((config.HOST, int(port)))
    except OSError:
        raise SystemExit(f"port {port} is in use — another jaga is probably running: stop it (Ctrl-C in its "
                         "terminal) or pass --port") from None
    finally:
        s.close()


def build_watcher(a, *, scored: bool = False, session: str | None = None):
    from .act import Notifier, load_driver
    from .capture import CameraSource, FileSource, Ring
    from .config import TUNING
    from .daemon import Watcher
    from .db import DB
    from .embed import shared
    from .guards import Guards
    from .server import Broadcaster
    from .tripwires import fresh_threshold, frozen_threshold, load, load_vectors

    db = DB(a.db) if a.db else DB()
    session = session or a.session or time.strftime("live-%Y%m%d-%H%M%S")
    if session_used(db, session):
        raise SystemExit(f"session {session!r} already has data: every run counts, so a new run needs a new name "
                         "(e.g. B1-run2, U1-run2); the earlier run stays on record")
    missing = [s for s in _slots(getattr(a, "arm", None)) if s not in _slots(a.load)]
    if missing:
        raise SystemExit(f"--arm {','.join(missing)}: arm only tripwires you --load (loaded: {a.load or 'none'})")
    fixtures = []
    for slot in _slots(a.load):  # every refusal happens before the model loads and before the session is opened
        fx = load(slot)
        th = frozen_threshold(fx)
        if scored and th is None:
            raise SystemExit(f"tripwire {slot}: a scored session needs a frozen θ (scripts/seed.py freeze-theta)")
        fixtures.append((slot, fx, th))
    port_free(getattr(a, "port", None))
    print("loading NVIDIA Cosmos-Embed1-224p …", flush=True)
    emb = shared(a.device)
    print(f"  {emb.info['device']} fp32 · loaded in {emb.load_s:.1f} s", flush=True)
    ring = Ring()
    if a.source == "camera":
        src = CameraSource(ring, a.camera)
    else:
        src = FileSource(ring, a.source, realtime=not a.offline, loop=a.loop)
    driver = load_driver()
    notifier = Notifier(os_banner=not a.no_os_banner)
    bc = Broadcaster()
    w = Watcher(db=db, embedder=emb, ring=ring, driver=driver, notifier=notifier, session=session,
                camera_pos=a.camera_pos, scored=scored, tuning=TUNING, guards=Guards(TUNING, zone=a.zone),
                publish=bc.publish, source=src, frozen_for=session if scored else None, offline=a.offline,
                labels=getattr(a, "labels", False))
    for slot, fx, th in fixtures:
        pos, neg = load_vectors(fx)
        if not scored:
            job = db.latest_job_threshold(slot)
            if job and (th is None or job["set_at"] > (fx.get("frozen") or {}).get("set_at", "")):
                from .score import Threshold

                th = Threshold("labelled-m", job["theta"], job["mu0"], job["sigma0"], job["version"], job["source"])
        w.load_tripwire(slot, fx["sentence"], fx["spec"], pos, neg, th or fresh_threshold(TUNING),
                        {k: fx.get(k) for k in ("tf_request_id", "tf_response_id", "model", "created", "temperature",
                                                "operating")})
    for item in a.dwell or []:
        slot, sec = item.split("=")
        w.override_dwell(slot.zfill(2), float(sec), "demo")
    return w, src, bc, emb


def cmd_serve(a, *, scored: bool = False, session: str | None = None, on_started=None) -> int:
    from . import config
    from .compile import compile_sentence
    from .server import create_app, run

    w, src, bc, emb = build_watcher(a, scored=scored, session=session)
    app = create_app(w, compile_fn=lambda s: compile_sentence(s).to_dict(), embed_text_fn=emb.embed_text,
                     broadcaster=bc, port=a.port, info={**emb.info, "session": w.session})
    w.start()
    src.start()
    for slot in _slots(a.arm):
        w.arm(slot, a.capture_s)
    if on_started:
        on_started(w, app)

    def ready():
        print(f"jaga is watching · session {w.session} · http://{config.HOST}:{a.port}/  (Ctrl-C to stop)", flush=True)

    def _term(signum, frame):  # SIGTERM ends like Ctrl-C: the watcher flushes its vectors and closes the session
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    try:
        run(app, a.port, on_ready=ready)
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()
        w.stop()
        print("jaga stopped", flush=True)
    return 0


def cmd_soak(a) -> int:
    from .soak import ScheduleRunner, load_schedule

    if not a.session:
        raise SystemExit("a soak needs its own --session name (B1, B2, A1 …)")
    if a.kind == "B":
        if not a.schedule:
            raise SystemExit("Soak B needs its pre-registered schedule: --schedule docs/protocol/<session>.csv")
        a.load = a.load or "01"
        a.arm = a.arm or a.load
    else:
        if not a.load:
            raise SystemExit("Soak A: name the tripwires in this camera view, e.g. --load 02,03")
        a.arm = a.arm or a.load
    steps = load_schedule(a.schedule) if a.schedule else []
    started = threading.Event()

    def watch_capture(w, app):
        def run():  # the schedule clock starts with the first window: step 0 is the 2-min normal capture
            while w.windows < 1:
                time.sleep(0.5)
            started.set()

        def complete():  # the schedule is over: stop watching, so nothing after it is mistaken for the session
            print("schedule complete — jaga stops in 60 s", flush=True)
            time.sleep(60)
            server = getattr(app.state, "server", None)
            if server is not None:
                server.should_exit = True

        threading.Thread(target=run, daemon=True).start()
        if steps:
            ScheduleRunner(w, steps, say=not a.no_say, started=started, on_complete=complete).start()

    caffeinate = None
    if sys.platform == "darwin":  # a soak must not be ended by sleep: hold an assertion for as long as jaga runs
        try:
            caffeinate = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
        except OSError:
            print("warning: caffeinate not available — keep the laptop awake yourself", file=sys.stderr)
    try:
        return cmd_serve(a, scored=True, session=a.session, on_started=watch_capture)
    finally:
        if caffeinate:
            caffeinate.terminate()


# ------------------------------------------------------------------------------------------- score
def cmd_score(a) -> int:
    from .offline import clip_id, committed_vector, embed_clip, published_decision, score_vector
    from .tripwires import frozen_threshold, load, load_vectors

    fx = load(a.tripwire)
    pos, neg = load_vectors(fx)
    th = frozen_threshold(fx)
    slot = fx["slot"]
    out = {"tripwire": slot, "sentence": fx["sentence"], "tf_request_id": fx.get("tf_request_id")}
    if a.fresh:
        return _score_fresh(a, fx, pos, neg, out)
    cid = clip_id(a.clip)
    out["clip"] = cid
    if a.replay:
        vec = committed_vector(cid)
        if vec is None:
            print(f"no committed vector for {cid} (fixtures/vecs/{cid}.npy) — run without --replay", file=sys.stderr)
            return 2
        out["source"] = f"replay: fixtures/vecs/{cid}.npy (no model download)"
    else:
        from .embed import shared

        vec, meta = embed_clip(a.clip, shared(a.device), a.window)
        out["source"] = f"live embed on {shared().device} fp32 · window {meta['start_s']}–{meta['start_s'] + 8:.1f} s"
    r = score_vector(vec, pos, neg, None if th is None else th.theta)
    out.update(margin=round(r["m"], 6), theta_m=None if th is None else round(th.theta, 6),
               theta_version=None if th is None else th.version, above=r["above"])
    pub = published_decision(cid, slot)
    if pub is not None and r["above"] is not None:
        out["published_above"] = pub.get("above") in ("1", "True", "true")
        out["matches_published"] = out["published_above"] == r["above"]
        out["published_margin"] = float(pub["margin"])
    if a.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"{cid} · tripwire {slot} “{fx['sentence']}”")
        print(f"  {out['source']}")
        if th is None:
            print(f"  margin {r['m']:+.6f}  (no frozen θ yet: margin only)")
        else:
            flag = "ABOVE θ (would count toward a trip)" if r["above"] else "below θ"
            print(f"  margin {r['m']:+.6f}  θ_m {th.theta:+.6f} (v{th.version})  → {flag}")
        if "matches_published" in out:
            print(f"  matches fixtures/decisions_frozen.csv: {'yes' if out['matches_published'] else 'NO'}")
        print("  (one 8-s clip is one window; the dwell is not evaluated on a single clip)")
    return 0


def _score_fresh(a, fx, pos, neg, out) -> int:
    from .config import TUNING
    from .embed import shared
    from .offline import embed_long, fresh_mode

    if not a.normal:
        raise SystemExit("--fresh needs --normal <2-min normal-state recording>")
    emb = shared(a.device)
    stride = max(1, round(a.stride))
    n_ts, n_v = embed_long(a.normal, emb, stride)
    p_ts, p_v = embed_long(a.clip, emb, stride)
    r = fresh_mode(n_ts, n_v, p_ts, p_v, pos, neg, dwell_s=a.dwell_s, stride_s=float(stride), tuning=TUNING)
    out.update(mode="fresh", positive=str(a.clip), normal=str(a.normal), stride_s=stride, dwell_s=a.dwell_s, **r)
    print(json.dumps(out, indent=2, default=float))
    return 0 if r.get("ok") else 1


# ------------------------------------------------------------------------------------------- compile
def cmd_compile(a) -> int:
    from .compile import CompileError, compile_sentence

    try:
        r = compile_sentence(a.sentence)
    except CompileError as e:
        print(f"compile failed: {e.kind}: {e.detail}", file=sys.stderr)
        return 1
    d = r.to_dict()
    if a.json:
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return 0
    s = d["spec"]
    print(f"“{d['sentence']}” → {s['name']}  (visual={s['visual']})")
    if not s["visual"]:
        print(f"  can't see that: {s['reason']}")
    for c in s["positives"]:
        print(f"  + {c}")
    for c in s["hard_negatives"]:
        print(f"  − {c}")
    print(f"  dwell {s['dwell_s']} s → {s['action']} · cool-down {s['cooldown_s']} s · language {s['language_in']}")
    print(f"  {d['model']} · request {d['tf_request_id']} · {d['latency_s']} s")
    return 0


# ------------------------------------------------------------------------------------------- record
def cmd_record(a) -> int:
    import cv2

    if not re.fullmatch(r"[a-z0-9-]+", a.camera_pos, re.I):  # the importer reads the name back: no underscores
        raise SystemExit("--camera-pos takes letters, digits and hyphens only (e.g. board, kitchen-door)")
    out_dir = Path(a.out).expanduser() / a.set
    out_dir.mkdir(parents=True, exist_ok=True)
    n = a.n
    if n is None:
        stems = [p.stem.split("_") for p in out_dir.glob(f"{a.label}_*")]
        taken = sorted(int(s[1]) for s in stems if len(s) > 1 and s[1].isdigit())
        n = (taken[-1] + 1) if taken else 1
    name = f"{a.label}_{n:02d}_{a.lighting}_{a.camera_pos}.mp4"
    path = out_dir / name
    if path.exists():
        raise SystemExit(f"{path} exists — pick another --n")
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    cap = cv2.VideoCapture(a.camera, backend)
    if not cap.isOpened():
        raise SystemExit("camera unavailable — allow camera access for this terminal "
                         "(System Settings → Privacy & Security → Camera)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"label {a.label.upper()} · set {a.set} · {name} · camera {a.camera} (use --camera 1 for a USB webcam)")
    for k in range(a.countdown, 0, -1):
        print(f"  recording in {k} …", flush=True)
        time.sleep(1)
    tmp = path.with_suffix(".part.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # streamed: no frames held in memory
    frames = 0
    t0 = time.monotonic()
    t_end = t0 + a.seconds
    while time.monotonic() < t_end:
        ok, f = cap.read()
        if ok:
            vw.write(f)
            frames += 1
    elapsed = time.monotonic() - t0
    vw.release()
    cap.release()
    if frames == 0:
        tmp.unlink(missing_ok=True)
        raise SystemExit("the camera delivered no frames — nothing saved (check the camera and its permission)")
    real_fps = frames / elapsed if elapsed > 0 else fps
    if real_fps < 0.9 * fps:
        if not shutil.which("ffmpeg"):
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"the camera delivered {real_fps:.1f} fps, not the {fps:.0f} it reports, and ffmpeg is "
                             "not installed to retime the clip — nothing saved (brew install ffmpeg, or more light)")
        # the camera delivered fewer frames than it reports (dim light): retime so the file lasts the real time
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-filter:v",
                        f"setpts=PTS*{fps / real_fps:.6f}", "-r", f"{real_fps:.3f}", "-an", str(path)], check=True)
        tmp.unlink()
        print(f"  note: the camera delivered {real_fps:.1f} fps, not the {fps:.0f} it reports; retimed to real time")
    else:
        tmp.rename(path)
    print(f"  saved {path} ({frames} frames over {elapsed:.1f} s). Next: scripts/seed.py import --raw {a.out}")
    return 0


# ------------------------------------------------------------------------------------------- label / plug / info
def cmd_label(a) -> int:
    from .db import DB

    db = DB()
    session = a.session or db.open_session()
    if not session:
        raise SystemExit("no open session — start `jaga serve` or `jaga soak` first, or pass --session")
    if not db.query("SELECT 1 FROM session WHERE name=?", (session,)):
        raise SystemExit(f"no session named {session!r}")
    last = db.query("SELECT MAX(ts) AS t FROM embed WHERE session=?", (session,))[0]["t"]
    if last is None or time.time() - last > 300:
        print(f"warning: session {session} has no window in the last 5 minutes — is jaga still running?",
              file=sys.stderr)
    lid = db.add_label(session, a.kind, a.note or "")
    print(f"label {lid}: {a.kind} → session {session}")
    return 0


def cmd_plug(a) -> int:
    from .act import ActionFailed, load_driver

    d = load_driver()
    if a.what == "status":
        st = d.status()
        print(json.dumps({**d.describe(), "on": st}))
        return 0
    try:
        ack = d.turn_off()
    except ActionFailed as e:
        print(f"ACTION FAILED: {e}", file=sys.stderr)
        return 1
    print(f"OFF acknowledged in {ack.latency_ms:.0f} ms ({ack.detail})")
    return 0


def cmd_calibrate(a) -> int:
    from .calibrate import dispatch

    return dispatch(smoke=a.smoke, input_csv=a.input, wait=not a.no_wait, dry_run=a.dry_run)


def cmd_info(a) -> int:
    from . import __version__, config

    info = {"jaga": __version__, "home": str(config.jaga_home()), "config_dir": str(config.config_dir()),
            "model": config.COSMOS_MODEL, "revision": config.COSMOS_REVISION, "compiler": config.TF_MODEL,
            "server": f"http://{config.HOST}:{config.PORT}/"}
    try:
        import torch

        info.update(torch=torch.__version__, mps=torch.backends.mps.is_available())
    except ImportError:
        pass
    print(json.dumps(info, indent=2))
    return 0


# ------------------------------------------------------------------------------------------- parser
def _serve_args(p: argparse.ArgumentParser) -> None:
    from . import config

    p.add_argument("--source", default="camera", help="'camera' (default) or a video file to play through the loop")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--offline", action="store_true", help="file source: no real-time pacing (media time)")
    p.add_argument("--loop", action="store_true", help="file source: loop the recording")
    p.add_argument("--session", help="session name (default live-<time>)")
    p.add_argument("--camera-pos", default=None, help="name of this camera position (logged on every window)")
    p.add_argument("--zone", type=_zone, default=None, help="person-in-zone baseline box x0,y0,x1,y1 (fractions)")
    p.add_argument("--load", default="", help="frozen tripwires to load, e.g. 01,03")
    p.add_argument("--arm", default="", help="tripwires to arm at start (2-min normal capture), e.g. 01")
    p.add_argument("--capture-s", type=float, default=None, help="normal-state capture length (default 120)")
    p.add_argument("--dwell", action="append", help="labelled demo dwell override, e.g. 01=30")
    p.add_argument("--no-os-banner", action="store_true",
                   help="no macOS banner (filming): the overlay row is the notice")
    p.add_argument("--labels", action="store_true",
                   help="turn on the label keys (1-7, 0) for an unscored session, e.g. the U1 tuning session")
    p.add_argument("--port", type=int, default=config.PORT)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--db", default=None, help="SQLite path (default ~/.jaga/jaga.db)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="jaga", description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="live watcher + overlay")
    _serve_args(p)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("soak", help="a scored session on its pre-registered schedule")
    p.add_argument("kind", choices=["A", "B"])
    _serve_args(p)
    p.add_argument("--schedule", help="docs/protocol/<session>.csv (Soak B)")
    p.add_argument("--no-say", action="store_true", help="do not speak the schedule")
    p.set_defaults(fn=cmd_soak)

    p = sub.add_parser("score", help="score a recorded clip with a frozen tripwire")
    p.add_argument("clip", help="clip file, or a clip id like iron/pos_03 with --replay")
    p.add_argument("--tripwire", "--spec", dest="tripwire", required=True,
                   help="fixtures/tripwires/<slot>.json or a slot number")
    p.add_argument("--replay", action="store_true", help="use the committed vector: no model download, no camera")
    p.add_argument("--window", default="middle", choices=["start", "middle", "end"])
    p.add_argument("--fresh", action="store_true", help="fresh mode over a long positive clip (outside people)")
    p.add_argument("--normal", help="--fresh: the 2-min normal-state recording")
    p.add_argument("--dwell-s", type=float, default=30.0, help="--fresh: dwell for the k-of-n trip (default 30)")
    p.add_argument("--stride", type=float, default=1.0, help="--fresh: window stride in seconds (the live period)")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_score)

    p = sub.add_parser("compile", help="sentence → TripwireSpec (Nemotron 3 Super on Token Factory)")
    p.add_argument("sentence")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_compile)

    p = sub.add_parser("record", help="guided recorder for one labelled clip")
    p.add_argument("set", help="clip set, e.g. iron, iron-inview, door, fridge, cat, iron-day2")
    p.add_argument("--label", required=True, choices=["pos", "neg"], help="chosen BEFORE recording")
    p.add_argument("--n", type=int, default=None, help="clip number (default: next free)")
    p.add_argument("--lighting", default="lamp", choices=["day", "lamp", "dusk"])
    p.add_argument("--camera-pos", default="board")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--countdown", type=int, default=3)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--out", default=os.environ.get("JAGA_RAW_CLIPS", str(Path.home() / ".jaga" / "raw-clips")))
    p.set_defaults(fn=cmd_record)

    p = sub.add_parser("label", help="log a label into the running session")
    p.add_argument("kind")
    p.add_argument("--note")
    p.add_argument("--session")
    p.set_defaults(fn=cmd_label)

    p = sub.add_parser("calibrate", help="dispatch the calibration Serverless Job (Nebius AI Cloud)")
    p.add_argument("--smoke", action="store_true",
                   help="plumbing run over the committed test fixture (not a calibration)")
    p.add_argument("--input", help="labelled margins CSV (default: built from fixtures + unscored sessions)")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="validate the Job request only (nothing created)")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("plug", help="plug status, or OFF")
    p.add_argument("what", choices=["status", "off"])
    p.set_defaults(fn=cmd_plug)

    p = sub.add_parser("info", help="versions and paths")
    p.set_defaults(fn=cmd_info)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
