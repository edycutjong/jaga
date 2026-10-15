#!/usr/bin/env python3
"""With the internet cut off, an armed tripwire still trips and the plug still switches OFF.

Every socket connection to a non-LAN address is refused inside this process (loopback and private LAN
ranges are allowed: the plug lives there). Then a tripwire is armed and fed windows until it trips.

    python scripts/verify_offline.py            CI variant: committed window vectors when they exist (else the
                                                frozen caption vectors as stand-in windows), and the fake LAN plug
                                                from tests/fakeplug.py. No model, no camera, no key.
    python scripts/verify_offline.py --real --source <clip.mp4>
                                                the cached Cosmos-Embed1 on this laptop and the real plug from
                                                ~/.config/jaga; the recording starts with ≥ 20 s of the normal
                                                state, then the event (run once on camera for the record)
Exit 0 = tripped, OFF acknowledged, and no outbound non-LAN connection succeeded.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

BLOCKED: list[str] = []
_real_connect = socket.socket.connect
_real_create = socket.create_connection


def _lan(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host in ("localhost",)
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _guard_connect(self, address):
    host = address[0] if isinstance(address, tuple) else str(address)
    if isinstance(address, tuple) and not _lan(host):
        BLOCKED.append(host)
        raise OSError(f"verify_offline: outbound connection to {host} blocked")
    return _real_connect(self, address)


def _guard_create(address, *a, **kw):
    host = address[0]
    try:
        infos = socket.getaddrinfo(host, address[1])
        addrs = {i[4][0] for i in infos}
    except OSError:
        addrs = {host}
    if not all(_lan(x) for x in addrs):
        BLOCKED.append(host)
        raise OSError(f"verify_offline: outbound connection to {host} blocked")
    return _real_create(address, *a, **kw)


def block_egress() -> None:
    socket.socket.connect = _guard_connect
    socket.create_connection = _guard_create
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def probe_outbound() -> bool:
    """True if an outbound connection to a public address succeeds from this process."""
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=3).close()
        return True
    except OSError:
        return False


def run_ci(tmp: Path) -> dict:
    import numpy as np
    from fakeplug import FakePlug

    from jaga.act import Notifier, ShellyPlug
    from jaga.capture import Ring, Window
    from jaga.config import TUNING
    from jaga.daemon import Watcher
    from jaga.db import DB
    from jaga.guards import Guards
    from jaga.offline import committed_vector
    from jaga.score import Threshold
    from jaga.tripwires import load, load_vectors

    fx = load("01")
    pos, neg = load_vectors(fx)
    vec_pos, vec_neg, source = pos[0], neg[0], "frozen caption vectors as stand-in windows (no clips committed yet)"
    clips = REPO / "fixtures" / "clips.csv"
    if clips.exists():
        import csv

        rows = list(csv.DictReader(clips.open()))
        p = next((committed_vector(r["clip_id"]) for r in rows if r["tripwire"] == "01" and r["label"] == "1"
                  and committed_vector(r["clip_id"]) is not None), None)
        n = next((committed_vector(r["clip_id"]) for r in rows if r["tripwire"] == "01" and r["label"] == "0"
                  and committed_vector(r["clip_id"]) is not None), None)
        if p is not None and n is not None:
            vec_pos, vec_neg, source = p, n, "committed window vectors of labelled clips"

    class Replay:
        device = "replay"
        side = "neg"

        def embed_video(self, frames):
            return vec_pos if self.side == "pos" else vec_neg

        def embed_text(self, texts):
            raise RuntimeError("not used")

    plug = FakePlug(on=True)
    url = plug.start()
    os.environ["JAGA_HOME"] = str(tmp)
    t = replace(TUNING, rearm_poll_s=0.2)
    emb = Replay()
    w = Watcher(db=DB(tmp / "offline.db"), embedder=emb, ring=Ring(), driver=ShellyPlug(url),
                notifier=Notifier(os_banner=False, ntfy_topic=""), session="verify-offline", tuning=t,
                guards=Guards(t, hog=False), offline=True)
    m_pos = float(np.max(pos @ vec_pos) - np.max(neg @ vec_pos))
    m_neg = float(np.max(pos @ vec_neg) - np.max(neg @ vec_neg))
    theta = (m_pos + m_neg) / 2
    w.load_tripwire("01", fx["sentence"], fx["spec"], pos, neg,
                    Threshold("labelled-m", theta, version=0, source="verify-offline"), {"operating": fx["operating"]})
    w.override_dwell("01", 10, "verify")
    w.start()
    try:
        w.arm("01", 3)  # a short normal-state capture, as a user's arming does (it also sets the camera reference)
        rng = np.random.default_rng(0)
        frames = np.clip(140 + rng.integers(-20, 20, (8, 90, 160, 3)), 0, 255).astype(np.uint8)
        ts = 1000.0
        for side, n in (("neg", 6), ("pos", 12)):
            emb.side = side
            for _ in range(n):
                w.process(Window(seq=w.windows + 1, ts=[ts - 7 + k for k in range(8)], frames=frames))
                ts += 1.0
        end = time.monotonic() + 5
        while time.monotonic() < end and w.plug["state"] != "off":
            time.sleep(0.05)
        tripped = any(e["kind"] == "trip" for e in w.db.recent_events(20, "verify-offline"))
        return {"variant": "ci", "window_source": source, "tripped": tripped, "plug_off_acknowledged":
                w.plug["state"] == "off", "fake_plug_off_calls": plug.off_calls, "fake_plug_on_calls": plug.on_calls}
    finally:
        w.stop()
        plug.stop()


def run_real(source: str, tmp: Path) -> dict:
    from jaga.act import Notifier, load_driver
    from jaga.capture import FileSource, Ring
    from jaga.daemon import Watcher
    from jaga.db import DB
    from jaga.embed import Embedder
    from jaga.tripwires import frozen_threshold, load, load_vectors

    fx = load("01")
    th = frozen_threshold(fx)
    if th is None:
        raise SystemExit("--real needs the frozen θ v1 for tripwire 01 (scripts/seed.py freeze-theta)")
    pos, neg = load_vectors(fx)
    os.environ["JAGA_HOME"] = str(tmp)
    ring = Ring()
    src = FileSource(ring, source, realtime=False)
    driver = load_driver()
    w = Watcher(db=DB(tmp / "offline.db"), embedder=Embedder("auto"), ring=ring, driver=driver,
                notifier=Notifier(os_banner=False, ntfy_topic=""), session="verify-offline-real", source=src,
                offline=True)
    w.load_tripwire("01", fx["sentence"], fx["spec"], pos, neg, th, {"operating": fx["operating"]})
    w.override_dwell("01", 30, "verify")
    w.start()
    w.arm("01", 20)  # the recording must start with ≥ 20 s of the normal state (someone ironing), then the event
    src.start()
    src.join()
    end = time.monotonic() + 10
    while time.monotonic() < end and w.plug["state"] != "off":
        time.sleep(0.1)
    w.stop()
    return {"variant": "real", "source": source, "driver": driver.describe(), "plug_off_acknowledged":
            w.plug["state"] == "off", "tripped": any(e["kind"] == "trip" for e in w.db.recent_events(50,
                                                                                                  "verify-offline-real"))}


def main() -> int:
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--source", help="--real: a recorded clip that contains the event")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    online = probe_outbound()  # the machine's own network, before the block (informational)
    block_egress()
    leaked = probe_outbound()  # must be False: the block holds
    with tempfile.TemporaryDirectory() as tmp:
        res = run_real(a.source, Path(tmp)) if a.real else run_ci(Path(tmp))
    res.update({"machine_online_before_block": online, "outbound_after_block": leaked,
                "blocked_attempts": sorted(set(BLOCKED))})
    ok = res["tripped"] and res["plug_off_acknowledged"] and not leaked and res.get("fake_plug_on_calls", 0) == 0
    res["pass"] = bool(ok)
    print(json.dumps(res, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
