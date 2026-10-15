"""Shared test helpers. No test downloads the model or calls Token Factory: vectors are scripted.

Scripted vectors exercise the daemon's machinery (decision rule, guards, actuator, server, re-arm), not the
model's judgement — the model's judgement is measured by scripts/bench.py on real labelled clips.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

from jaga.config import TUNING  # noqa: E402

DIM = 256


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def rand_unit(seed: int, n: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n is None:
        return unit(rng.normal(size=DIM))
    a = rng.normal(size=(n, DIM)).astype(np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


class Captions:
    """Three positive and three negative caption vectors, plus window vectors near either side."""

    def __init__(self, seed: int = 7):
        self.pos = rand_unit(seed, 3)
        self.neg = rand_unit(seed + 1, 3)
        self.rng = np.random.default_rng(seed + 2)

    def window(self, side: str, noise: float = 0.15) -> np.ndarray:
        base = self.pos[0] if side == "pos" else self.neg[0]
        return unit(base + noise * self.rng.normal(size=DIM).astype(np.float32) / np.sqrt(DIM) * 4)


class ScriptedEmbedder:
    """embed_video returns whatever side the test sets: 'pos' (the sentence is true) or 'neg'."""

    device = "cpu"

    def __init__(self, caps: Captions):
        self.caps = caps
        self.side = "neg"
        self.calls = 0

    def embed_video(self, frames):
        self.calls += 1
        return self.caps.window(self.side)

    def embed_text(self, texts):
        return rand_unit(abs(hash(tuple(texts))) % 2**31, len(texts))

    @property
    def info(self):
        return {"model": "scripted (test)", "device": "cpu", "dtype": "float32"}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("JAGA_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("JAGA_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("JAGA_PLUG_URL", raising=False)
    return tmp_path


@pytest.fixture
def tuning():
    return replace(TUNING, rearm_poll_s=0.1, action_retry_s=0.1, action_retries=3, watchdog_floor_s=0.5,
                   capture_s=10.0, suspend_s=5.0)


@pytest.fixture
def fakeplug():
    from fakeplug import FakePlug

    p = FakePlug(on=True)
    url = p.start()
    p.url = url
    yield p
    p.stop()


SPEC = {"visual": True, "reason": "", "name": "BOARD UNATTENDED",
        "positives": ["an empty ironing board", "an ironing board with nobody near it", "a laundry room, no people"],
        "hard_negatives": ["a person ironing a shirt", "a person standing at the ironing board",
                           "a person folding clothes at the board"],
        "dwell_s": 10, "action": "plug_off", "cooldown_s": 60, "language_in": "en"}


def frame(value: int = 140, h: int = 90, w: int = 160, seed: int = 0) -> np.ndarray:
    """A textured synthetic frame (so the camera-moved check has structure to correlate)."""
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), value, np.uint8)
    tex = (rng.random((h, w, 1)) * 40).astype(np.uint8)
    return np.clip(base.astype(np.int16) + tex - 20, 0, 255).astype(np.uint8)


def make_watcher(home, tuning, plug_url: str | None = None, caps: Captions | None = None, offline: bool = True,
                 action: str = "plug_off", dwell_s: float = 10, theta=None, session: str = "test"):
    from jaga.act import Notifier, NullPlug, ShellyPlug
    from jaga.capture import Ring
    from jaga.daemon import Watcher
    from jaga.db import DB
    from jaga.guards import Guards
    from jaga.score import Threshold

    caps = caps or Captions()
    emb = ScriptedEmbedder(caps)
    events = []
    driver = ShellyPlug(plug_url) if plug_url else NullPlug()
    w = Watcher(db=DB(home / "t.db"), embedder=emb, ring=Ring(), driver=driver,
                notifier=Notifier(os_banner=False, ntfy_topic=""), session=session, camera_pos="test", tuning=tuning,
                guards=Guards(tuning, hog=False), publish=lambda k, p: events.append((k, p)), offline=offline)
    spec = {**SPEC, "action": action, "dwell_s": dwell_s}
    th = theta or Threshold("labelled-m", 0.0, version=1, source="test")
    w.load_tripwire("01", "nobody is at the ironing board", spec, caps.pos, caps.neg, th, {"tf_request_id": "test"})
    w.published = events
    return w, emb


def push_windows(w, emb, side: str, n: int, t0: float, step: float = 1.0, value: int = 140):
    """Feed n windows through Watcher.process with media timestamps t0, t0+step, …"""
    from jaga.capture import Window

    emb.side = side
    out = []
    for i in range(n):
        ts = t0 + i * step
        frames = np.stack([frame(value, seed=k) for k in range(8)])
        out.append(w.process(Window(seq=w.windows + 1, ts=[ts - 7 + k for k in range(8)], frames=frames)))
    return out


def wait_for(cond, timeout: float = 5.0, every: float = 0.02) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(every)
    return cond()


def write_test_video(path: Path, seconds: float, fps: int = 10, size=(320, 180), value: int = 120) -> Path:
    """A small synthetic test video (tests only; never used as data)."""
    import cv2

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(0)
    for i in range(int(seconds * fps)):
        f = np.full((size[1], size[0], 3), value, np.uint8)
        f[:, (i * 3) % size[0]] = 255  # a moving bar
        f = np.clip(f.astype(np.int16) + rng.integers(-5, 5, f.shape), 0, 255).astype(np.uint8)
        vw.write(f)
    vw.release()
    return path
