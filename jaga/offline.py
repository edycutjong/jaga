"""Offline scoring with the daemon's own maths: one 8-s clip = one window; a long recording = sliding windows.

`jaga score`, `scripts/seed.py` and `scripts/bench.py` all go through here, and here goes through
`jaga/score.py` — the same margin, z and k-of-n rule the live watcher runs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import TUNING, WINDOW_FRAMES, Tuning
from .frames import iter_1fps, read_window
from .score import margin, margins, normal_reference, replay_rule

REPO = Path(__file__).resolve().parents[1]
VECS = REPO / "fixtures" / "vecs"
CLIPS_CSV = REPO / "fixtures" / "clips.csv"
DECISIONS_FROZEN = REPO / "fixtures" / "decisions_frozen.csv"


def clip_id(path_or_id: str | Path) -> str:
    """'data/clips/iron/pos_03.mp4' → 'iron/pos_03' (the key of the committed vector)."""
    p = Path(str(path_or_id))
    parts = p.with_suffix("").parts
    if "clips" in parts:
        parts = parts[parts.index("clips") + 1:]
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]


def committed_vector(cid: str) -> np.ndarray | None:
    p = VECS / f"{cid}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def score_vector(vec: np.ndarray, pos: np.ndarray, neg: np.ndarray, theta_m: float | None) -> dict:
    m = margin(vec, pos, neg)
    return {"m": m, "theta_m": theta_m, "above": None if theta_m is None else bool(m >= theta_m)}


def published_decision(cid: str, slot: str) -> dict | None:
    if not DECISIONS_FROZEN.exists():
        return None
    with DECISIONS_FROZEN.open() as f:
        for row in csv.DictReader(f):
            if row.get("clip_id") == cid and row.get("tripwire") == slot:
                return row
    return None


def embed_clip(path: str | Path, embedder, where: str = "middle") -> tuple[np.ndarray, dict]:
    frames, meta = read_window(path, where)
    return embedder.embed_video(frames), meta


def sliding_windows(path: str | Path, stride_frames: int = 1, start_s: float = 0.0, end_s: float | None = None):
    """Yield (t of the newest frame, [8,H,W,3]) over a long recording at 1 fps, one window every `stride_frames`."""
    buf: list[np.ndarray] = []
    since = 0
    for t, f in iter_1fps(path, start_s, end_s):
        buf.append(f)
        if len(buf) > WINDOW_FRAMES:
            buf.pop(0)
        since += 1
        if len(buf) == WINDOW_FRAMES and since >= stride_frames:
            since = 0
            yield t, np.stack(buf)


def embed_long(path: str | Path, embedder, stride_frames: int = 1, start_s: float = 0.0,
               end_s: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    ts, vecs = [], []
    for t, w in sliding_windows(path, stride_frames, start_s, end_s):
        ts.append(t)
        vecs.append(embedder.embed_video(w))
    return np.asarray(ts, np.float64), (np.stack(vecs) if vecs else np.zeros((0, 256), np.float32))


def fresh_mode(normal_ts, normal_vecs, pos_ts, pos_vecs, pos: np.ndarray, neg: np.ndarray, *, dwell_s: float,
               stride_s: float, tuning: Tuning = TUNING, split_s: float = 60.0) -> dict:
    """The zero-training path, scored the way the outside-people metric is defined.

    The 2-min normal capture is split: windows in its first `split_s` seconds give μ₀/σ₀; the rest is a
    held-out false-positive test. "Tripped correctly first time" = a k-of-n trip at `dwell_s` inside the
    positive recording AND no trip on the held-out normal half.
    """
    n_ts = np.asarray(normal_ts, np.float64)
    n_m = margins(normal_vecs, pos, neg) if len(normal_vecs) else np.zeros(0)
    t0 = n_ts[0] if len(n_ts) else 0.0
    fit = n_m[n_ts - t0 < split_s]
    hold_ts = n_ts[n_ts - t0 >= split_s]
    hold_m = n_m[n_ts - t0 >= split_s]
    if fit.size < 2:
        return {"ok": False, "detail": f"only {fit.size} normal windows before {split_s:.0f} s"}
    mu0, sigma0 = normal_reference(fit, tuning.sigma0_floor)
    z_hold = (hold_m - mu0) / sigma0
    p_m = margins(pos_vecs, pos, neg) if len(pos_vecs) else np.zeros(0)
    z_pos = (p_m - mu0) / sigma0
    theta = tuning.theta_z
    trips_hold = replay_rule(list(hold_ts), list(z_hold >= theta), dwell_s, 600.0, stride_s,
                             k_fraction=tuning.k_fraction)
    pos_ts = np.asarray(pos_ts, np.float64)
    trips_pos = replay_rule(list(pos_ts), list(z_pos >= theta), dwell_s, 600.0, stride_s, k_fraction=tuning.k_fraction)
    return {"ok": True, "mu0": mu0, "sigma0": sigma0, "theta_z": theta, "n_fit": int(fit.size),
            "n_holdout": int(hold_m.size), "n_pos": int(p_m.size), "holdout_trips": len(trips_hold),
            "pos_tripped": bool(trips_pos), "time_to_trip_s": (trips_pos[0] - pos_ts[0]) if trips_pos else None,
            "tripped_correctly": bool(trips_pos) and not trips_hold,
            "z_pos_max": float(z_pos.max()) if z_pos.size else None,
            "z_hold_max": float(z_hold.max()) if z_hold.size else None}
