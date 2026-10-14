"""Per-session window vectors: vectors-<session>.npz, keyed by embed.id. Local only, never uploaded.

Soaks keep no frames, so these vectors are what lets the bench re-score a session offline (ablations,
fresh mode) without a second live run.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

from . import config


def vectors_dir() -> Path:
    p = config.jaga_home() / "vectors"
    p.mkdir(parents=True, exist_ok=True)
    return p


def path_for(session: str) -> Path:
    return vectors_dir() / f"vectors-{session}.npz"


class VectorStore:
    def __init__(self, session: str, flush_every_s: float = 120.0):
        self.session = session
        self.path = path_for(session)
        self.flush_every_s = flush_every_s
        self._ids: list[int] = []
        self._vecs: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        if self.path.exists():  # resuming a session: keep what is on disk
            ids, vecs = load(session)
            self._ids, self._vecs = list(map(int, ids)), list(vecs)

    def add(self, embed_id: int, vec: np.ndarray) -> None:
        with self._lock:
            self._ids.append(int(embed_id))
            self._vecs.append(np.asarray(vec, dtype=np.float32).reshape(-1))
        if time.monotonic() - self._last_flush > self.flush_every_s:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            if not self._ids:
                return
            ids = np.asarray(self._ids, dtype=np.int64)
            vecs = np.stack(self._vecs).astype(np.float32)
            self._last_flush = time.monotonic()
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez(tmp, ids=ids, vecs=vecs)
        os.replace(tmp, self.path)

    def __len__(self) -> int:
        return len(self._ids)


def load(session: str) -> tuple[np.ndarray, np.ndarray]:
    p = path_for(session)
    with np.load(p) as z:
        return z["ids"].copy(), z["vecs"].copy()
