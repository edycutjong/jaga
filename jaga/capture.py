"""Capture at 1 fps into an 8-frame ring buffer — the window Cosmos-Embed1 embeds.

The card says the model is optimised for 8 frames sampled at 1–2 fps. The camera is grabbed continuously
(so a retrieved frame is fresh, not one buffered seconds ago) and one frame is kept per second.
A recorded file can be played through the same ring (rehearsal, QA, a camera-less machine); the window
maths downstream is identical.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import CAPTURE_FPS, WINDOW_FRAMES
from .frames import canon, iter_1fps


@dataclass
class Window:
    seq: int
    ts: list[float]
    frames: np.ndarray  # [8,H,W,3] RGB uint8, canonical size

    @property
    def newest(self) -> np.ndarray:
        return self.frames[-1]

    @property
    def previous(self) -> np.ndarray:
        return self.frames[-2]


class Ring:
    def __init__(self, n: int = WINDOW_FRAMES):
        self.n = n
        self.buf: deque[tuple[float, np.ndarray]] = deque(maxlen=n)
        self.cond = threading.Condition()
        self.seq = 0
        self.consumed = 0
        self.closed = False

    def mark_consumed(self, seq: int) -> None:
        with self.cond:
            self.consumed = max(self.consumed, seq)
            self.cond.notify_all()

    def wait_consumed(self, seq: int, timeout: float = 60.0) -> bool:
        """Back-pressure for offline playback: wait until the consumer has embedded window `seq`."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while not self.closed and self.consumed < seq:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.cond.wait(left)
            return True

    def push(self, ts: float, frame: np.ndarray) -> None:
        with self.cond:
            self.buf.append((ts, frame))
            self.seq += 1
            self.cond.notify_all()

    def close(self) -> None:
        with self.cond:
            self.closed = True
            self.cond.notify_all()

    def wait_window(self, after_seq: int, timeout: float = 1.0) -> Window | None:
        """Block until the ring is full and holds at least one frame newer than `after_seq`."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while not self.closed and (len(self.buf) < self.n or self.seq <= after_seq):
                left = deadline - time.monotonic()
                if left <= 0:
                    return None
                self.cond.wait(left)
            if len(self.buf) < self.n or self.seq <= after_seq:
                return None
            ts = [t for t, _ in self.buf]
            frames = np.stack([f for _, f in self.buf])
            return Window(self.seq, ts, frames)


@dataclass
class SourceStatus:
    kind: str
    name: str
    ok: bool = False
    frames: int = 0
    error: str | None = None
    ended: bool = False
    extra: dict = field(default_factory=dict)


class CameraSource(threading.Thread):
    REOPEN_AFTER_S = 5.0  # no decoded frame for this long: release and reopen the device
    REOPEN_PAUSE_S = 2.0

    def __init__(self, ring: Ring, index: int = 0, fps: float = CAPTURE_FPS):
        super().__init__(daemon=True, name="jaga-camera")
        self.ring, self.index, self.fps = ring, index, fps
        self.status = SourceStatus("camera", f"camera {index}")
        self._halt = threading.Event()  # not `_stop`: Thread._stop is CPython's

    def stop(self) -> None:
        self._halt.set()

    def _open(self):
        import cv2

        backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
        cap = cv2.VideoCapture(self.index, backend)
        return cap if cap.isOpened() else None

    def run(self) -> None:
        import cv2

        cap = self._open()
        if cap is None:
            self.status.error = ("camera unavailable — on macOS, allow camera access for the terminal app "
                                 "(System Settings → Privacy & Security → Camera)")
            self.ring.close()
            return
        self.status.extra = {"width": cap.get(cv2.CAP_PROP_FRAME_WIDTH), "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
        period = 1.0 / self.fps
        next_t = time.monotonic()
        last_frame = time.monotonic()
        try:
            while not self._halt.is_set():
                if time.monotonic() - last_frame > self.REOPEN_AFTER_S:
                    # unplugged, asleep, or grabbing without decoding: release and reopen, so a night soak
                    # survives a USB hiccup (a time rule: a loaded laptop stretches every sleep)
                    self.status.ok, self.status.error = False, "camera stopped delivering frames — reopening"
                    cap.release()
                    self._halt.wait(self.REOPEN_PAUSE_S)
                    cap = self._open() or cap
                    self.status.extra["reopens"] = self.status.extra.get("reopens", 0) + 1
                    last_frame = time.monotonic()
                    continue
                if not cap.grab():
                    time.sleep(0.02)
                    continue
                now = time.monotonic()
                if now < next_t:
                    continue
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    time.sleep(0.02)
                    continue
                last_frame = now
                next_t += period
                if next_t < now:  # fell behind (laptop busy): resynchronise, never burst
                    next_t = now + period
                self.ring.push(time.time(), canon(frame, bgr=True))
                self.status.ok, self.status.error = True, None
                self.status.frames += 1
        finally:
            cap.release()


class FileSource(threading.Thread):
    """Plays a recording into the ring at 1 fps; `realtime` paces it like a camera."""

    def __init__(self, ring: Ring, path: str | Path, fps: float = CAPTURE_FPS, realtime: bool = True,
                 loop: bool = False, speed: float = 1.0):
        super().__init__(daemon=True, name="jaga-file")
        self.ring, self.path, self.fps = ring, Path(path), fps
        self.realtime, self.loop, self.speed = realtime, loop, speed
        self.status = SourceStatus("file", self.path.name)
        self._halt = threading.Event()  # not `_stop`: Thread._stop is CPython's

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        try:
            while not self._halt.is_set():
                start = time.monotonic()
                wall0 = time.time()
                first_t = None
                n = 0
                for t, frame in iter_1fps(self.path, fps=self.fps):
                    if self._halt.is_set():
                        break
                    if first_t is None:
                        first_t = t
                    if self.realtime:
                        wait = start + (t - first_t) / self.speed - time.monotonic()
                        if wait > 0:
                            time.sleep(wait)
                    elif self.ring.seq >= self.ring.n:  # offline: never overwrite a window not yet embedded
                        self.ring.wait_consumed(self.ring.seq)
                    ts = time.time() if self.realtime else wall0 + (t - first_t)  # offline: media time
                    self.ring.push(ts, frame)
                    self.status.ok = True
                    self.status.frames += 1
                    n += 1
                if not self.loop or n == 0:
                    break
        except Exception as e:  # unreadable file
            self.status.error = f"{type(e).__name__}: {e}"
        finally:
            self.status.ended = True
            if not self.loop:
                self.ring.close()
