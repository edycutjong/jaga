"""Frames: one canonical size for the live camera and for recorded files, so both take the same path.

Cosmos-Embed1's processor squashes any frame straight to 224×224 (bilinear, no crop). Resizing every frame
to the same canonical size first means a live window and a recorded clip of the same scene reach the model
through identical pixels. Files are read with PyAV (the card's sample uses decord, which has no macOS arm64
wheel: evidence/day0/decord-check.txt).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CAPTURE_FPS, WINDOW_FRAMES

CANON_LONG_SIDE = 640  # px; the long side of every frame before embedding and before the guards


def canon(frame: np.ndarray, *, bgr: bool = False) -> np.ndarray:
    """Return an RGB uint8 frame whose long side is CANON_LONG_SIDE (aspect kept)."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 frame, got {frame.shape}")
    if bgr:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    s = CANON_LONG_SIDE / max(h, w)
    if abs(s - 1.0) > 1e-6:
        frame = cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame, dtype=np.uint8)


def gray_small(frame_rgb: np.ndarray, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    """A small grayscale copy for differencing (float32, 0–255)."""
    g = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (3, 3), 0).astype(np.float32)


def luma(frame_rgb: np.ndarray) -> float:
    """Mean luma in [0, 1] (BT.601 weights via OpenCV's grayscale conversion)."""
    return float(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY).mean() / 255.0)


def to_jpeg(frame_rgb: np.ndarray, width: int | None = None, quality: int = 80) -> bytes:
    """JPEG for the local overlay only (served on 127.0.0.1; never stored, never uploaded)."""
    f = frame_rgb
    if width and f.shape[1] != width:
        h = round(f.shape[0] * width / f.shape[1])
        f = cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


@dataclass
class VideoInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    frames: int
    creation_time: str | None


def probe(path: str | Path) -> VideoInfo:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        dur = None
        if s.duration is not None and s.time_base is not None:
            dur = float(s.duration * s.time_base)
        elif c.duration is not None:
            dur = c.duration / 1_000_000
        fps = float(s.average_rate) if s.average_rate else 0.0
        meta = dict(c.metadata or {})
        meta.update(dict(s.metadata or {}))
        return VideoInfo(duration_s=float(dur or 0.0), fps=fps, width=s.codec_context.width,
                         height=s.codec_context.height, frames=int(s.frames or 0),
                         creation_time=meta.get("creation_time") or meta.get("com.apple.quicktime.creationdate"))


def iter_frames_at(path: str | Path, times_s: list[float]):
    """Yield (t, rgb_frame) for the decoded frame nearest each requested time (times ascending)."""
    import av

    want = sorted(float(t) for t in times_s)
    if not want:
        return
    i = 0
    prev = None  # (t, frame) of the last decoded frame
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for fr in c.decode(stream):
            if fr.pts is None:
                continue
            t = float(fr.pts * fr.time_base)
            while i < len(want) and t >= want[i]:
                # nearest of the previous frame and this one
                if prev is not None and abs(prev[0] - want[i]) < abs(t - want[i]):
                    yield want[i], prev[1].to_ndarray(format="rgb24")
                else:
                    yield want[i], fr.to_ndarray(format="rgb24")
                i += 1
            if i >= len(want):
                return
            prev = (t, fr)
        while i < len(want) and prev is not None:  # requested times past the last frame: use the last one
            yield want[i], prev[1].to_ndarray(format="rgb24")
            i += 1


def window_times(start_s: float, n: int = WINDOW_FRAMES, fps: float = CAPTURE_FPS) -> list[float]:
    """n frame times at `fps`, centred in each 1/fps slot: start+0.5, start+1.5, … at 1 fps."""
    step = 1.0 / fps
    return [start_s + step * (k + 0.5) for k in range(n)]


def clip_window_start(duration_s: float, where: str = "middle", seconds: float = WINDOW_FRAMES / CAPTURE_FPS) -> float:
    """Start of the 8-s window inside a recording. 'middle' avoids the record/stop handling at either end."""
    if duration_s <= seconds:
        return 0.0
    if where == "start":
        return 0.0
    if where == "end":
        return duration_s - seconds
    if where == "middle":
        return (duration_s - seconds) / 2.0
    raise ValueError(f"window must be start|middle|end, got {where!r}")


def read_window(path: str | Path, where: str = "middle", start_s: float | None = None) -> tuple[np.ndarray, dict]:
    """The clip's one window: 8 canonical RGB frames at 1 fps, plus where they came from."""
    info = probe(path)
    t0 = clip_window_start(info.duration_s, where) if start_s is None else float(start_s)
    times = window_times(t0)
    frames = [canon(f) for _, f in iter_frames_at(path, times)]
    if len(frames) != WINDOW_FRAMES:
        raise ValueError(f"{path}: got {len(frames)} frames for the window, need {WINDOW_FRAMES}")
    return np.stack(frames), {"start_s": round(t0, 3), "times_s": [round(t, 3) for t in times],
                              "duration_s": round(info.duration_s, 3), "source_fps": round(info.fps, 3)}


def iter_1fps(path: str | Path, start_s: float = 0.0, end_s: float | None = None, fps: float = CAPTURE_FPS):
    """Yield (t, canonical RGB frame) at `fps` across a long recording (normal captures, outside clips)."""
    info = probe(path)
    end = info.duration_s if end_s is None else min(end_s, info.duration_s)
    step = 1.0 / fps
    times = []
    t = start_s + step * 0.5
    while t < end:
        times.append(t)
        t += step
    for t, f in iter_frames_at(path, times):
        yield t, canon(f)


def window_batch(frames: np.ndarray) -> np.ndarray:
    """[T,H,W,3] uint8 → [1,T,3,H,W] (BTCHW), the layout the Cosmos-Embed1 processor expects."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3], got {frames.shape}")
    return np.ascontiguousarray(np.transpose(frames[None], (0, 1, 4, 2, 3)))
