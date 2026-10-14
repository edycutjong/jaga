"""Per-window environment guards and the live baseline inputs. None of these can trip a tripwire.

Guards (they suspend or unarm jaga):
* luma jump   — mean luma changes by more than P1 between windows → suspend trips for P2 s
* too dark    — mean luma below P3 → "too dark to see", unarmed until the light returns
* camera moved— phase correlation of the top 15 % strip (full width) against the arming reference:
                a shift above 4 % of the width or a response below P4, for 3 windows in a row → unarmed,
                "re-arm". Measured on day 0: the 10 % side strips are too narrow to correlate a pan
                reliably, while a person-sized blob in the centre leaves the top strip untouched. If the
                top strip has no texture at arming (a blank wall), the whole frame is used instead.

Baseline inputs (logged per window for bench.py, never used by a trip; soaks keep no frames, so they
must be computed live):
* motion_pct  — % of pixels that changed between the two newest frames (1 s apart): frame difference
* still_s     — seconds since motion_pct last reached P8: the stillness timer's input
* person_in_zone — OpenCV's built-in HOG pedestrian detector on the newest frame, inside the zone box
                   for this camera position (no model download)
* ref_diff_pct — % of pixels that differ from the median frame of the arming capture. The reference
                 frame lives in memory only and is never stored.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import TUNING, Tuning
from .frames import gray_small, luma

PHASE_SIZE = (256, 144)  # downsample for the camera-moved check
TOP_STRIP = 0.15  # the top 15 % of the frame: wall or ceiling, where a person rarely stands
MIN_STRIP_STD = 4.0  # grey levels; below this the top strip is too flat to correlate


def _hann(shape_hw: tuple[int, int]) -> np.ndarray:
    return cv2.createHanningWindow((shape_hw[1], shape_hw[0]), cv2.CV_32F)


@dataclass
class GuardResult:
    luma: float
    luma_jump: bool
    too_dark: bool
    motion_pct: float | None
    still_s: float
    person_in_zone: int | None
    ref_diff_pct: float | None
    camera_moved: bool
    phase_shift_frac: float | None
    phase_response: float | None

    def as_row(self) -> dict:
        return {"luma": self.luma, "motion_pct": self.motion_pct, "still_s": self.still_s,
                "person_in_zone": self.person_in_zone, "ref_diff_pct": self.ref_diff_pct}


def changed_pct(a: np.ndarray, b: np.ndarray, delta: int) -> float:
    """% of pixels whose grey level differs by more than `delta` (both small float32 greys)."""
    return float((np.abs(a - b) > delta).mean() * 100.0)


class Guards:
    def __init__(self, tuning: Tuning = TUNING, zone: tuple[float, float, float, float] | None = None,
                 hog: bool = True):
        self.t = tuning
        self.zone = zone or (0.0, 0.0, 1.0, 1.0)
        self.prev_luma: float | None = None
        self.last_motion_ts: float | None = None
        self.first_ts: float | None = None
        self.ref_gray: np.ndarray | None = None  # median of the arming capture, small grey (memory only)
        self.ref_phase: np.ndarray | None = None  # camera-moved reference (top strip, or whole frame)
        self.phase_region = "top-strip"
        self.moved_streak = 0
        self._hog = None
        if hog:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # references -------------------------------------------------------------------------------
    def begin_reference(self) -> None:
        """Start collecting the arming capture (small greys only; no frame is kept)."""
        self._acc_grey: list[np.ndarray] = []
        self._acc_phase: list[np.ndarray] = []

    def add_reference(self, frame_rgb: np.ndarray) -> None:
        if not hasattr(self, "_acc_grey"):
            self.begin_reference()
        self._acc_grey.append(gray_small(frame_rgb))
        self._acc_phase.append(self._phase_img(frame_rgb))

    def end_reference(self) -> bool:
        """Median of the capture → difference reference and camera-moved reference."""
        greys, phases = getattr(self, "_acc_grey", []), getattr(self, "_acc_phase", [])
        self._acc_grey, self._acc_phase = [], []
        if not greys:
            return False
        self.ref_gray = np.median(np.stack(greys), axis=0).astype(np.float32)
        phase = np.median(np.stack(phases), axis=0).astype(np.float32)
        top = phase[: self._top_rows(), :]
        self.phase_region = "top-strip" if float(top.std()) >= MIN_STRIP_STD else "full-frame"
        self.ref_phase = self._region(phase)
        self.moved_streak = 0
        return True

    def set_reference(self, frames: list[np.ndarray]) -> None:
        self.begin_reference()
        for f in frames:
            self.add_reference(f)
        self.end_reference()

    @property
    def has_reference(self) -> bool:
        return self.ref_phase is not None

    def clear_reference(self) -> None:
        self.ref_gray = None
        self.ref_phase = None
        self.moved_streak = 0

    @staticmethod
    def _phase_img(frame_rgb: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, PHASE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)

    # checks -----------------------------------------------------------------------------------
    @staticmethod
    def _top_rows() -> int:
        return max(8, round(PHASE_SIZE[1] * TOP_STRIP))

    def _region(self, img: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(img[: self._top_rows(), :] if self.phase_region == "top-strip" else img)

    def camera_shift(self, frame_rgb: np.ndarray) -> tuple[float, float] | None:
        """(shift as a fraction of the frame width, phase-correlation response) against the reference."""
        if self.ref_phase is None:
            return None
        cur = self._region(self._phase_img(frame_rgb))
        (dx, dy), resp = cv2.phaseCorrelate(self.ref_phase, cur, _hann(cur.shape))
        return float(np.hypot(dx, dy) / PHASE_SIZE[0]), float(resp)

    def person_in_zone(self, frame_rgb: np.ndarray) -> int | None:
        if self._hog is None:
            return None
        g = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        rects, weights = self._hog.detectMultiScale(g, winStride=(8, 8), padding=(8, 8), scale=1.05)
        h, w = g.shape
        x0, y0, x1, y1 = self.zone
        for (x, y, rw, rh), wt in zip(rects, np.ravel(weights) if len(rects) else [], strict=True):
            if float(wt) < self.t.hog_min_weight:
                continue
            cx, cy = (x + rw / 2) / w, (y + rh / 2) / h
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return 1
        return 0

    def update(self, ts: float, newest: np.ndarray, previous: np.ndarray | None) -> GuardResult:
        if self.first_ts is None:
            self.first_ts = ts
        lum = luma(newest)
        jump = False
        if self.prev_luma is not None:
            jump = abs(lum - self.prev_luma) / max(self.prev_luma, 0.02) > self.t.luma_jump
        self.prev_luma = lum
        too_dark = lum < self.t.dark_floor

        g_new = gray_small(newest)
        motion = None
        if previous is not None:
            motion = changed_pct(g_new, gray_small(previous), self.t.motion_pixel_delta)
            if motion >= self.t.still_motion_pct:
                self.last_motion_ts = ts
        still = ts - (self.last_motion_ts if self.last_motion_ts is not None else self.first_ts)

        ref_diff = None
        if self.ref_gray is not None:
            ref_diff = changed_pct(g_new, self.ref_gray, self.t.motion_pixel_delta)

        moved = False
        shift = resp = None
        cs = self.camera_shift(newest)
        if cs is not None:
            shift, resp = cs
            bad = shift > self.t.camera_shift_frac or resp < self.t.phase_response_min
            self.moved_streak = self.moved_streak + 1 if bad else 0
            moved = self.moved_streak >= self.t.camera_moved_windows

        return GuardResult(luma=round(lum, 4), luma_jump=jump, too_dark=too_dark,
                           motion_pct=None if motion is None else round(motion, 3), still_s=round(still, 2),
                           person_in_zone=self.person_in_zone(newest), ref_diff_pct=None if ref_diff is None
                           else round(ref_diff, 3), camera_moved=moved,
                           phase_shift_frac=None if shift is None else round(shift, 4),
                           phase_response=None if resp is None else round(resp, 4))
