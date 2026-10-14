"""The deterministic trigger: margin → z → k-of-n over the dwell → act. No model runs here (W4).

The model chooses where to look; this code decides what is true.

* margin   m_t = max_i cos(v_t, p_i) − max_j cos(v_t, n_j) over the compiled positives p and hard negatives n
* fresh    z_t = (m_t − μ₀) / max(σ₀, floor) ≥ θ_z, with μ₀, σ₀ from a 2-min "normal state" capture at arming
* labelled m_t ≥ θ_m, θ_m = the highest margin that keeps clip recall ≥ 0.9 (z is display only)
* trip     at least k of the n windows inside the dwell are above threshold, and the newest one is;
           n = dwell ÷ stride, k = ⌈0.8·n⌉ (P7). Dwell ≥ 2 × the p95 window period (I2).

`DecisionRule` is shared verbatim by the live tripwire and by the bench's baselines, so every system is
judged by the same dwell, k-of-n, suspend and cool-down.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import TUNING, Tuning
from .schema import clamp_dwell


# --- maths ---------------------------------------------------------------------------------------
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def margin(v: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    """max cosine to the positives minus max cosine to the hard negatives (vectors are L2-normalised)."""
    v = unit(v).reshape(-1)
    p = unit(np.atleast_2d(pos)) @ v
    if neg is None or len(neg) == 0:
        return float(p.max())
    n = unit(np.atleast_2d(neg)) @ v
    return float(p.max() - n.max())


def margins(V: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    V = unit(np.atleast_2d(V))
    p = V @ unit(np.atleast_2d(pos)).T
    if neg is None or len(neg) == 0:
        return p.max(axis=1)
    n = V @ unit(np.atleast_2d(neg)).T
    return p.max(axis=1) - n.max(axis=1)


def zscore(m: float, mu0: float | None, sigma0: float | None, floor: float = TUNING.sigma0_floor) -> float | None:
    if mu0 is None or sigma0 is None:
        return None
    return float((m - mu0) / max(sigma0, floor))


def normal_reference(ms: list[float] | np.ndarray, floor: float = TUNING.sigma0_floor) -> tuple[float, float]:
    """μ₀, σ₀ over the normal-state windows (the non-event), with the σ₀ floor P5 applied."""
    a = np.asarray(ms, dtype=np.float64)
    if a.size == 0:
        raise ValueError("no normal-state windows")
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return float(a.mean()), max(sd, floor)


def theta_at_recall(pos_margins: list[float] | np.ndarray, recall: float = 0.9) -> float:
    """The highest θ_m that keeps recall ≥ `recall` on the positives (ties go to the larger θ)."""
    p = np.sort(np.asarray(pos_margins, dtype=np.float64))[::-1]
    if p.size == 0:
        raise ValueError("no positive margins")
    need = math.ceil(recall * p.size - 1e-9)
    return float(p[max(need, 1) - 1])


def percentile(values, q: float) -> float | None:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return None
    return float(np.percentile(a, q * 100))


def window_period(ts: list[float] | np.ndarray) -> tuple[float | None, float | None]:
    """p50 / p95 of diff(embed.ts): the live window period N that every clamp and the watchdog use."""
    t = np.asarray(list(ts), dtype=np.float64)
    if t.size < 2:
        return None, None
    d = np.diff(t)
    d = d[d > 0]
    if d.size == 0:
        return None, None
    return float(np.percentile(d, 50)), float(np.percentile(d, 95))


# --- the shared decision rule ----------------------------------------------------------------------
@dataclass
class Decision:
    fire: bool
    count: int
    k: int
    n: int
    newest_above: bool

    @property
    def progress(self) -> float:
        return 0.0 if self.k <= 0 else min(1.0, self.count / self.k)


class DecisionRule:
    """k-of-n over the dwell, with suspend and cool-down. Used by jaga and by every baseline."""

    def __init__(self, dwell_s: float, cooldown_s: float, k_fraction: float = TUNING.k_fraction):
        if dwell_s <= 0:
            raise ValueError("dwell must be positive")
        self.dwell_s = float(dwell_s)
        self.cooldown_s = float(cooldown_s)
        self.k_fraction = float(k_fraction)
        self.history: deque[tuple[float, bool]] = deque()
        self.suspended_until = -math.inf
        self.cooldown_until = -math.inf

    def kn(self, stride_s: float) -> tuple[int, int]:
        n = max(1, int(math.floor(self.dwell_s / max(stride_s, 1e-6) + 1e-9)))
        k = max(1, int(math.ceil(self.k_fraction * n - 1e-9)))
        return k, n

    def suspend(self, ts: float, seconds: float) -> None:
        """A sudden scene change: no trip for `seconds`, and the dwell restarts afterwards."""
        self.suspended_until = max(self.suspended_until, ts + seconds)
        self.history.clear()

    def reset(self) -> None:
        self.history.clear()
        self.cooldown_until = -math.inf

    def start_cooldown(self, ts: float) -> None:
        self.cooldown_until = ts + self.cooldown_s
        self.history.clear()

    def push(self, ts: float, above: bool, stride_s: float) -> Decision:
        if ts < self.suspended_until:
            k, n = self.kn(stride_s)
            return Decision(False, 0, k, n, False)
        self.history.append((ts, bool(above)))
        while self.history and self.history[0][0] <= ts - self.dwell_s:
            self.history.popleft()
        k, n = self.kn(stride_s)
        count = sum(1 for _, a in self.history if a)
        newest = bool(above)
        fire = newest and count >= k and ts >= self.cooldown_until
        return Decision(fire, count, k, n, newest)


def replay_rule(ts: list[float], flags: list[bool], dwell_s: float, cooldown_s: float, stride_s: float,
                suspend_at: list[float] | None = None, suspend_s: float = TUNING.suspend_s,
                k_fraction: float = TUNING.k_fraction, pause_after_trip: bool = False,
                resume_at: list[float] | None = None) -> list[float]:
    """Run per-window flags through the shared rule; return trip times.

    `pause_after_trip` models a plug_off trip: power stays off until an operator switches it back on, so
    windows after a trip are ignored until the next time in `resume_at` (the logged operator ON); the
    rule then restarts from an empty history. With no later resume the replay ends at the trip.
    """
    rule = DecisionRule(dwell_s, cooldown_s, k_fraction)
    sus = sorted(suspend_at or [])
    res = sorted(resume_at or [])
    si = 0
    trips: list[float] = []
    paused_until: float | None = None
    for t, f in zip(ts, flags, strict=True):
        while si < len(sus) and sus[si] <= t:
            rule.suspend(sus[si], suspend_s)
            si += 1
        if paused_until is not None:
            if t < paused_until:
                continue
            paused_until = None
            rule.reset()
        d = rule.push(t, f, stride_s)
        if d.fire:
            trips.append(t)
            if pause_after_trip:
                later = [r for r in res if r > t]
                if not later:
                    break
                paused_until = later[0]
            else:
                rule.start_cooldown(t)
    return trips


# --- the live tripwire --------------------------------------------------------------------------
@dataclass
class Threshold:
    mode: str  # 'labelled-m' | 'fresh-z'
    theta: float | None
    mu0: float | None = None
    sigma0: float | None = None
    version: int | None = None
    source: str = "default"
    frozen_for: str | None = None

    def above(self, m: float, z: float | None) -> bool | None:
        if self.theta is None:
            return None
        if self.mode == "labelled-m":
            return m >= self.theta
        return z is not None and z >= self.theta


@dataclass
class WindowResult:
    slot: str
    m: float
    z: float | None
    above: bool | None
    fired: bool
    state: str
    progress: float
    count: int
    k: int
    n: int
    dwell_s: float


@dataclass
class Tripwire:
    """One armed sentence. States: capturing → armed ⇄ holding → tripped → (rearmed) armed; or unarmed."""

    slot: str
    name: str
    sentence: str
    action: str  # 'plug_off' | 'notify'
    dwell_s: float
    cooldown_s: float
    pos: np.ndarray
    neg: np.ndarray
    threshold: Threshold
    tripwire_id: int | None = None
    tuning: Tuning = field(default_factory=lambda: TUNING)
    state: str = "unarmed"
    unarmed_reason: str | None = "not armed"
    tripped_at: float | None = None
    capture_until: float | None = None
    capture_s: float = 120.0
    capture_margins: list = field(default_factory=list)
    last: WindowResult | None = None

    def __post_init__(self):
        self.rule = DecisionRule(self.dwell_s, self.cooldown_s, self.tuning.k_fraction)

    # lifecycle ---------------------------------------------------------------------------------
    def start_capture(self, ts: float, seconds: float) -> None:
        """The capture clock starts at the first captured window, so a ring still filling never shortens it."""
        self.state, self.unarmed_reason = "capturing", None
        self.capture_s = float(seconds)
        self.capture_until = None
        self.capture_margins = []

    def arm(self) -> None:
        if self.threshold.theta is None:
            raise ValueError("no threshold: capture a normal state (fresh) or freeze θ (labelled) first")
        self.state, self.unarmed_reason = "armed", None
        self.rule.reset()

    def unarm(self, reason: str) -> None:
        self.state, self.unarmed_reason = "unarmed", reason
        self.rule.reset()

    def rearm(self) -> None:
        """The plug came back ON (its own button or the operator test): re-arm with the stored reference.

        Ends a plug_off cool-down so someone who resumes ironing is guarded again. Never switches anything.
        """
        self.tripped_at = None
        self.state, self.unarmed_reason = "armed", None
        self.rule.reset()

    def noise_floor_ok(self, example_margins: list[float]) -> bool:
        """The optional positive examples must clear the normal state by z ≥ 1 on average."""
        zs = [zscore(m, self.threshold.mu0, self.threshold.sigma0, self.tuning.sigma0_floor) for m in example_margins]
        zs = [z for z in zs if z is not None]
        return bool(zs) and float(np.mean(zs)) >= 1.0

    def effective_dwell(self, period_p95: float | None) -> float:
        return clamp_dwell(self.dwell_s, period_p95)

    # one window ----------------------------------------------------------------------------------
    def step(self, ts: float, v: np.ndarray, *, stride_s: float, period_p95: float | None) -> WindowResult:
        m = margin(v, self.pos, self.neg)
        th = self.threshold
        z = zscore(m, th.mu0, th.sigma0, self.tuning.sigma0_floor)
        dwell = self.effective_dwell(period_p95)
        if dwell != self.rule.dwell_s:
            self.rule.dwell_s = dwell
        fired = False
        above = None
        if self.state == "capturing":
            if self.capture_until is None:
                self.capture_until = ts + self.capture_s
            self.capture_margins.append(m)
            if ts >= self.capture_until:
                self._finish_capture()
            d = Decision(False, 0, *self.rule.kn(stride_s), False)
        elif self.state in ("armed", "holding"):
            above = th.above(m, z)
            d = self.rule.push(ts, bool(above), stride_s)
            if d.fire:
                fired = True
                self.state = "tripped"
                self.tripped_at = ts
                self.rule.start_cooldown(ts)
            else:
                self.state = "holding" if d.newest_above and d.count > 0 else "armed"
        elif self.state == "tripped":
            above = th.above(m, z)
            d = Decision(False, 0, *self.rule.kn(stride_s), bool(above))
            if self.action == "notify" and ts >= self.rule.cooldown_until:
                self.state = "armed"
                self.rule.history.clear()
        else:
            d = Decision(False, 0, *self.rule.kn(stride_s), False)
        self.last = WindowResult(self.slot, m, z, above, fired, self.state, d.progress if self.state != "tripped"
                                 else 1.0, d.count, d.k, d.n, dwell)
        return self.last

    def suspend(self, ts: float, seconds: float) -> None:
        if self.state in ("armed", "holding"):
            self.rule.suspend(ts, seconds)
            self.state = "armed"

    def _finish_capture(self) -> None:
        mu0, sigma0 = normal_reference(self.capture_margins, self.tuning.sigma0_floor)
        self.threshold.mu0, self.threshold.sigma0 = mu0, sigma0
        if self.threshold.mode == "fresh-z":
            if self.threshold.theta is None:
                self.threshold.theta = self.tuning.theta_z
            self.threshold.source = "normal-state"
        self.capture_until = None
        self.arm()
