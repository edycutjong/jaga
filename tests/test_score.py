"""The deterministic trigger (jaga/score.py): margin, z, θ at recall, k-of-n, suspend, cool-down."""
import math

import numpy as np
import pytest
from conftest import Captions

from jaga.schema import clamp_dwell
from jaga.score import (
    DecisionRule,
    Threshold,
    Tripwire,
    margin,
    margins,
    normal_reference,
    replay_rule,
    theta_at_recall,
    window_period,
    zscore,
)


def test_margin_is_best_positive_minus_best_negative():
    caps = Captions(1)
    v = caps.window("pos", noise=0.0)
    expect = float(np.max(caps.pos @ v) - np.max(caps.neg @ v))
    assert margin(v, caps.pos, caps.neg) == pytest.approx(expect, abs=1e-6)
    assert margin(caps.window("pos"), caps.pos, caps.neg) > 0 > margin(caps.window("neg"), caps.pos, caps.neg)


def test_margins_vectorised_matches_scalar():
    caps = Captions(2)
    V = np.stack([caps.window("pos"), caps.window("neg"), caps.window("pos")])
    assert np.allclose(margins(V, caps.pos, caps.neg), [margin(v, caps.pos, caps.neg) for v in V], atol=1e-6)


def test_positives_only_margin_when_no_negatives():
    caps = Captions(3)
    v = caps.window("pos")
    assert margin(v, caps.pos, np.zeros((0, 256))) == pytest.approx(float(np.max(caps.pos @ v)), abs=1e-6)


@pytest.mark.parametrize("kind", ["absence", "presence"])
def test_normal_reference_polarity_for_absence_and_presence(kind):
    """μ₀/σ₀ come from the NON-event: someone at the board (absence tripwire) or the closed door in use (presence).

    Either way the event moves the margin up from the normal state, so z > 0 means 'the sentence is true'."""
    caps = Captions(4)
    normal = [margin(caps.window("neg"), caps.pos, caps.neg) for _ in range(60)]
    event = [margin(caps.window("pos"), caps.pos, caps.neg) for _ in range(20)]
    mu0, sigma0 = normal_reference(normal)
    z_event = [zscore(m, mu0, sigma0) for m in event]
    z_normal = [zscore(m, mu0, sigma0) for m in normal]
    assert min(z_event) > 3.0 and abs(np.mean(z_normal)) < 0.5, kind


def test_sigma0_floor_applies():
    mu0, sigma0 = normal_reference([0.1, 0.1, 0.1], floor=0.01)
    assert mu0 == pytest.approx(0.1) and sigma0 == 0.01


def test_theta_at_recall_is_highest_threshold_keeping_recall():
    pos = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    th = theta_at_recall(pos, 0.9)
    assert th == pytest.approx(0.1)
    assert sum(p >= th for p in pos) / len(pos) >= 0.9
    assert sum(p >= th + 1e-9 for p in pos) / len(pos) < 0.9


def test_k_of_n_fires_at_k_not_before():
    rule = DecisionRule(dwell_s=10, cooldown_s=60, k_fraction=0.8)
    k, n = rule.kn(stride_s=1.0)
    assert (k, n) == (8, 10)
    fired_at = None
    for i in range(12):
        d = rule.push(float(i), True, 1.0)
        if d.fire and fired_at is None:
            fired_at = i
    assert fired_at == k - 1  # the 8th consecutive above-θ window


def test_newest_window_must_be_above():
    rule = DecisionRule(dwell_s=10, cooldown_s=60, k_fraction=0.8)
    for i in range(9):
        rule.push(float(i), True, 1.0)
    d = rule.push(9.0, False, 1.0)  # 9 of 10 above, but the newest is not
    assert d.count >= d.k and not d.fire


def test_single_bad_window_cannot_trip():
    """I2: one window above θ never trips (k ≥ 2 whenever the dwell spans ≥ 2 windows)."""
    rule = DecisionRule(dwell_s=10, cooldown_s=60, k_fraction=0.8)
    for i in range(30):
        assert not rule.push(float(i), i == 15, 1.0).fire


def test_cooldown_blocks_a_second_trip():
    rule = DecisionRule(dwell_s=5, cooldown_s=30, k_fraction=0.8)
    trips = []
    for i in range(60):
        d = rule.push(float(i), True, 1.0)
        if d.fire:
            trips.append(i)
            rule.start_cooldown(float(i))
    assert trips[0] == 3 and all(b - a >= 30 for a, b in zip(trips, trips[1:], strict=False))


def test_suspend_clears_history_and_blocks_trips():
    rule = DecisionRule(dwell_s=10, cooldown_s=60)
    for i in range(7):
        rule.push(float(i), True, 1.0)
    rule.suspend(7.0, 5.0)
    assert not any(rule.push(float(i), True, 1.0).fire for i in range(7, 12))  # suspended
    fired = [i for i in range(12, 30) if rule.push(float(i), True, 1.0).fire]
    assert fired and fired[0] >= 12 + 8 - 1  # the dwell restarted after the suspend


def test_dwell_clamp_reads_the_window_period_not_the_embed_time():
    """I2: dwell ≥ 2 × p95 of diff(embed.ts) — the live window period, which includes baselines and SSE."""
    ts = [0.0, 1.0, 2.0, 3.0, 9.0, 10.0, 11.0, 18.0]  # windows with occasional long periods
    p50, p95 = window_period(ts)
    assert p95 > 5.0
    assert clamp_dwell(5, p95) == pytest.approx(2 * p95)
    assert clamp_dwell(120, p95) == 120
    caps = Captions(5)
    tw = Tripwire("01", "X", "s", "plug_off", 5, 60, caps.pos, caps.neg, Threshold("labelled-m", 0.0, version=1))
    tw.arm()
    r = tw.step(100.0, caps.window("pos"), stride_s=1.0, period_p95=p95)
    assert r.dwell_s == pytest.approx(2 * p95)


def test_window_period_needs_two_windows():
    assert window_period([1.0]) == (None, None)


def test_shared_rule_replay_equals_live():
    """The bench's baselines run through replay_rule; it must decide exactly like the live DecisionRule."""
    rng = np.random.default_rng(0)
    flags = list(rng.random(300) < 0.7)
    ts = [float(i) for i in range(300)]
    live, rule = [], DecisionRule(20, 30, 0.8)
    for t, f in zip(ts, flags, strict=True):
        d = rule.push(t, f, 1.0)
        if d.fire:
            live.append(t)
            rule.start_cooldown(t)
    assert replay_rule(ts, flags, 20, 30, 1.0) == live


def test_replay_pause_after_trip_until_operator_on():
    ts = [float(i) for i in range(100)]
    flags = [True] * 100
    trips = replay_rule(ts, flags, 10, 600, 1.0, pause_after_trip=True, resume_at=[50.0])
    assert trips[0] == 7 and trips[1] == 57 and len(trips) == 2


def test_fresh_mode_needs_capture_then_arms_with_default_theta_z(tuning):
    caps = Captions(6)
    th = Threshold("fresh-z", None)
    tw = Tripwire("03", "DOOR", "the door is left open", "notify", 10, 60, caps.pos, caps.neg, th, tuning=tuning)
    tw.start_capture(0.0, 10)
    t = 0.0
    while tw.state == "capturing":
        tw.step(t, caps.window("neg"), stride_s=1.0, period_p95=1.0)
        t += 1.0
    assert tw.state == "armed" and th.theta == tuning.theta_z and th.mu0 is not None and th.sigma0 >= 0.01
    assert len(tw.capture_margins) >= 10  # the capture clock starts at the first window


def test_noise_floor_rejects_examples_that_look_normal(tuning):
    caps = Captions(8)
    th = Threshold("fresh-z", None)
    tw = Tripwire("04", "X", "s", "notify", 10, 60, caps.pos, caps.neg, th, tuning=tuning)
    tw.start_capture(0.0, 10)
    for i in range(12):
        tw.step(float(i), caps.window("neg"), stride_s=1.0, period_p95=1.0)
    normal_like = [margin(caps.window("neg"), caps.pos, caps.neg) for _ in range(3)]
    event_like = [margin(caps.window("pos"), caps.pos, caps.neg) for _ in range(3)]
    assert not tw.noise_floor_ok(normal_like) and tw.noise_floor_ok(event_like)


def test_zscore_none_without_reference():
    assert zscore(0.1, None, None) is None
    assert math.isclose(zscore(0.3, 0.1, 0.1), 2.0)
