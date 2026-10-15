"""The decision rule is the one function that must never be wrong, so it is verified over its whole small input space.

A trip switches mains power, so the rule (k-of-n windows over a dwell, with a cool-down and a suspend) is checked
two ways, neither of which loads a model or a camera:

* exhaustively: every possible above/below pattern of 12 consecutive windows, for 4 strides x 3 dwells, against an
  independent integer-arithmetic oracle (EXHAUSTIVE_CASES sequences, the number the README quotes);
* by property: random jittered timestamps, random suspends, random cool-downs (hypothesis, derandomized).
"""
import itertools
from fractions import Fraction

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jaga.score import MIN_WINDOWS, DecisionRule

LENGTH = 12
STRIDES = (1.0, 1.25, 2.0, 5.0)  # 5.0 s makes the dwell hold only 0-2 windows: the case where one window could trip
DWELLS = (4.0, 6.0, 10.0)
EXHAUSTIVE_CASES = len(STRIDES) * len(DWELLS) * 2**LENGTH  # 49,152


def oracle_kn(dwell: float, stride: float) -> tuple[int, int]:
    n = max(2, int(Fraction(dwell).limit_denominator(1000) / Fraction(stride).limit_denominator(1000)))
    k = max(2, -(-4 * n // 5))  # ceil(0.8 n) in integers
    return k, n


def oracle_fires(flags, stride: float, dwell: float, cooldown: float) -> list[int]:
    """Plain-Python restatement of the rule: which window indices trip."""
    k, _ = oracle_kn(dwell, stride)
    fired, kept, cool_until = [], [], float("-inf")
    for i, above in enumerate(flags):
        t = i * stride
        kept = [(u, a) for u, a in kept if u > t - dwell] + [(t, above)]
        if above and sum(a for _, a in kept) >= k and t >= cool_until:
            fired.append(i)
            kept, cool_until = [], t + cooldown
    return fired


def run_rule(flags, stride: float, dwell: float, cooldown: float) -> list[int]:
    rule = DecisionRule(dwell, cooldown)
    out = []
    for i, above in enumerate(flags):
        d = rule.push(i * stride, above, stride)
        if d.fire:
            out.append(i)
            rule.start_cooldown(i * stride)
    return out


def test_every_window_pattern_matches_the_oracle_and_one_window_never_trips():
    cases = 0
    for stride, dwell in itertools.product(STRIDES, DWELLS):
        cooldown = 3 * stride
        for flags in itertools.product((False, True), repeat=LENGTH):
            got = run_rule(flags, stride, dwell, cooldown)
            assert got == oracle_fires(flags, stride, dwell, cooldown), (stride, dwell, flags)
            if sum(flags) < 2:
                assert got == [], (stride, dwell, flags)  # I2: fewer than two windows above θ never trips
            for i in got:
                assert flags[i], (stride, dwell, flags)  # a trip lands on a window that is itself above θ
            cases += 1
    assert cases == EXHAUSTIVE_CASES


def test_an_all_below_stream_never_trips_whatever_the_dwell_and_stride():
    for stride, dwell in itertools.product(STRIDES + (0.5, 3.0), DWELLS + (1.0, 60.0)):
        assert run_rule((False,) * 200, stride, dwell, 3 * stride) == []


@settings(max_examples=400, derandomize=True, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    steps=st.lists(st.tuples(st.floats(0.2, 4.0), st.booleans()), min_size=1, max_size=80),
    dwell=st.floats(0.5, 120.0),
    cooldown=st.floats(0.0, 30.0),
    stride=st.floats(0.2, 4.0),
    suspend_at=st.lists(st.integers(0, 79), max_size=3),
    suspend_s=st.floats(0.5, 60.0),
)
def test_no_input_stream_trips_inside_a_suspend_a_cooldown_or_on_one_window(
        steps, dwell, cooldown, stride, suspend_at, suspend_s):
    rule = DecisionRule(dwell, cooldown)
    t, last_fire, suspended_until = 0.0, None, float("-inf")
    for i, (gap, above) in enumerate(steps):
        t += gap
        if i in suspend_at:
            rule.suspend(t, suspend_s)
            suspended_until = max(suspended_until, t + suspend_s)
        d = rule.push(t, above, stride)
        assert d.k >= MIN_WINDOWS and d.n >= d.k
        if d.fire:
            assert above
            assert d.count >= d.k >= MIN_WINDOWS
            assert t >= suspended_until
            assert last_fire is None or t >= last_fire + cooldown
            last_fire = t
            rule.start_cooldown(t)
