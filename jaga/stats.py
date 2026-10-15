"""The statistics bench.py publishes. Small n is the norm here, so every rate carries an exact interval.

* AUC          — Mann–Whitney, ties count half; 95 % CI by Hanley & McNeil (1982)
* proportions  — Clopper–Pearson exact 95 % interval
* rates        — exact Poisson (Garwood) 95 % interval per unit of exposure
* paired claims— exact two-sided sign test on discordant items only (no claim from overlapping CIs)
"""
from __future__ import annotations

import math

import numpy as np

Z95 = 1.959963984540054


def auc(pos, neg) -> float:
    p = np.asarray(pos, dtype=np.float64).reshape(-1, 1)
    n = np.asarray(neg, dtype=np.float64).reshape(1, -1)
    if p.size == 0 or n.size == 0:
        raise ValueError("AUC needs at least one positive and one negative")
    return float(((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def hanley_mcneil(a: float, n_pos: int, n_neg: int, z: float = Z95) -> tuple[float, float]:
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return max(0.0, a - z * se), min(1.0, a + z * se)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta

    if n <= 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def poisson_rate(k: int, exposure: float, alpha: float = 0.05) -> tuple[float, float, float]:
    """(rate, lo, hi) per unit of exposure (e.g. false alarms per armed hour)."""
    from scipy.stats import chi2

    if exposure <= 0:
        raise ValueError("exposure must be positive")
    lo = 0.0 if k == 0 else float(chi2.ppf(alpha / 2, 2 * k) / 2)
    hi = float(chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2)
    return k / exposure, lo / exposure, hi / exposure


def sign_test(b: int, c: int) -> float:
    """Exact two-sided p for b vs c discordant pairs (H0: each discordant pair is a fair coin)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def confusion(pos_scores, neg_scores, theta: float) -> dict:
    p = np.asarray(pos_scores, dtype=np.float64)
    n = np.asarray(neg_scores, dtype=np.float64)
    tp, fn = int((p >= theta).sum()), int((p < theta).sum())
    fp, tn = int((n >= theta).sum()), int((n < theta).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": None if tp + fp == 0 else tp / (tp + fp),
            "recall": None if tp + fn == 0 else tp / (tp + fn)}


def median(values) -> float | None:
    a = np.asarray(list(values), dtype=np.float64)
    return None if a.size == 0 else float(np.median(a))
