#!/usr/bin/env python3
"""Do the decision-rule tests actually notice a broken rule?

Copies the package to a temp directory, breaks `DecisionRule` / the k-of-n floor one way at a time, and runs the rule's
tests (`tests/test_decision_exhaustive.py` + `tests/test_score.py`) against the broken copy (PYTHONPATH points at the
copy; the repo is never touched). A mutant the tests miss is reported as SURVIVED, with the reason when it cannot change
the rule's behaviour at the shipped settings.

    python scripts/mutation_check.py          # exit 0 only if every non-equivalent mutant is killed
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILE = "jaga/score.py"
# name, old, new, why it cannot change behaviour at the shipped settings (None = the tests must kill it)
MUTANTS = [
    ("one window may trip (MIN_WINDOWS 2 -> 1)", "MIN_WINDOWS = 2", "MIN_WINDOWS = 1", None),
    ("k rounds down (ceil -> floor)", "int(math.ceil(self.k_fraction * n - 1e-9))",
     "int(math.floor(self.k_fraction * n - 1e-9))", None),
    ("k fraction 0.8 -> 0.5", "k_fraction: float = TUNING.k_fraction):\n        if dwell_s <= 0:",
     "k_fraction: float = 0.5):\n        if dwell_s <= 0:", None),
    ("dwell window keeps its oldest sample (<= -> <)", "self.history[0][0] <= ts - self.dwell_s",
     "self.history[0][0] < ts - self.dwell_s", None),
    ("cool-down ignored", "fire = newest and count >= k and ts >= self.cooldown_until",
     "fire = newest and count >= k", None),
    ("suspend ignored", "        if ts < self.suspended_until:\n            k, n = self.kn(stride_s)\n"
     "            return Decision(False, 0, k, n, False)\n", "", None),
    ("newest window not required", "fire = newest and count >= k and ts >= self.cooldown_until",
     "fire = count >= k and ts >= self.cooldown_until", None),
    ("k floor MIN_WINDOWS dropped", "k = max(MIN_WINDOWS, int(math.ceil(self.k_fraction * n - 1e-9)))",
     "k = int(math.ceil(self.k_fraction * n - 1e-9))",
     "n is already floored at MIN_WINDOWS = 2, so ceil(0.8 n) >= 2 at the shipped k fraction 0.8 without the floor"),
]


def run(workdir: Path) -> int:
    env = {**os.environ, "PYTHONPATH": str(workdir)}
    cmd = [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
           "tests/test_decision_exhaustive.py", "tests/test_score.py"]
    return subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True).returncode


def main() -> int:
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        shutil.copytree(REPO / "jaga", work / "jaga", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "tests", work / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(REPO / "pyproject.toml", work / "pyproject.toml")
        original = (work / FILE).read_text()
        print(f"unmutated: {'pass' if run(work) == 0 else 'FAIL'}")
        for name, old, new, why in MUTANTS:
            if old not in original:
                print(f"ERROR   {name}: pattern not found in {FILE}")
                bad += 1
                continue
            (work / FILE).write_text(original.replace(old, new, 1))
            killed = run(work) != 0
            (work / FILE).write_text(original)
            if killed:
                print(f"KILLED   {name}")
            elif why:
                print(f"SURVIVED {name} -- equivalent: {why}")
            else:
                print(f"SURVIVED {name} -- NOT explained")
                bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
