"""The published statistics (jaga/stats.py) and the stdlib calibration Job (jobs/calibrate.py)."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from jaga import stats as S
from jaga.sync import MARGIN_COLUMNS, margins_csv

REPO = Path(__file__).resolve().parents[1]
JOB = REPO / "jobs" / "calibrate.py"


def test_auc_counts_ties_as_half():
    assert S.auc([3, 2], [1, 0]) == 1.0
    assert S.auc([1], [1]) == 0.5
    assert S.auc([2, 1], [1, 0]) == pytest.approx(0.875)


def test_hanley_mcneil_interval_contains_auc_and_is_clipped():
    lo, hi = S.hanley_mcneil(0.9, 10, 10)
    assert lo < 0.9 < hi <= 1.0
    lo, hi = S.hanley_mcneil(1.0, 10, 10)
    assert hi == 1.0


def test_clopper_pearson_known_values():
    lo, hi = S.clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-4)
    lo, hi = S.clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.6915, abs=1e-4)
    lo, hi = S.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-4) and hi == pytest.approx(0.8129, abs=1e-4)


def test_poisson_rate_interval():
    rate, lo, hi = S.poisson_rate(0, 8.0)
    assert rate == 0 and lo == 0 and hi == pytest.approx(3.6889 / 8, abs=1e-3)


def test_sign_test_exact():
    assert S.sign_test(0, 0) == 1.0
    assert S.sign_test(0, 6) == pytest.approx(2 / 64)
    assert S.sign_test(3, 3) == 1.0


def test_job_is_stdlib_only_and_fits_the_inject_limit():
    tree = ast.parse(JOB.read_text())
    mods = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= set(sys.stdlib_module_names), mods - set(sys.stdlib_module_names)
    assert JOB.stat().st_size <= 64 * 1024


def run_job(tmp_path, csv_text):
    f = tmp_path / "m.csv"
    f.write_text(csv_text)
    env = {k: v for k, v in os.environ.items() if not k.startswith("JAGA_")}
    env.update({"JAGA_IN_FILE": str(f), "JAGA_RUN_KIND": "smoke"})
    p = subprocess.run([sys.executable, "-I", str(JOB)], capture_output=True, text=True, env=env, timeout=60)
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith("JAGA_RESULT "))
    return json.loads(line[len("JAGA_RESULT "):])


def test_job_theta_at_recall_and_counts(tmp_path):
    smoke = (REPO / "fixtures" / "test" / "smoke_margins.csv").read_text()
    r = run_job(tmp_path, smoke)
    t = r["tripwires"]["00"]
    assert r["kind"] == "smoke" and r["n_rows"] == 10 and r["transport_in"] == "inject-file"
    assert t["n_pos"] == 5 and t["n_neg"] == 5
    assert t["theta_m"] == pytest.approx(0.0187)  # the 5th-highest positive keeps recall ≥ 0.9 → all 5
    assert t["tp"] == 5 and t["fp"] == 1 and t["precision"] == pytest.approx(5 / 6)
    assert 0 <= t["auc"] <= 1 and len(t["auc_ci95"]) == 2


def test_job_matches_the_laptop_statistics(tmp_path):
    from jaga.score import theta_at_recall

    pos = [0.31, 0.22, 0.18, 0.05, 0.12, 0.27, 0.09, 0.15, 0.20, 0.11]
    neg = [0.02, -0.04, 0.10, 0.06, -0.10, 0.01, 0.03, 0.12, -0.02, 0.00]
    rows = "item_id,tripwire,m,label,source\n" + "".join(f"p{i},01,{m},1,clip\n" for i, m in enumerate(pos)) + \
        "".join(f"n{i},01,{m},0,clip\n" for i, m in enumerate(neg))
    r = run_job(tmp_path, rows)["tripwires"]["01"]
    assert r["theta_m"] == pytest.approx(theta_at_recall(pos, 0.9))
    assert r["auc"] == pytest.approx(round(S.auc(pos, neg), 4))
    lo, hi = S.hanley_mcneil(S.auc(pos, neg), 10, 10)
    assert r["auc_ci95"] == pytest.approx([round(lo, 4), round(hi, 4)])


def test_sync_refuses_columns_outside_the_egress_table():
    ok = margins_csv([{"item_id": "a", "tripwire": "01", "m": 0.1, "label": 1, "source": "clip"}])
    assert ok.decode().splitlines()[0] == ",".join(MARGIN_COLUMNS)
    with pytest.raises(ValueError):
        margins_csv([{"item_id": "a", "tripwire": "01", "m": 0.1, "label": 1, "source": "clip", "vec": "…"}])
