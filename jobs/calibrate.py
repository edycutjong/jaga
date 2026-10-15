#!/usr/bin/env python3
"""jaga calibration — runs as a Nebius AI Cloud Serverless Job (cpu-d3, stock python:3.11, stdlib only).

Input  : labelled margins, one row per labelled item: item_id, tripwire, m, label (1 = event), source.
         Only margins and labels: no vectors, no frames (architecture: the one egress table).
         Read from JAGA_IN_URL (a presigned GET) or, if that fails, the injected file JAGA_IN_FILE.
Output : per-tripwire ROC summary and θ_m = the HIGHEST margin that keeps recall ≥ JAGA_RECALL (0.9),
         ties to the larger θ, written to JAGA_OUT_URL (a presigned PUT) and printed on one stdout line
         prefixed `JAGA_RESULT ` (readable afterwards with `nebius ai job logs <id>`).
The work is light. Running it off the laptop keeps the watcher from pausing and leaves a receipt per run.
"""
import csv
import hashlib
import io
import json
import math
import os
import platform
import sys
import time
import urllib.request

SCHEMA = "jaga.calibration.v1"


def read_input():
    url = os.environ.get("JAGA_IN_URL")
    path = os.environ.get("JAGA_IN_FILE", "/app/labelled_margins.csv")
    errors = []
    if url:
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read(), "presigned-get", errors
        except Exception as e:  # fall back to the injected copy
            errors.append(f"presigned GET failed: {type(e).__name__}")
    with open(path, "rb") as f:
        return f.read(), "inject-file", errors


def parse(data):
    """Rows with a finite margin and a 0/1 label; anything else is dropped and counted."""
    rows, dropped = [], 0
    for r in csv.DictReader(io.StringIO(data.decode("utf-8"))):
        try:
            m, y = float(r["m"]), int(r["label"])
        except (KeyError, ValueError, TypeError):
            dropped += 1
            continue
        if not math.isfinite(m) or y not in (0, 1):
            dropped += 1
            continue
        rows.append({"item_id": r["item_id"], "tripwire": r["tripwire"], "m": m, "label": y,
                     "source": r.get("source", "")})
    return rows, dropped


def auc(pos, neg):
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def hanley_mcneil(a, n1, n2, z=1.959963984540054):
    """95 % CI for an AUC; undefined (None) at AUC 0 or 1, where the formula's variance collapses to 0."""
    if a <= 1e-12 or a >= 1 - 1e-12:
        return None
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)
    se = math.sqrt(max(var, 0.0))
    return max(0.0, a - z * se), min(1.0, a + z * se)


def theta_at_recall(pos, recall):
    p = sorted(pos, reverse=True)
    need = max(1, math.ceil(recall * len(p) - 1e-9))
    return p[need - 1]


def confusion(pos, neg, theta):
    tp = sum(1 for x in pos if x >= theta)
    fp = sum(1 for x in neg if x >= theta)
    return {"tp": tp, "fp": fp, "fn": len(pos) - tp, "tn": len(neg) - fp,
            "precision": None if tp + fp == 0 else tp / (tp + fp), "recall": tp / len(pos) if pos else None}


def roc(pos, neg, max_points=64):
    th = sorted(set(pos + neg), reverse=True)
    if len(th) > max_points:
        step = len(th) / max_points
        th = [th[int(i * step)] for i in range(max_points)]
    pts = []
    for t in th:
        tpr = sum(1 for x in pos if x >= t) / len(pos)
        fpr = sum(1 for x in neg if x >= t) / len(neg)
        pts.append([round(t, 6), round(fpr, 4), round(tpr, 4)])
    return pts


def summarise(pos, neg, recall):
    out = {"n_pos": len(pos), "n_neg": len(neg)}
    if not pos:
        out["status"] = "no positives"
        return out
    theta = theta_at_recall(pos, recall)
    out.update(theta_m=theta, at_recall=recall, **confusion(pos, neg, theta))  # unrounded: the boundary clip stays in
    if neg:
        a = auc(pos, neg)
        ci = hanley_mcneil(a, len(pos), len(neg))
        out.update(auc=round(a, 4), auc_ci95=None if ci is None else [round(ci[0], 4), round(ci[1], 4)],
                   roc=roc(pos, neg))
        if ci is None:
            out["auc_ci95_note"] = "undefined at AUC 0 or 1 (Hanley-McNeil variance is 0)"
    out["status"] = "ok"
    return out


def calibrate(rows, recall):
    by = {}
    for r in rows:
        by.setdefault(r["tripwire"], ([], []))[0 if r["label"] == 1 else 1].append(r["m"])
    res = {tw: summarise(pos, neg, recall) for tw, (pos, neg) in sorted(by.items())}
    allp = [r["m"] for r in rows if r["label"] == 1]
    alln = [r["m"] for r in rows if r["label"] != 1]
    return res, (summarise(allp, alln, recall) if allp else {"status": "no positives"})


def put(url, body):
    req = urllib.request.Request(url, data=body, method="PUT", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status


def main():
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))
    data, transport, errors = read_input()
    rows, dropped = parse(data)
    recall = float(os.environ.get("JAGA_RECALL", "0.9"))
    per, pooled = calibrate(rows, recall)
    result = {"schema": SCHEMA, "kind": os.environ.get("JAGA_RUN_KIND", "calibration"),
              "run_id": os.environ.get("JAGA_RUN_ID", ""), "input_sha256": hashlib.sha256(data).hexdigest(),
              "input_bytes": len(data), "n_rows": len(rows), "dropped": dropped, "transport_in": transport,
              "transport_errors": errors,
              "recall_target": recall, "tripwires": per, "global_theta": pooled, "python": sys.version.split()[0],
              "platform": platform.platform(), "cpu_count": os.cpu_count(), "started_at": started,
              "elapsed_s": round(time.time() - t0, 3)}
    line = json.dumps(result, separators=(",", ":"))
    out = os.environ.get("JAGA_OUT_URL")
    if out:
        try:
            result["put_status"] = put(out, json.dumps(result, indent=1).encode())
        except Exception as e:
            result["put_status"] = f"failed: {type(e).__name__}"
        line = json.dumps(result, separators=(",", ":"))
    print("JAGA_RESULT " + line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
