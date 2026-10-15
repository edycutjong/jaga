"""Dispatch the calibration Job on Nebius AI Cloud: `nebius ai job create` (cpu-d3 · 4vcpu-16gb · python:3.11).

The laptop builds the labelled margins (labelled clips, "✗ wrong" marks, unscored sessions — never a scored
soak), uploads them to Object Storage, and hands the Job one presigned GET and one presigned PUT, so the Job
never holds a credential. `jobs/calibrate.py` (stdlib) is injected into a stock image; the same CSV is also
injected (≤ 64 KiB) as a fallback. The result comes back from the bucket, or from `nebius ai job logs`.
Every run writes evidence/jobs/<job-id>.json and a row in docs/RUNS.md: job ID, wall clock, cost at both prices.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from . import sync

REPO = Path(__file__).resolve().parents[1]
JOB_SCRIPT = REPO / "jobs" / "calibrate.py"
SMOKE_FIXTURE = REPO / "fixtures" / "test" / "smoke_margins.csv"
RUNS_MD = REPO / "docs" / "RUNS.md"
EVIDENCE = REPO / "evidence" / "jobs"

PROJECT = os.environ.get("JAGA_NEBIUS_PROJECT", "project-e01phrcvpa003r6v2aep14")  # eu-west1
SUBNET = os.environ.get("JAGA_NEBIUS_SUBNET", "vpcsubnet-e01mdp8yxx4t2hccw5")
PLATFORM, PRESET, IMAGE, DISK, TIMEOUT = "cpu-d3", "4vcpu-16gb", "python:3.11", "16Gi", "1h"
REGION = "eu-west1"
# $/hour for 4 vCPU + 16 GiB: the docs list price (4×0.012 + 16×0.0032)
# and a higher price seen earlier (4×0.015 + 16×0.0045)
PRICE_DOCS, PRICE_HIGH = 0.0992, 0.132
INJECT_LIMIT = 64 * 1024
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "CANCELED", "ERROR", "DELETED", "TERMINATED", "STOPPED"}


# ------------------------------------------------------------------------------------- input rows
def clip_rows() -> list[dict]:
    """Labelled clip margins with the frozen tripwires (needs fixtures/clips.csv + fixtures/vecs)."""
    from .offline import CLIPS_CSV, committed_vector
    from .score import margin
    from .tripwires import load, load_vectors

    if not CLIPS_CSV.exists():
        return []
    rows, cache = [], {}
    with CLIPS_CSV.open() as f:
        for r in csv.DictReader(f):
            slot = r["tripwire"].zfill(2)
            if r.get("heldout") in ("1", "true"):
                continue  # held-out day-2 clips never calibrate
            try:
                if slot not in cache:
                    cache[slot] = load_vectors(load(slot))
            except FileNotFoundError:
                continue
            vec = committed_vector(r["clip_id"])
            if vec is None:
                continue
            pos, neg = cache[slot]
            rows.append({"item_id": r["clip_id"], "tripwire": slot, "m": f"{margin(vec, pos, neg):.6f}",
                         "label": int(r["label"]), "source": "clip"})
    return rows


def wrong_rows(db) -> list[dict]:
    """A trip the user marked "✗ wrong" is a labelled negative at its trip margin (unscored sessions only)."""
    q = ("SELECT e.id, t.slot, w.margin FROM event e JOIN tripwire t ON t.id = e.tripwire_id "
         "JOIN session s ON s.name = e.session JOIN win w ON w.tripwire_id = e.tripwire_id AND w.ts = e.ts "
         "WHERE e.kind = 'trip' AND e.wrong = 1 AND s.scored = 0")
    return [{"item_id": f"wrong-{r['id']}", "tripwire": r["slot"], "m": f"{r['margin']:.6f}", "label": 0,
             "source": "wrong-mark"} for r in db.query(q)]


def labelled_rows(db=None) -> list[dict]:
    rows = clip_rows()
    if db is not None:
        rows += wrong_rows(db)
    return rows


# ------------------------------------------------------------------------------------- nebius CLI
def nebius(args: list[str], timeout: float = 180) -> tuple[int, str]:
    p = subprocess.run(["nebius", *args], capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr).strip()


def create_job(name: str, env: dict, inject: list[tuple[Path, str]], dry_run: bool = False,
               quota_wait_s: float = 1800) -> tuple[str | None, str]:
    args = ["ai", "job", "create", "--no-browser", "--parent-id", PROJECT, "--subnet-id", SUBNET,
            "--name", name, "--platform", PLATFORM, "--preset", PRESET, "--image", IMAGE, "--disk-size", DISK,
            "--timeout", TIMEOUT, "--container-command", "python3", "--args", "/app/calibrate.py", "--async"]
    for local, remote in inject:
        args += ["--inject-file", f"{local}:{remote}"]
    for k, v in env.items():
        args += ["--env", f"{k}={v}"]
    if dry_run:
        args.append("--dry-run")
    deadline = time.monotonic() + quota_wait_s
    while True:
        rc, out = nebius(args, timeout=300)
        if dry_run:
            return None, _redact(out)
        m = re.search(r"aijob-[a-z0-9]+", out)
        if rc == 0 and m:
            return m.group(0), _redact(out)
        if ("ResourceExhausted" in out or "uota" in out) and time.monotonic() < deadline:
            time.sleep(60)  # the region's 2-instance quota is shared with other projects: wait, never run 2
            continue
        return None, _redact(out)


def _redact(text: str) -> str:
    """Presigned URLs carry signatures: never let one reach a log or an evidence file."""
    return re.sub(r"https?://\S*X-Amz-[^\s\"']*", "<presigned-url>", text)


def wait_job(job_id: str, timeout_s: float = 2400) -> dict:
    deadline = time.monotonic() + timeout_s
    last = {}
    while time.monotonic() < deadline:
        rc, out = nebius(["ai", "job", "get", job_id, "--format", "json"], timeout=120)
        if rc == 0:
            try:
                last = json.loads(out[out.index("{"):])
            except (ValueError, json.JSONDecodeError):
                last = {}
            if (last.get("status") or {}).get("state") in TERMINAL:
                return last
        time.sleep(15)
    return last


def result_from_logs(job_id: str) -> dict | None:
    rc, out = nebius(["ai", "job", "logs", job_id, "--since", "6h"], timeout=120)
    for line in reversed(out.splitlines()):
        if line.startswith("JAGA_RESULT "):
            try:
                return json.loads(line[len("JAGA_RESULT "):])
            except json.JSONDecodeError:
                return None
    return None


def _ts(s: str | None) -> float | None:
    if not s:
        return None
    s = re.sub(r"(\.\d{6})\d*", r"\1", s.replace("Z", "+00:00"))
    from datetime import datetime

    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


# ------------------------------------------------------------------------------------- dispatch
def dispatch(smoke: bool = False, input_csv: str | None = None, wait: bool = True, dry_run: bool = False,
             db=None) -> int:
    kind = "smoke" if smoke else "calibration"
    if smoke:
        data = SMOKE_FIXTURE.read_bytes()
        n_rows = sum(1 for _ in csv.DictReader(io.StringIO(data.decode())))
    elif input_csv:
        data = Path(input_csv).read_bytes()
        n_rows = sum(1 for _ in csv.DictReader(io.StringIO(data.decode())))
    else:
        if db is None:
            from .db import DB

            db = DB()
        rows = labelled_rows(db)
        if not rows:
            print("no labelled margins yet (no clips in fixtures/clips.csv, no ✗ wrong marks) — "
                  "use --smoke for a plumbing run, which is not a calibration")
            return 2
        data = sync.margins_csv(rows)
        n_rows = len(rows)
    if len(data) > INJECT_LIMIT:
        raise SystemExit(f"labelled margins are {len(data)} B > 64 KiB inject limit")
    run_id = time.strftime("%Y%m%dT%H%M%S")
    env = {"JAGA_RUN_KIND": kind, "JAGA_RUN_ID": run_id, "JAGA_IN_FILE": "/app/labelled_margins.csv"}
    transport = "inject-file + logs"
    loc = sync.resolve_location()
    client = None
    if loc and not dry_run:
        try:
            client = sync.s3_client()
            key_in, key_out = f"runs/{run_id}/labelled_margins.csv", f"runs/{run_id}/thresholds.json"
            client.put_object(Bucket=loc.bucket, Key=loc.key(key_in), Body=data, ContentType="text/csv")
            env["JAGA_IN_URL"] = sync.presign(client, "GET", loc, key_in, 7200)
            env["JAGA_OUT_URL"] = sync.presign(client, "PUT", loc, key_out, 7200)
            transport = f"object storage {loc.bucket}/{loc.prefix}runs/{run_id}/ (presigned GET + PUT)"
        except Exception as e:  # no key / no bucket: fall back to inject + logs (spec D0 stage S3)
            print(f"object storage unavailable ({type(e).__name__}); using inject-file + logs")
            client = None
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "labelled_margins.csv"
        csv_path.write_bytes(data)
        job_id, out = create_job(f"jaga-calibrate-{kind}-{run_id.lower()}", env,
                                 [(JOB_SCRIPT, "/app/calibrate.py"), (csv_path, "/app/labelled_margins.csv")],
                                 dry_run=dry_run)
    if dry_run:
        print(out)
        return 0 if "succeeded" in out else 1
    if not job_id:
        print(f"Job create failed:\n{out[:1200]}")
        return 1
    print(f"created {job_id} ({kind}) · {PLATFORM} {PRESET} · {REGION}")
    if not wait:
        return 0
    info = wait_job(job_id)
    st = info.get("status") or {}
    state = st.get("state")
    result, result_via = None, None
    if client and state == "COMPLETED":
        try:
            body = client.get_object(Bucket=loc.bucket, Key=loc.key(f"runs/{run_id}/thresholds.json"))["Body"].read()
            result, result_via = json.loads(body), "object storage (presigned PUT by the Job)"
        except Exception:
            result = None
    if result is None:
        result = result_from_logs(job_id)
        result_via = "nebius ai job logs" if result else None
    t_start, t_end = _ts(st.get("started_at")), _ts(st.get("finished_at"))
    wall = round(t_end - t_start, 1) if t_start and t_end else None
    created = (info.get("metadata") or {}).get("created_at")
    ev = {"job_id": job_id, "kind": kind, "state": state, "run_id": run_id, "region": REGION, "platform": PLATFORM,
          "preset": PRESET, "image": IMAGE, "disk_size": DISK, "created_at": created,
          "started_at": st.get("started_at"),
          "finished_at": st.get("finished_at"), "wall_clock_s": wall, "transport": transport, "result_via": result_via,
          "input_rows": n_rows, "input_bytes": len(data), "input_sha256": hashlib.sha256(data).hexdigest(),
          "input": "fixtures/test/smoke_margins.csv (synthetic test fixture — plumbing only)" if smoke else
          (input_csv or "labelled clip margins + ✗ wrong marks (unscored sessions only)"),
          "cost_usd_docs_price": None if wall is None else round(wall / 3600 * PRICE_DOCS, 6),
          "cost_usd_high_price": None if wall is None else round(wall / 3600 * PRICE_HIGH, 6),
          "billed_usd": None, "billed_note": "read the billed amount from the Nebius console (the formula has no disk "
                                              "or provisioning term)", "result": result}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / f"{job_id}.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False) + "\n")
    append_runs_md(ev)
    if kind == "calibration" and state == "COMPLETED" and result:
        store_thresholds(job_id, result, db)
    print(f"{job_id}: {state} · wall {wall} s · result via {result_via}")
    return 0 if state == "COMPLETED" and result else 1


def append_runs_md(ev: dict) -> None:
    RUNS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not RUNS_MD.exists():
        RUNS_MD.write_text(RUNS_HEADER)
    res = ev.get("result") or {}
    tws = res.get("tripwires") or {}
    summary = "; ".join(f"{k}: θ_m {v.get('theta_m')} AUC {v.get('auc')}" for k, v in tws.items()) or "—"
    kind = "**smoke, not a calibration**" if ev["kind"] == "smoke" else "calibration"
    row = (f"| {(ev.get('created_at') or '')[:19].replace('T', ' ')} | `{ev['job_id']}` | {kind} | {ev['state']} | "
           f"{ev['wall_clock_s']} | {ev['cost_usd_docs_price']} | {ev['cost_usd_high_price']} | "
           f"{ev['input_rows']} rows · sha256 {ev['input_sha256'][:12]} | {summary} | {ev['result_via']} |\n")
    with RUNS_MD.open("a") as f:
        f.write(row)


RUNS_HEADER = """# Calibration runs — Nebius AI Cloud Serverless Jobs

Every row is written by `jaga calibrate` (`jaga/calibrate.py`) from the Job's own status and output; the
evidence file for each run is `evidence/jobs/<job-id>.json`. Compute: `cpu-d3` · `4vcpu-16gb` · stock
`python:3.11` · `--disk-size 16Gi` (the smallest size the API accepted on day 0) · region eu-west1. Input: labelled margins
only (no vectors, no frames). Cost = container wall clock × hourly price for 4 vCPU + 16 GiB, at the docs list
price ($0.0992/h) and at a higher price seen earlier ($0.132/h); the billed amount is read from the console.

A **smoke** run checks the plumbing on a committed synthetic test fixture. It is not a calibration and its
numbers mean nothing about any tripwire.

| created (UTC) | job ID | kind | state | wall clock (s) | cost @ $0.0992/h | cost @ $0.132/h | input | result | result read via |
|---|---|---|---|---|---|---|---|---|---|
"""


def store_thresholds(job_id: str, result: dict, db=None) -> None:
    if db is None:
        from .db import DB

        db = DB()
    for slot, r in (result.get("tripwires") or {}).items():
        if r.get("status") != "ok":
            continue
        tw = db.tripwire(slot)
        if not tw:
            continue
        prev = db.query("SELECT MAX(version) AS v FROM threshold WHERE tripwire_id=?", (tw["id"],))[0]["v"] or 0
        db.add_threshold(tw["id"], int(prev) + 1, "labelled-m", r["theta_m"], None, None, f"nebius-job:{job_id}",
                         None, r.get("recall"))
