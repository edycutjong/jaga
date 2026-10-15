"""The sponsor-integration receipts: Token Factory request IDs and Nebius Serverless Job IDs.

jaga has no hosted URL (it runs on the laptop), so the receipts live in the local overlay (`/receipts`),
in `docs/RUNS.md` and in `evidence/jobs/*.json`. Nothing on this page is typed by hand: it is read from
the database and from the evidence files the dispatcher writes.
"""
from __future__ import annotations

import html
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO / "evidence" / "jobs"
BENCH_SUMMARY = REPO / "bench" / "out" / "summary.json"


def job_runs() -> list[dict]:
    if not JOBS.exists():
        return []
    runs = []
    for p in sorted(JOBS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        d["_file"] = str(p.relative_to(REPO))
        runs.append(d)
    runs.sort(key=lambda d: d.get("created_at") or "")
    return runs


def footer_line() -> str:
    runs = [r for r in job_runs() if r.get("state") == "COMPLETED"]
    if not runs:
        return "θ: Nebius Job — none yet"
    r = runs[-1]
    kind = r.get("kind", "calibration")
    day = (r.get("created_at") or "")[:10]
    return f"θ: Nebius Job {r.get('job_id')} · {day}" + (" (smoke run)" if kind == "smoke" else "")


def collect(watcher) -> dict:
    tws = []
    for slot in sorted(watcher.tripwires):
        tw = watcher.tripwires[slot]
        meta = watcher.meta.get(slot, {})
        th = tw.threshold
        tws.append({"slot": slot, "name": tw.name, "sentence": tw.sentence, "action": tw.action,
                    "tf_request_id": meta.get("tf_request_id"), "tf_response_id": meta.get("tf_response_id"),
                    "model": meta.get("model"), "compiled": meta.get("created"), "temperature": meta.get("temperature"),
                    "threshold_mode": th.mode, "theta": th.theta, "theta_version": th.version,
                    "theta_source": th.source, "state": tw.state})
    bench = None
    if BENCH_SUMMARY.exists():
        try:
            bench = json.loads(BENCH_SUMMARY.read_text())
        except json.JSONDecodeError:
            bench = None
    info = {}
    try:
        info = watcher.embedder.info
    except Exception:
        pass
    return {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tripwires": tws,
            "compiles": watcher.db.recent_compiles(12), "jobs": job_runs()[-12:], "bench": bench, "model": info}


def _e(x) -> str:
    return html.escape("—" if x is None or x == "" else str(x))


def _usd(x) -> str:
    return "—" if x is None else f"${x:.6f}"


def _theta(t: dict) -> str:
    out = str(t["threshold_mode"])
    if t["theta"] is not None:
        out += f" {t['theta']:.4f}"
    if t["theta_version"] is not None:
        out += f" v{t['theta_version']}"
    return html.escape(out)


def render_html(d: dict) -> str:
    rows_tw = "".join(
        f"<tr><td class=slot>{_e(t['slot'])}</td><td>{_e(t['name'])}<br><span class=muted>{_e(t['sentence'])}</span>"
        f"</td><td class=num>{_e(t['tf_request_id'])}</td><td>{_e(t['model'])}</td>"
        f"<td class=num>{_theta(t)}</td></tr>" for t in d["tripwires"])
    rows_c = "".join(
        f"<tr><td class=num>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(c['ts']))}</td><td>{_e(c['sentence'])}</td>"
        f"<td class=num>{_e(c['tf_request_id'])}</td><td>{'ok' if c['ok'] else 'failed'}"
        f"{' · ' + html.escape(str(c['detail'])) if c['detail'] else ''}</td></tr>"
        for c in d["compiles"])
    rows_j = "".join(
        f"<tr><td class=num>{_e(j.get('job_id'))}</td><td>{_e(j.get('kind'))}</td><td>{_e(j.get('state'))}</td>"
        f"<td class=num>{_e(j.get('created_at'))}</td><td class=num>{_e(j.get('wall_clock_s'))} s</td>"
        f"<td>{_e(j.get('platform'))} {_e(j.get('preset'))}</td><td class=num>{_usd(j.get('cost_usd_docs_price'))}</td></tr>"
        for j in reversed(d["jobs"]))
    m = d.get("model") or {}
    b = d.get("bench") or {}
    sp = (b.get("speed") or {}).get("rows") or []
    pe = b.get("period") or {}
    rows_b = "".join(f"<tr><td>{_e(r['device']).upper()} fp32</td><td class=num>{_e(r['p50_s'])} s</td>"
                     f"<td class=num>{_e(r['p95_s'])} s</td><td class=num>{_e(r['n'])}</td></tr>" for r in sp)
    period = (f"<p class=muted>Live window period p50 {pe['period_p50_s']:.2f} s, p95 {pe['period_p95_s']:.2f} s over "
              f"{pe['windows']} windows ({_e(pe.get('input'))}).</p>") if pe.get("windows") else ""
    bench_html = (f"<section><h2>Measured · scripts/bench.py (DEMO.md)</h2><table><tr><th>device</th><th>p50 s/clip</th>"
                  f"<th>p95 s/clip</th><th>timed calls</th></tr>{rows_b}</table>{period}</section>") if rows_b else ""
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><title>jaga · receipts</title>
<link rel=stylesheet href="/static/overlay.css"></head><body class=receipts>
<header class=strip><span class=name>JAGA</span><span class=meta>receipts · generated {_e(d['generated_at'])}</span>
<a class=back href="/">← overlay</a></header>
<main class=rc>
<section><h2>Perception · NVIDIA Cosmos-Embed1</h2><p class=num>{_e(m.get('model'))} · rev {_e(m.get('revision'))} ·
{_e(m.get('device'))} {_e(m.get('dtype'))} · torch {_e(m.get('torch'))}</p></section>
<section><h2>Compiled tripwires · Nemotron 3 Super on Nebius Token Factory</h2>
<table><tr><th>slot</th><th>tripwire</th><th>TF request ID</th><th>model</th><th>threshold</th></tr>{rows_tw}</table></section>
<section><h2>Recent compiles</h2><table><tr><th>time</th><th>sentence</th><th>TF request ID</th><th>result</th></tr>{rows_c}
</table></section>
<section><h2>Calibration · Nebius AI Cloud Serverless Jobs</h2>
<table><tr><th>job ID</th><th>kind</th><th>state</th><th>created</th><th>wall clock</th><th>compute</th><th>cost (docs price)</th></tr>
{rows_j or '<tr><td colspan=7 class=muted>no Job run yet</td></tr>'}</table>
<p class=muted>Every run is also listed in docs/RUNS.md with its cost at both prices.</p></section>
{bench_html}
</main><footer class=foot>Built on NVIDIA Cosmos · video frames never leave this laptop</footer></body></html>"""
