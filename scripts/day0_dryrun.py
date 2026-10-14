#!/usr/bin/env python3
"""Day-0 stage G ($0): `nebius ai job create --dry-run` only — nothing is created, nothing is billed.

Probes the two GPU fallback options (recorded, never run) and the smallest `--disk-size` the CPU
calibration Job accepts. A dry-run also passes through the tenant's quota check, so a full region
answers ResourceExhausted; that answer is recorded as-is.

Usage:  python scripts/day0_dryrun.py --out evidence/day0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EU_WEST1 = {"project": "project-e01phrcvpa003r6v2aep14", "subnet": "vpcsubnet-e01mdp8yxx4t2hccw5"}
EU_NORTH1 = {"project": "project-e00cvdxypr00ke3z0eq068", "subnet": "vpcsubnet-e00bqeygmz5mevv1we"}

PROBES = [
    ("gpu-l40s-a 1gpu-8vcpu-32gb, eu-north1", EU_NORTH1, "gpu-l40s-a", "1gpu-8vcpu-32gb", None),
    ("gpu-h200-sxm 1gpu-16vcpu-200gb, eu-west1", EU_WEST1, "gpu-h200-sxm", "1gpu-16vcpu-200gb", None),
] + [
    (f"cpu-d3 4vcpu-16gb disk {d}, eu-west1", EU_WEST1, "cpu-d3", "4vcpu-16gb", d)
    for d in ("1Gi", "8Gi", "16Gi", "32Gi", "64Gi", "128Gi")
]


def dry_run(where: dict, platform: str, preset: str, disk: str | None) -> dict:
    cmd = ["nebius", "ai", "job", "create", "--dry-run", "--no-browser",
           "--parent-id", where["project"], "--subnet-id", where["subnet"],
           "--platform", platform, "--preset", preset, "--image", "python:3.11", "--timeout", "1h",
           "--name", "jaga-dry-run", "--container-command", "python3", "--args", "-c print(1)"]
    if disk:
        cmd += ["--disk-size", disk]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    text = (p.stdout + p.stderr).strip()
    first = text.splitlines()[0] if text else ""
    if "validation succeeded" in text:
        verdict = "validated"
    elif "ResourceExhausted" in text or "Quota" in text:
        verdict = "quota"
    else:
        verdict = "rejected"
    return {"rc": p.returncode, "verdict": verdict, "first_line": first[:300],
            "detail": [ln for ln in text.splitlines()[1:6] if ln.strip()]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evidence/day0")
    args = ap.parse_args()
    rows = []
    for label, where, platform, preset, disk in PROBES:
        r = dry_run(where, platform, preset, disk)
        rows.append({"probe": label, "platform": platform, "preset": preset, "disk_size": disk, **r})
        print(f"{r['verdict']:9s} {label}: {r['first_line'][:110]}")
    cpu_ok = [r["disk_size"] for r in rows if r["platform"] == "cpu-d3" and r["verdict"] == "validated"]
    ev = {"stage": "G", "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "dry_run_only": True,
          "smallest_cpu_disk_validated": cpu_ok[0] if cpu_ok else None, "probes": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "g-stage.json").write_text(json.dumps(ev, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
