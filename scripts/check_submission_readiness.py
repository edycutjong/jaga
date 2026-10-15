#!/usr/bin/env python3
"""Is the repo ready to submit? Fails loudly on anything a judge would trip over.

Checks (each prints PASS or FAIL with the reason):
  readme        README.md has the required elements: setup instructions, how the NVIDIA models are used, where
                Token Factory is used, the other Nebius services used, and the licence
  cosmos        "Built on NVIDIA Cosmos" appears in README.md and NOTICE (NVIDIA Open Model License §3.2)
  license       LICENSE is the Apache License 2.0 (OSI-approved)
  placeholders  no <bench:…> / <traction:…> tokens, TODO or TBD left in judge-facing files
  numbers       DEMO.md was written by bench.py, and no bench section is older than the code it measured
  video         a video URL (YouTube) is linked from README.md (or --submission file)
  kitchen       no working-notes or key files tracked in git
  tests         the test count README.md states is the number pytest collects

    python scripts/check_submission_readiness.py [--submission <submission-text.md>] [--allow video]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JUDGE_FILES = ["README.md", "DEMO.md", "ARCHITECTURE.md", "docs/PROTOCOL.md", "docs/RUNS.md", "docs/DX-REPORT.md"]
SCORING_CODE = ["jaga/score.py", "jaga/embed.py", "jaga/frames.py", "jaga/offline.py", "jaga/stats.py",
                "scripts/bench.py"]
# the code each measurement depends on: (files, (file, function) pairs). A section is stale when any of it changed
# after the section's commit. Speed and period measure the embedder and the live loop, not the scoring scripts.
SECTION_CODE = {
    "speed": (["jaga/embed.py", "jaga/frames.py", "jaga/config.py"], [("scripts/bench.py", "cmd_speed")]),
    "period": (["jaga/daemon.py", "jaga/capture.py", "jaga/embed.py", "jaga/frames.py", "jaga/score.py",
                "jaga/guards.py", "jaga/config.py", "jaga/server.py"], [("scripts/bench.py", "cmd_period")]),
}
KITCHEN = re.compile(r"spec|progress|predict|radar|CLAUDE|AGENTS|\.claude|\.env$|pem$|a2a|autopilot|devices\.json"
                     r"|tinytuya\.json|snapshot\.json|tuya-raw\.json|consent", re.I)  # + the tinytuya wizard's files
README_TESTS = re.compile(r"pytest\s+#\s*(\d+) tests")
README_ELEMENTS = {
    "setup instructions": re.compile(r"pip install", re.I),
    "how the NVIDIA models are used": re.compile(r"Cosmos-Embed1.*?(Nemotron|Super)|(Nemotron|Super).*?Cosmos-Embed1",
                                                 re.I | re.S),
    "where Token Factory is used": re.compile(r"Token Factory", re.I),
    "other Nebius services": re.compile(r"Serverless Job", re.I),
    "Nebius Object Storage": re.compile(r"Object Storage", re.I),
    "licence named": re.compile(r"Apache", re.I),
}
VIDEO = re.compile(r"https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]{6,}")
PLACEHOLDER = re.compile(r"<bench:[^>]*>|<traction:[^>]*>|\bTODO\b|\bTBD\b")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def git_ok(*args: str) -> tuple[bool, str]:
    p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return p.returncode == 0, p.stdout


def check(submission: Path | None, allow: set[str]) -> list[tuple[str, bool, str]]:
    res = []
    readme = (REPO / "README.md").read_text() if (REPO / "README.md").exists() else ""
    missing = [k for k, rx in README_ELEMENTS.items() if not rx.search(readme)]
    res.append(("readme", not missing and bool(readme), "missing: " + ", ".join(missing) if missing else "all present"))

    notice = (REPO / "NOTICE").read_text() if (REPO / "NOTICE").exists() else ""
    ok = "Built on NVIDIA Cosmos" in readme and "Built on NVIDIA Cosmos" in notice
    res.append(("cosmos", ok, "found in README.md and NOTICE" if ok else "add \"Built on NVIDIA Cosmos\""))

    lic = (REPO / "LICENSE").read_text() if (REPO / "LICENSE").exists() else ""
    ok = "Apache License" in lic and "Version 2.0" in lic
    res.append(("license", ok, "Apache-2.0" if ok else "LICENSE is missing or not Apache-2.0"))

    hits = []
    files = [REPO / f for f in JUDGE_FILES] + ([submission] if submission else [])
    for f in files:
        if f.exists():
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if PLACEHOLDER.search(line):
                    hits.append(f"{f.name}:{i}")
    res.append(("placeholders", not hits, ", ".join(hits[:12]) if hits else "none"))

    demo = (REPO / "DEMO.md").read_text() if (REPO / "DEMO.md").exists() else ""
    summary = REPO / "bench" / "out" / "summary.json"
    stale = []
    if summary.exists():
        for name, sec in json.loads(summary.read_text()).items():
            c = sec.get("_commit")
            if sec.get("status") == "not yet run":  # no number in it, so nothing can be stale
                continue
            if not c or c == "unknown":  # a result that cannot say which code made it is never current
                stale.append(f"{name}@{c or 'no commit'}")
                continue
            if c.endswith("+dirty"):  # measured on uncommitted code: never current
                stale.append(f"{name}@{c}")
                continue
            files, funcs = SECTION_CODE.get(name, (SCORING_CODE, []))
            ok, out = git_ok("diff", "--name-only", c, "HEAD", "--", *files)
            changed = (not ok) or bool(out.strip())  # an unknown commit fails closed
            for f, fn in funcs:
                if changed:
                    break
                ok, out = git_ok("log", "-L", f":{fn}:{f}", f"{c}..HEAD", "-s", "--format=%h")
                changed = (not ok) or bool(out.strip())  # a renamed function fails closed
            if changed:
                stale.append(f"{name}@{c}")
    ok = "written by `scripts/bench.py`" in demo and not stale
    res.append(("numbers", ok, ("stale: " + ", ".join(stale)) if stale else
                ("DEMO.md from bench.py" if ok else "DEMO.md missing or not generated by bench.py")))

    text = readme + ((submission.read_text()) if submission and submission.exists() else "")
    ok = bool(VIDEO.search(text))
    res.append(("video", ok or "video" in allow, "linked" if ok else ("not linked yet (allowed)" if "video" in allow
                                                                      else "no YouTube link yet")))

    tracked = [f for f in git("ls-files").splitlines() if KITCHEN.search(f)]
    res.append(("kitchen", not tracked, ", ".join(tracked[:8]) if tracked else "clean"))

    m = README_TESTS.search(readme)
    n = collected_tests()
    ok = bool(m) and n is not None and int(m.group(1)) == n
    res.append(("tests", ok, f"README says {m.group(1) if m else '—'}, pytest collects {n}"))
    return res


def collected_tests() -> int | None:
    p = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=REPO,
                       capture_output=True, text=True, timeout=300)
    ids = [ln for ln in p.stdout.splitlines() if "::" in ln]
    if ids:
        return len(ids)
    per_file = [int(m.group(1)) for m in re.finditer(r"^tests/\S+\.py: (\d+)$", p.stdout, re.M)]
    return sum(per_file) if per_file else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", type=Path, default=None)
    ap.add_argument("--allow", action="append", default=[], choices=["video"])
    a = ap.parse_args()
    res = check(a.submission, set(a.allow))
    for name, ok, why in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name:13s} {why}")
    bad = [r for r in res if not r[1]]
    print(f"\n{len(res) - len(bad)}/{len(res)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
