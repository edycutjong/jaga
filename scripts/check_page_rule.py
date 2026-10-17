#!/usr/bin/env python3
"""The landing page's "play the rule" widget runs a JavaScript port of DecisionRule.push. This script checks that the
port trips exactly where the Python rule does on every pattern of tests/test_decision_exhaustive.py (4 strides × 3
dwells × 2^12 window patterns, cool-down 3 strides).

It takes the port from docs/index.html itself (between the page-rule markers), runs it with Node, and compares.

    python scripts/check_page_rule.py          # needs node on PATH; exit 0 = every pattern matches
"""
from __future__ import annotations

import itertools
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from jaga.score import DecisionRule  # noqa: E402

LENGTH, STRIDES, DWELLS = 12, (1.0, 1.25, 2.0, 5.0), (4.0, 6.0, 10.0)

HARNESS = """
const R = require(process.argv[1]);
const S = [1.0, 1.25, 2.0, 5.0], D = [4.0, 6.0, 10.0], L = 12, out = [];
for (const s of S) for (const d of D) for (let m = 0; m < (1 << L); m++) {
  const flags = []; for (let i = 0; i < L; i++) flags.push(((m >> (L - 1 - i)) & 1) === 1);
  out.push(R.run(flags, s, d, 3 * s).fired);
}
process.stdout.write(JSON.stringify(out));
"""


def python_trips(flags, stride: float, dwell: float, cooldown: float) -> list[int]:
    rule, out = DecisionRule(dwell, cooldown), []
    for i, above in enumerate(flags):
        if rule.push(i * stride, above, stride).fire:
            out.append(i)
            rule.start_cooldown(i * stride)
    return out


def main() -> int:
    node = shutil.which("node")
    if not node:
        print("node is not on PATH — nothing checked", file=sys.stderr)
        return 2
    page = (REPO / "docs/index.html").read_text(encoding="utf-8")
    m = re.search(r"/\* page-rule:start \*/(.*?)/\* page-rule:end \*/", page, re.S)
    if not m:
        print("no page-rule block in docs/index.html", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as d:
        port = Path(d) / "rule.js"
        port.write_text(m.group(1), encoding="utf-8")
        js = json.loads(subprocess.run([node, "-e", HARNESS, str(port)], capture_output=True, text=True,
                                       check=True).stdout)
    i = 0
    for s in STRIDES:
        for dw in DWELLS:
            for flags in itertools.product((False, True), repeat=LENGTH):
                want = python_trips(flags, s, dw, 3 * s)
                if js[i] != want:
                    print(f"MISMATCH at stride {s} dwell {dw} flags {flags}: page {js[i]} python {want}")
                    return 1
                i += 1
    print(f"the page's rule matches DecisionRule on all {i:,} patterns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
