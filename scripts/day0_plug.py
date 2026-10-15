#!/usr/bin/env python3
"""Day-0 stage PLUG: the real plug switches the real lamp, and how fast it acknowledges.

An operator script (a person runs it with the lamp in view): N cycles of OFF (timed until the plug confirms
OFF, the latency a trip pays) then ON through jaga/operator_on.py (the operator path; no trip can do this).
Needs the plug paired and ~/.config/jaga/devices.json (Tuya, from `python -m tinytuya wizard`) or shelly.json.

    python scripts/day0_plug.py --cycles 20 --out evidence/day0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jaga.act import ActionFailed, NullPlug, load_driver  # noqa: E402
from jaga.operator_on import switch_on  # noqa: E402


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between switches (let the lamp settle)")
    ap.add_argument("--out", default="evidence/day0")
    a = ap.parse_args()
    d = load_driver()
    if isinstance(d, NullPlug):
        print(f"no plug configured: {d.reason}")
        return 2
    off_ms, on_ms, failures = [], [], []
    initial = d.status()
    for i in range(a.cycles):
        try:
            off_ms.append(d.turn_off().latency_ms)
        except ActionFailed as e:
            failures.append({"cycle": i + 1, "call": "off", "why": str(e)})
        time.sleep(a.pause)
        try:
            on_ms.append(switch_on(d).latency_ms)
        except ActionFailed as e:
            failures.append({"cycle": i + 1, "call": "on", "why": str(e)})
        time.sleep(a.pause)
        print(f"cycle {i + 1}/{a.cycles}: off {off_ms[-1] if off_ms else '—'} ms · on {on_ms[-1] if on_ms else '—'} ms",
              flush=True)
    ev = {"stage": "PLUG", "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "driver": d.describe(),
          "initial_on": initial, "cycles": a.cycles, "off_ok": len(off_ms), "on_ok": len(on_ms), "failures": failures,
          "off_ack_ms": {"p50": round(statistics.median(off_ms), 1), "p95": round(pct(off_ms, 0.95), 1),
                         "max": round(max(off_ms), 1)} if off_ms else None,
          "on_ack_ms": {"p50": round(statistics.median(on_ms), 1), "p95": round(pct(on_ms, 0.95), 1)} if on_ms else None,
          "note": "OFF latency = from the call until the plug confirms OFF (what a trip pays before the overlay shows OFF)",
          "pass": len(off_ms) == a.cycles}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plug-stage.json").write_text(json.dumps(ev, indent=2) + "\n")
    print(json.dumps({k: ev[k] for k in ("off_ok", "on_ok", "off_ack_ms", "pass")}))
    return 0 if ev["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
