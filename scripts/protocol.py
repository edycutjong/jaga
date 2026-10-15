#!/usr/bin/env python3
"""Generate the pre-registered Soak B schedules (docs/protocol/<session>.csv) from a fixed seed.

The order and length of every still episode, exit and activity is drawn here, not chosen during the
session. Re-running with the same seed reproduces the committed files byte for byte (tests check it).

    python scripts/protocol.py B1 B2
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "protocol"
SEEDS = {"B1": "jaga-soak-B1-v1", "B2": "jaga-soak-B2-v1"}
N_STILL, N_LEAVE, N_INVIEW = 10, 4, 2  # per session: ≥ 10 still episodes, ≥ 6 exits (≥ 2 in view)
MIN_TOTAL_MIN = 120.0

ACTIVITIES = {
    "ironing": "iron the shirt as usual (the iron prop stays unplugged; the lamp is the load)",
    "folding": "fold clothes at the board",
    "phone": "scroll your phone, standing at the board",
    "reading": "read at the board",
}


def generate(session: str) -> list[dict]:
    rng = random.Random(SEEDS[session])
    events = ["still"] * N_STILL + ["exit_leave"] * N_LEAVE + ["exit_inview"] * N_INVIEW
    rng.shuffle(events)
    steps = [{"kind": "capture", "minutes": 2, "text": "iron as usual: this is the 2-minute normal-state capture"}]
    for ev in events:
        act = rng.choice(sorted(ACTIVITIES))
        steps.append({"kind": f"activity:{act}", "minutes": rng.choice([3, 4, 5, 6]), "text": ACTIVITIES[act]})
        if ev == "still":
            pose = rng.choice(["stand", "sit"])
            steps.append({"kind": "still", "minutes": rng.choice([3, 4, 5]),
                          "text": f"{pose} still at the board: no ironing, no phone, hands still"})
        elif ev == "exit_leave":
            steps.append({"kind": "exit_leave", "minutes": rng.choice([4, 5, 6]),
                          "text": "leave the room and stay away; when you come back, if the lamp is off, "
                                  "press the plug's own button"})
        else:
            steps.append({"kind": "exit_inview", "minutes": rng.choice([4, 5, 6]),
                          "text": "step at least 2 metres away from the board and stay in view; when you come "
                                  "back, if the lamp is off, press the plug's own button"})
    steps.append({"kind": "activity:ironing", "minutes": 3, "text": ACTIVITIES["ironing"]})
    t, rows = 0.0, []
    for i, s in enumerate(steps):
        rows.append({"step": i, "start_min": f"{t:.1f}", "end_min": f"{t + s['minutes']:.1f}", "kind": s["kind"],
                     "text": s["text"]})
        t += s["minutes"]
    while t < MIN_TOTAL_MIN:  # top up with activities until the session is ≥ 2 h
        act = rng.choice(sorted(ACTIVITIES))
        rows.append({"step": len(rows), "start_min": f"{t:.1f}", "end_min": f"{t + 5:.1f}",
                     "kind": f"activity:{act}", "text": ACTIVITIES[act]})
        t += 5
    return rows


def write(session: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{session}.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "start_min", "end_min", "kind", "text"], lineterminator="\n")
        w.writeheader()
        w.writerows(generate(session))
    return p


def summary(session: str) -> dict:
    rows = generate(session)
    kinds = [r["kind"] for r in rows]
    return {"session": session, "seed": SEEDS[session], "steps": len(rows), "minutes": float(rows[-1]["end_min"]),
            "still": kinds.count("still"), "exit_leave": kinds.count("exit_leave"),
            "exit_inview": kinds.count("exit_inview")}


if __name__ == "__main__":
    for s in sys.argv[1:] or list(SEEDS):
        print(write(s), summary(s))
